//! Shared memory transport with synchronized in-process ring buffers.

use crate::{
    protocol::{BinaryCodec, Message, MessageCodec},
    Transport, TransportConfig, TransportError, TransportStats,
};
use async_trait::async_trait;
use crossbeam::channel::{bounded, unbounded, Receiver, Sender};
use dashmap::DashMap;
use std::{
    mem::size_of,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info};

#[cfg(not(target_arch = "wasm32"))]
use shared_memory::{Shmem, ShmemConf};

/// Shared memory segment info
#[derive(Debug, Clone)]
pub struct SharedMemoryInfo {
    pub name: String,
    pub size: usize,
    pub ring_buffer_size: usize,
}

/// Bounded multi-producer, multi-consumer ring buffer.
///
/// Payload bytes and queue metadata share one mutex, so checking capacity,
/// copying a frame and publishing it form a single atomic operation.
pub struct RingBuffer {
    state: parking_lot::Mutex<RingState>,
}

struct RingState {
    buffer: Vec<u8>,
    head: usize,
    tail: usize,
    size: usize,
}

impl RingBuffer {
    /// Create a buffer with the given byte capacity, including frame prefixes.
    pub fn new(capacity: usize) -> Self {
        Self {
            state: parking_lot::Mutex::new(RingState {
                buffer: vec![0; capacity],
                head: 0,
                tail: 0,
                size: 0,
            }),
        }
    }

    /// Write one complete frame, or leave the buffer unchanged when full.
    pub fn write(&self, data: &[u8]) -> Result<(), TransportError> {
        let mut state = self.state.lock();
        let capacity = state.buffer.len();
        let available = capacity - state.size;
        let total_len = data.len().checked_add(size_of::<u32>());
        if data.len() > u32::MAX as usize || total_len.map_or(true, |len| len > available) {
            return Err(TransportError::MessageTooLarge {
                size: total_len.unwrap_or(usize::MAX),
                max: available,
            });
        }

        let mut position = state.tail;
        // At most two bulk copies per slice, including when the frame wraps.
        for bytes in [&(data.len() as u32).to_le_bytes()[..], data] {
            let first = bytes.len().min(capacity - position);
            state.buffer[position..position + first].copy_from_slice(&bytes[..first]);
            state.buffer[..bytes.len() - first].copy_from_slice(&bytes[first..]);
            position = (position + bytes.len()) % capacity;
        }
        state.tail = position;
        state.size += total_len.unwrap();
        Ok(())
    }

    /// Read and remove one complete frame.
    pub fn read(&self) -> Option<Vec<u8>> {
        let mut state = self.state.lock();
        if state.size < size_of::<u32>() {
            return None;
        }
        let capacity = state.buffer.len();
        let mut position = state.head;
        let mut len_bytes = [0; size_of::<u32>()];
        for byte in &mut len_bytes {
            *byte = state.buffer[position];
            position = (position + 1) % capacity;
        }
        let data_len = u32::from_le_bytes(len_bytes) as usize;
        let total_len = data_len.checked_add(size_of::<u32>())?;
        if total_len > state.size {
            return None;
        }

        let mut data = vec![0; data_len];
        let first = data_len.min(capacity - position);
        data[..first].copy_from_slice(&state.buffer[position..position + first]);
        data[first..].copy_from_slice(&state.buffer[..data_len - first]);
        state.head = (position + data_len) % capacity;
        state.size -= total_len;
        Some(data)
    }

    /// Get the remaining byte capacity, including frame prefixes.
    pub fn available_space(&self) -> usize {
        let state = self.state.lock();
        state.buffer.len() - state.size
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.state.lock().size == 0
    }
}

/// Shared memory transport implementation
pub struct SharedMemoryTransport {
    info: SharedMemoryInfo,
    config: TransportConfig,
    codec: Arc<dyn MessageCodec>,
    local_id: String,
    peers: Arc<DashMap<String, Arc<RingBuffer>>>,
    incoming_rx: mpsc::Receiver<(String, Message)>,
    incoming_tx: mpsc::Sender<(String, Message)>,
    is_running: Arc<AtomicBool>,
    stats: Arc<RwLock<TransportStats>>,
    #[cfg(not(target_arch = "wasm32"))]
    shmem: Option<Arc<parking_lot::Mutex<Shmem>>>,
}

// SAFETY: SharedMemoryTransport is safe to send between threads because:
// 1. All fields except shmem are Send/Sync by default:
//    - Arc<DashMap> is Send/Sync
//    - mpsc channels are Send/Sync
//    - AtomicBool is Send/Sync
//    - RwLock is Send/Sync
// 2. shmem field is protected by Arc<parking_lot::Mutex<>> which provides thread-safe access
// 3. The shared memory (Shmem) itself is a memory-mapped file descriptor that can be
//    safely accessed from multiple threads when properly synchronized
// 4. All shared memory operations go through the Mutex, preventing data races
// 5. Ring buffer payloads and metadata are protected by one mutex
//
// SAFETY: SharedMemoryTransport is safe for shared access (Sync) because:
// 1. All mutable state is protected by synchronization primitives (Mutex, RwLock, Atomic)
// 2. The DashMap provides concurrent access to peer mappings
// 3. Ring buffers synchronize head/tail pointers with their payload bytes
// 4. No interior mutability is exposed without proper synchronization
unsafe impl Send for SharedMemoryTransport {}
unsafe impl Sync for SharedMemoryTransport {}

impl SharedMemoryTransport {
    /// Create a new shared memory transport
    pub async fn new(
        info: SharedMemoryInfo,
        config: TransportConfig,
    ) -> Result<Self, TransportError> {
        let (incoming_tx, incoming_rx) = mpsc::channel(1024);

        #[cfg(not(target_arch = "wasm32"))]
        let shmem = Some(Arc::new(parking_lot::Mutex::new(
            Self::create_or_open_shmem(&info)?,
        )));

        let transport = Self {
            info,
            config,
            codec: Arc::new(BinaryCodec),
            local_id: uuid::Uuid::new_v4().to_string(),
            peers: Arc::new(DashMap::new()),
            incoming_rx,
            incoming_tx,
            is_running: Arc::new(AtomicBool::new(true)),
            stats: Arc::new(RwLock::new(TransportStats::default())),
            #[cfg(not(target_arch = "wasm32"))]
            shmem,
        };

        transport.start_polling();

        Ok(transport)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn create_or_open_shmem(info: &SharedMemoryInfo) -> Result<Shmem, TransportError> {
        match ShmemConf::new().size(info.size).flink(&info.name).create() {
            Ok(shmem) => {
                info!("Created shared memory segment: {}", info.name);
                Ok(shmem)
            }
            Err(_) => {
                // Try to open existing
                ShmemConf::new().flink(&info.name).open().map_err(|e| {
                    TransportError::Other(anyhow::anyhow!("Failed to open shared memory: {}", e))
                })
            }
        }
    }

    /// Start polling for messages
    fn start_polling(&self) {
        let peers = Arc::clone(&self.peers);
        let incoming_tx = self.incoming_tx.clone();
        let is_running = Arc::clone(&self.is_running);
        let codec = Arc::clone(&self.codec);
        let stats = Arc::clone(&self.stats);
        let max_message_size = self.config.max_message_size;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(1));

            while is_running.load(Ordering::SeqCst) {
                interval.tick().await;

                // Snapshot handles so no DashMap shard guard lives across an await.
                let buffers: Vec<_> = peers
                    .iter()
                    .map(|peer| (peer.key().clone(), Arc::clone(peer.value())))
                    .collect();
                for (peer_id, buffer) in buffers {
                    // Read messages from buffer
                    while let Some(data) = buffer.read() {
                        if data.len() > max_message_size {
                            stats.write().await.errors += 1;
                            continue;
                        }
                        match codec.decode(&data) {
                            Ok(msg) => {
                                // Update stats
                                {
                                    let mut stats = stats.write().await;
                                    stats.messages_received += 1;
                                    stats.bytes_received += data.len() as u64;
                                    stats.last_activity = Some(chrono::Utc::now());
                                }

                                // Forward message
                                if incoming_tx.send((peer_id.clone(), msg)).await.is_err() {
                                    error!("Failed to forward message from shared memory");
                                }
                            }
                            Err(e) => {
                                error!("Failed to decode message from shared memory: {}", e);
                                stats.write().await.errors += 1;
                            }
                        }
                    }
                }
            }
        });
    }

    /// Register a peer with a ring buffer
    pub fn register_peer(&self, peer_id: String, buffer: Arc<RingBuffer>) {
        self.peers.insert(peer_id.clone(), buffer);
        info!("Registered peer: {}", peer_id);
    }

    /// Create a ring buffer for a peer
    pub fn create_buffer(&self) -> Arc<RingBuffer> {
        Arc::new(RingBuffer::new(self.info.ring_buffer_size))
    }
}

#[cfg(target_arch = "wasm32")]
impl SharedMemoryTransport {
    /// WASM-specific implementation using SharedArrayBuffer
    pub async fn new_wasm(
        buffer: js_sys::SharedArrayBuffer,
        config: TransportConfig,
    ) -> Result<Self, TransportError> {
        use wasm_bindgen::JsCast;

        let (incoming_tx, incoming_rx) = mpsc::channel(1024);

        // Create ring buffer backed by SharedArrayBuffer
        let buffer_size = buffer.byte_length() as usize;
        let info = SharedMemoryInfo {
            name: "wasm_shared_memory".to_string(),
            size: buffer_size,
            ring_buffer_size: buffer_size / 4, // Use 1/4 for each direction
        };

        let transport = Self {
            info,
            config,
            codec: Arc::new(BinaryCodec),
            local_id: uuid::Uuid::new_v4().to_string(),
            peers: Arc::new(DashMap::new()),
            incoming_rx,
            incoming_tx,
            is_running: Arc::new(AtomicBool::new(true)),
            stats: Arc::new(RwLock::new(TransportStats::default())),
        };

        transport.start_polling();

        Ok(transport)
    }
}

#[async_trait]
impl Transport for SharedMemoryTransport {
    type Message = Message;
    type Error = TransportError;

    async fn send(&self, to: &str, msg: Self::Message) -> Result<(), Self::Error> {
        // Encode message
        let data = self.codec.encode(&msg)?;
        if data.len() > self.config.max_message_size {
            return Err(TransportError::MessageTooLarge {
                size: data.len(),
                max: self.config.max_message_size,
            });
        }

        // Find peer buffer
        let buffer = self.peers.get(to).map(|entry| Arc::clone(entry.value()));
        if let Some(buffer) = buffer {
            // Write to buffer
            buffer.write(&data)?;

            // Update stats
            let mut stats = self.stats.write().await;
            stats.messages_sent += 1;
            stats.bytes_sent += data.len() as u64;
            stats.last_activity = Some(chrono::Utc::now());

            Ok(())
        } else {
            Err(TransportError::ConnectionError(format!(
                "No shared memory buffer for peer: {}",
                to
            )))
        }
    }

    async fn receive(&mut self) -> Result<(String, Self::Message), Self::Error> {
        self.incoming_rx
            .recv()
            .await
            .ok_or_else(|| TransportError::ConnectionError("Channel closed".to_string()))
    }

    async fn broadcast(&self, msg: Self::Message) -> Result<(), Self::Error> {
        let data = self.codec.encode(&msg)?;
        if data.len() > self.config.max_message_size {
            return Err(TransportError::MessageTooLarge {
                size: data.len(),
                max: self.config.max_message_size,
            });
        }
        let mut errors = Vec::new();

        // Send to all peers
        for peer in self.peers.iter() {
            let (peer_id, buffer) = peer.pair();
            if let Err(e) = buffer.write(&data) {
                errors.push(format!("{}: {}", peer_id, e));
            }
        }

        if errors.is_empty() {
            // Update stats
            let mut stats = self.stats.write().await;
            stats.messages_sent += self.peers.len() as u64;
            stats.bytes_sent += data.len() as u64 * self.peers.len() as u64;
            stats.last_activity = Some(chrono::Utc::now());

            Ok(())
        } else {
            Err(TransportError::Other(anyhow::anyhow!(
                "Broadcast errors: {}",
                errors.join(", ")
            )))
        }
    }

    fn local_address(&self) -> Result<String, Self::Error> {
        Ok(format!("shm://{}#{}", self.info.name, self.local_id))
    }

    fn is_connected(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.is_running.store(false, Ordering::SeqCst);
        self.peers.clear();
        Ok(())
    }

    fn stats(&self) -> TransportStats {
        futures::executor::block_on(async { self.stats.read().await.clone() })
    }
}

/// Zero-copy message wrapper for shared memory
pub struct ZeroCopyMessage<'a> {
    data: &'a [u8],
    codec: &'a dyn MessageCodec,
}

impl<'a> ZeroCopyMessage<'a> {
    /// Create a new zero-copy message
    pub fn new(data: &'a [u8], codec: &'a dyn MessageCodec) -> Self {
        Self { data, codec }
    }

    /// Decode the message (performs allocation)
    pub fn decode(&self) -> Result<Message, TransportError> {
        self.codec.decode(self.data)
    }

    /// Get raw data without decoding
    pub fn raw_data(&self) -> &[u8] {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer() {
        let buffer = RingBuffer::new(1024);

        // Test write and read
        let data = b"Hello, World!";
        assert!(buffer.write(data).is_ok());

        let read_data = buffer.read().unwrap();
        assert_eq!(data, read_data.as_slice());

        // Test empty buffer
        assert!(buffer.is_empty());
        assert!(buffer.read().is_none());
    }

    #[test]
    fn test_ring_buffer_wrap_around() {
        let buffer = RingBuffer::new(64);

        // Fill buffer multiple times to test wrap-around
        for i in 0..10 {
            let data = format!("Message {}", i).into_bytes();
            buffer.write(&data).unwrap();
            let read_data = buffer.read().unwrap();
            assert_eq!(data, read_data);
        }
    }

    #[tokio::test]
    async fn test_shared_memory_transport() {
        let info = SharedMemoryInfo {
            name: "test_shmem".to_string(),
            size: 1024 * 1024,           // 1MB
            ring_buffer_size: 64 * 1024, // 64KB
        };

        let config = TransportConfig::default();
        let transport = SharedMemoryTransport::new(info, config).await;

        // Transport should be created successfully
        assert!(transport.is_ok());
    }
}
