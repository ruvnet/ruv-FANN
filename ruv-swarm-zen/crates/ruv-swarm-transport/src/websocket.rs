//! WebSocket transport implementation with auto-reconnection and TLS support

use crate::{
    protocol::{BinaryCodec, Message, MessageCodec},
    Transport, TransportConfig, TransportError, TransportStats,
};
use async_trait::async_trait;
use backoff::{future::retry, ExponentialBackoff};
use dashmap::DashMap;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use std::{
    io::{Read, Write},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{broadcast, mpsc, RwLock},
    time::timeout,
};
use tokio_tungstenite::{
    accept_async, connect_async,
    tungstenite::{Error as WsError, Message as WsMessage},
    MaybeTlsStream, WebSocketStream,
};
use tracing::{debug, error, info, warn};
use url::Url;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// WebSocket transport mode
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsMode {
    /// Client mode - connects to a server
    Client { url: String },
    /// Server mode - listens for connections
    Server { bind_addr: String },
}

/// WebSocket transport implementation
pub struct WebSocketTransport {
    mode: WsMode,
    config: TransportConfig,
    codec: Arc<dyn MessageCodec>,
    connections: Arc<DashMap<String, mpsc::Sender<Message>>>,
    incoming_rx: mpsc::Receiver<(String, Message)>,
    incoming_tx: mpsc::Sender<(String, Message)>,
    broadcast_tx: broadcast::Sender<Message>,
    is_running: Arc<AtomicBool>,
    stats: Arc<RwLock<TransportStats>>,
    reconnect_notify: Arc<broadcast::Sender<String>>,
}

impl WebSocketTransport {
    /// Create a new WebSocket transport
    pub async fn new(mode: WsMode, config: TransportConfig) -> Result<Self, TransportError> {
        let (incoming_tx, incoming_rx) = mpsc::channel(1024);
        let (broadcast_tx, _) = broadcast::channel(1024);
        let (reconnect_tx, _) = broadcast::channel(16);

        let transport = Self {
            mode,
            config,
            codec: Arc::new(BinaryCodec),
            connections: Arc::new(DashMap::new()),
            incoming_rx,
            incoming_tx,
            broadcast_tx,
            is_running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(TransportStats::default())),
            reconnect_notify: Arc::new(reconnect_tx),
        };

        transport.start().await?;
        Ok(transport)
    }

    /// Start the transport
    async fn start(&self) -> Result<(), TransportError> {
        self.is_running.store(true, Ordering::SeqCst);

        match &self.mode {
            WsMode::Client { url } => {
                self.start_client(url.clone()).await?;
            }
            WsMode::Server { bind_addr } => {
                self.start_server(bind_addr.clone()).await?;
            }
        }

        Ok(())
    }

    /// Start client mode
    async fn start_client(&self, url: String) -> Result<(), TransportError> {
        let connections = Arc::clone(&self.connections);
        let incoming_tx = self.incoming_tx.clone();
        let broadcast_tx = self.broadcast_tx.clone();
        let is_running = Arc::clone(&self.is_running);
        let config = self.config.clone();
        let codec = Arc::clone(&self.codec);
        let stats = Arc::clone(&self.stats);
        let reconnect_notify = Arc::clone(&self.reconnect_notify);

        tokio::spawn(async move {
            let backoff = ExponentialBackoff {
                max_elapsed_time: None,
                ..Default::default()
            };

            let _ = retry(backoff, || async {
                if !is_running.load(Ordering::SeqCst) {
                    return Ok(());
                }

                match Self::connect_with_retry(&url, &config).await {
                    Ok(ws_stream) => {
                        info!("Connected to WebSocket server: {}", url);
                        let _ = reconnect_notify.send(url.clone());

                        if let Err(e) = Self::handle_client_connection(
                            ws_stream,
                            url.clone(),
                            connections.clone(),
                            incoming_tx.clone(),
                            broadcast_tx.subscribe(),
                            codec.clone(),
                            config.clone(),
                            stats.clone(),
                            is_running.clone(),
                        )
                        .await
                        {
                            warn!("Client connection error: {}", e);
                            return Err(backoff::Error::transient(e));
                        }
                    }
                    Err(e) => {
                        error!("Failed to connect to {}: {}", url, e);
                        return Err(backoff::Error::transient(e));
                    }
                }

                Ok(())
            })
            .await;
        });

        Ok(())
    }

    /// Start server mode
    async fn start_server(&self, bind_addr: String) -> Result<(), TransportError> {
        let listener = TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| TransportError::ConnectionError(format!("Failed to bind: {}", e)))?;

        info!("WebSocket server listening on: {}", bind_addr);

        let connections = Arc::clone(&self.connections);
        let incoming_tx = self.incoming_tx.clone();
        let broadcast_tx = self.broadcast_tx.clone();
        let is_running = Arc::clone(&self.is_running);
        let config = self.config.clone();
        let codec = Arc::clone(&self.codec);
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            while is_running.load(Ordering::SeqCst) {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        let peer_addr = addr.to_string();
                        debug!("New connection from: {}", peer_addr);

                        let connections = connections.clone();
                        let incoming_tx = incoming_tx.clone();
                        let broadcast_tx = broadcast_tx.clone();
                        let codec = codec.clone();
                        let config = config.clone();
                        let stats = stats.clone();
                        let is_running = is_running.clone();

                        tokio::spawn(async move {
                            match accept_async(stream).await {
                                Ok(ws_stream) => {
                                    info!("WebSocket connection established: {}", peer_addr);

                                    if let Err(e) = Self::handle_server_connection(
                                        ws_stream,
                                        peer_addr,
                                        connections,
                                        incoming_tx,
                                        broadcast_tx.subscribe(),
                                        codec,
                                        config,
                                        stats,
                                        is_running,
                                    )
                                    .await
                                    {
                                        error!("Server connection error: {}", e);
                                    }
                                }
                                Err(e) => {
                                    error!("WebSocket handshake failed: {}", e);
                                }
                            }
                        });
                    }
                    Err(e) => {
                        error!("Failed to accept connection: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Connect with retry logic
    async fn connect_with_retry(
        url: &str,
        config: &TransportConfig,
    ) -> Result<WsStream, TransportError> {
        let url = Url::parse(url)
            .map_err(|e| TransportError::InvalidAddress(format!("Invalid URL: {}", e)))?;

        let duration = Duration::from_millis(config.connection_timeout_ms);

        match timeout(duration, connect_async(url.as_str())).await {
            Ok(Ok((ws_stream, _))) => Ok(ws_stream),
            Ok(Err(e)) => Err(TransportError::ConnectionError(format!(
                "WebSocket error: {}",
                e
            ))),
            Err(_) => Err(TransportError::Timeout),
        }
    }

    /// Handle client connection
    async fn handle_client_connection(
        ws_stream: WsStream,
        peer_addr: String,
        connections: Arc<DashMap<String, mpsc::Sender<Message>>>,
        incoming_tx: mpsc::Sender<(String, Message)>,
        broadcast_rx: broadcast::Receiver<Message>,
        codec: Arc<dyn MessageCodec>,
        config: TransportConfig,
        stats: Arc<RwLock<TransportStats>>,
        is_running: Arc<AtomicBool>,
    ) -> Result<(), TransportError> {
        Self::handle_connection(
            ws_stream,
            peer_addr,
            connections,
            incoming_tx,
            broadcast_rx,
            codec,
            config,
            stats,
            is_running,
        )
        .await
    }

    /// Handle server connection
    async fn handle_server_connection(
        ws_stream: WebSocketStream<TcpStream>,
        peer_addr: String,
        connections: Arc<DashMap<String, mpsc::Sender<Message>>>,
        incoming_tx: mpsc::Sender<(String, Message)>,
        broadcast_rx: broadcast::Receiver<Message>,
        codec: Arc<dyn MessageCodec>,
        config: TransportConfig,
        stats: Arc<RwLock<TransportStats>>,
        is_running: Arc<AtomicBool>,
    ) -> Result<(), TransportError> {
        // Handle the connection directly without type conversion
        Self::handle_raw_connection(
            ws_stream,
            peer_addr,
            connections,
            incoming_tx,
            broadcast_rx,
            codec,
            config,
            stats,
            is_running,
        )
        .await
    }

    /// Handle raw WebSocket connection (TcpStream)
    async fn handle_raw_connection(
        mut ws_stream: WebSocketStream<TcpStream>,
        peer_addr: String,
        connections: Arc<DashMap<String, mpsc::Sender<Message>>>,
        incoming_tx: mpsc::Sender<(String, Message)>,
        mut broadcast_rx: broadcast::Receiver<Message>,
        codec: Arc<dyn MessageCodec>,
        config: TransportConfig,
        stats: Arc<RwLock<TransportStats>>,
        is_running: Arc<AtomicBool>,
    ) -> Result<(), TransportError> {
        use futures_util::{SinkExt, StreamExt};

        let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<Message>(256);
        connections.insert(peer_addr.clone(), outgoing_tx);

        loop {
            if !is_running.load(Ordering::SeqCst) {
                break;
            }

            tokio::select! {
                // Handle incoming messages
                Some(result) = ws_stream.next() => {
                    match result {
                        Ok(WsMessage::Binary(data)) => {
                            // Decompress if needed
                            let data = if config.enable_compression && data.len() > config.compression_threshold {
                                Self::decompress(&data)?
                            } else {
                                data
                            };

                            // Decode message
                            match codec.decode(&data) {
                                Ok(msg) => {
                                    // Update stats
                                    {
                                        let mut stats = stats.write().await;
                                        stats.messages_received += 1;
                                        stats.bytes_received += data.len() as u64;
                                        stats.last_activity = Some(chrono::Utc::now());
                                    }

                                    // Forward to incoming channel
                                    if incoming_tx.send((peer_addr.clone(), msg)).await.is_err() {
                                        error!("Failed to forward incoming message");
                                        break;
                                    }
                                }
                                Err(e) => {
                                    error!("Failed to decode message: {}", e);
                                    stats.write().await.errors += 1;
                                }
                            }
                        }
                        Ok(WsMessage::Close(_)) => {
                            info!("Connection closed by peer: {}", peer_addr);
                            break;
                        }
                        Ok(WsMessage::Ping(data)) => {
                            if ws_stream.send(WsMessage::Pong(data)).await.is_err() {
                                break;
                            }
                        }
                        Ok(_) => {} // Ignore other message types
                        Err(e) => {
                            error!("WebSocket error: {}", e);
                            break;
                        }
                    }
                }

                // Handle outgoing messages
                Some(msg) = outgoing_rx.recv() => {
                    // Encode message
                    match codec.encode(&msg) {
                        Ok(mut data) => {
                            // Compress if needed
                            if config.enable_compression && data.len() > config.compression_threshold {
                                data = Self::compress(&data)?;
                            }

                            // Send message
                            if ws_stream.send(WsMessage::Binary(data.clone())).await.is_err() {
                                error!("Failed to send message");
                                break;
                            }

                            // Update stats
                            let mut stats = stats.write().await;
                            stats.messages_sent += 1;
                            stats.bytes_sent += data.len() as u64;
                            stats.last_activity = Some(chrono::Utc::now());
                        }
                        Err(e) => {
                            error!("Failed to encode message: {}", e);
                            stats.write().await.errors += 1;
                        }
                    }
                }

                // Handle broadcast messages
                Ok(msg) = broadcast_rx.recv() => {
                    // Skip if this message is not for us
                    if let Some(dest) = &msg.destination {
                        if dest != &peer_addr {
                            continue;
                        }
                    }

                    // Send broadcast message
                    match codec.encode(&msg) {
                        Ok(mut data) => {
                            if config.enable_compression && data.len() > config.compression_threshold {
                                data = Self::compress(&data)?;
                            }

                            if ws_stream.send(WsMessage::Binary(data)).await.is_err() {
                                error!("Failed to send broadcast");
                                break;
                            }
                        }
                        Err(e) => {
                            error!("Failed to encode broadcast: {}", e);
                        }
                    }
                }
            }
        }

        // Clean up connection
        connections.remove(&peer_addr);
        info!("Connection closed: {}", peer_addr);

        Ok(())
    }

    /// Common connection handler
    async fn handle_connection(
        mut ws_stream: WsStream,
        peer_addr: String,
        connections: Arc<DashMap<String, mpsc::Sender<Message>>>,
        incoming_tx: mpsc::Sender<(String, Message)>,
        mut broadcast_rx: broadcast::Receiver<Message>,
        codec: Arc<dyn MessageCodec>,
        config: TransportConfig,
        stats: Arc<RwLock<TransportStats>>,
        is_running: Arc<AtomicBool>,
    ) -> Result<(), TransportError> {
        use futures_util::{SinkExt, StreamExt};

        let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<Message>(256);
        connections.insert(peer_addr.clone(), outgoing_tx);

        loop {
            if !is_running.load(Ordering::SeqCst) {
                break;
            }

            tokio::select! {
                // Handle incoming messages
                Some(result) = ws_stream.next() => {
                    match result {
                        Ok(WsMessage::Binary(data)) => {
                            // Decompress if needed
                            let data = if config.enable_compression && data.len() > config.compression_threshold {
                                Self::decompress(&data)?
                            } else {
                                data
                            };

                            // Decode message
                            match codec.decode(&data) {
                                Ok(msg) => {
                                    // Update stats
                                    {
                                        let mut stats = stats.write().await;
                                        stats.messages_received += 1;
                                        stats.bytes_received += data.len() as u64;
                                        stats.last_activity = Some(chrono::Utc::now());
                                    }

                                    // Forward to incoming channel
                                    if incoming_tx.send((peer_addr.clone(), msg)).await.is_err() {
                                        error!("Failed to forward incoming message");
                                        break;
                                    }
                                }
                                Err(e) => {
                                    error!("Failed to decode message: {}", e);
                                    stats.write().await.errors += 1;
                                }
                            }
                        }
                        Ok(WsMessage::Close(_)) => {
                            info!("Connection closed by peer: {}", peer_addr);
                            break;
                        }
                        Ok(WsMessage::Ping(data)) => {
                            if ws_stream.send(WsMessage::Pong(data)).await.is_err() {
                                break;
                            }
                        }
                        Ok(_) => {} // Ignore other message types
                        Err(e) => {
                            error!("WebSocket error: {}", e);
                            break;
                        }
                    }
                }

                // Handle outgoing messages
                Some(msg) = outgoing_rx.recv() => {
                    // Encode message
                    match codec.encode(&msg) {
                        Ok(mut data) => {
                            // Compress if needed
                            if config.enable_compression && data.len() > config.compression_threshold {
                                data = Self::compress(&data)?;
                            }

                            // Send message
                            if ws_stream.send(WsMessage::Binary(data.clone())).await.is_err() {
                                error!("Failed to send message");
                                break;
                            }

                            // Update stats
                            let mut stats = stats.write().await;
                            stats.messages_sent += 1;
                            stats.bytes_sent += data.len() as u64;
                            stats.last_activity = Some(chrono::Utc::now());
                        }
                        Err(e) => {
                            error!("Failed to encode message: {}", e);
                            stats.write().await.errors += 1;
                        }
                    }
                }

                // Handle broadcast messages
                Ok(msg) = broadcast_rx.recv() => {
                    // Skip if this message is not for us
                    if let Some(dest) = &msg.destination {
                        if dest != &peer_addr {
                            continue;
                        }
                    }

                    // Send broadcast message
                    match codec.encode(&msg) {
                        Ok(mut data) => {
                            if config.enable_compression && data.len() > config.compression_threshold {
                                data = Self::compress(&data)?;
                            }

                            if ws_stream.send(WsMessage::Binary(data)).await.is_err() {
                                error!("Failed to send broadcast");
                                break;
                            }
                        }
                        Err(e) => {
                            error!("Failed to encode broadcast: {}", e);
                        }
                    }
                }
            }
        }

        // Clean up connection
        connections.remove(&peer_addr);
        info!("Connection closed: {}", peer_addr);

        Ok(())
    }

    /// Compress data using gzip
    pub fn compress(data: &[u8]) -> Result<Vec<u8>, TransportError> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| TransportError::Other(anyhow::anyhow!("Compression error: {}", e)))?;
        encoder
            .finish()
            .map_err(|e| TransportError::Other(anyhow::anyhow!("Compression error: {}", e)))
    }

    /// Decompress data using gzip
    pub fn decompress(data: &[u8]) -> Result<Vec<u8>, TransportError> {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| TransportError::Other(anyhow::anyhow!("Decompression error: {}", e)))?;
        Ok(decompressed)
    }
}

#[async_trait]
impl Transport for WebSocketTransport {
    type Message = Message;
    type Error = TransportError;

    async fn send(&self, to: &str, msg: Self::Message) -> Result<(), Self::Error> {
        if let Some(sender) = self.connections.get(to) {
            sender.send(msg).await.map_err(|_| {
                TransportError::ConnectionError("Failed to send message".to_string())
            })?;
            Ok(())
        } else {
            Err(TransportError::ConnectionError(format!(
                "No connection to: {}",
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
        self.broadcast_tx
            .send(msg)
            .map_err(|_| TransportError::ConnectionError("Broadcast failed".to_string()))?;
        Ok(())
    }

    fn local_address(&self) -> Result<String, Self::Error> {
        match &self.mode {
            WsMode::Client { url } => Ok(url.clone()),
            WsMode::Server { bind_addr } => Ok(bind_addr.clone()),
        }
    }

    fn is_connected(&self) -> bool {
        self.is_running.load(Ordering::SeqCst) && !self.connections.is_empty()
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.is_running.store(false, Ordering::SeqCst);
        self.connections.clear();
        Ok(())
    }

    fn stats(&self) -> TransportStats {
        // This will block briefly, but stats access should be infrequent
        futures::executor::block_on(async { self.stats.read().await.clone() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_websocket_transport_creation() {
        let config = TransportConfig::default();
        let mode = WsMode::Server {
            bind_addr: "127.0.0.1:0".to_string(),
        };

        let transport = WebSocketTransport::new(mode, config).await;
        assert!(transport.is_ok());
    }

    #[test]
    fn test_compression() {
        let data = b"Hello, World! This is a test message for compression.";
        let compressed = WebSocketTransport::compress(data).unwrap();
        let decompressed = WebSocketTransport::decompress(&compressed).unwrap();

        assert_eq!(data.as_slice(), decompressed.as_slice());
        // Compressed should be smaller for repetitive data
        assert!(compressed.len() <= data.len());
    }
}
