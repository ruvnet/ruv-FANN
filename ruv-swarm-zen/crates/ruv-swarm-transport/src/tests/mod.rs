//! Unit tests for transport module

#[cfg(test)]
mod protocol_tests {
    use crate::protocol::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::new("test".to_string(), MessageType::Heartbeat { seq: 1 });
        assert_eq!(msg.source, "test");
        assert_eq!(msg.priority, 128);
        assert!(msg.ttl.is_none());
    }

    #[test]
    fn test_protocol_version() {
        let v1 = ProtocolVersion::CURRENT;
        assert_eq!(v1.major, 1);
        assert_eq!(v1.minor, 0);
        assert_eq!(v1.patch, 0);
        assert_eq!(v1.to_string(), "1.0.0");
    }
}

#[cfg(test)]
mod ring_buffer_tests {
    use crate::shared_memory::RingBuffer;

    #[test]
    fn test_ring_buffer_basic() {
        let buffer = RingBuffer::new(1024);
        assert!(buffer.is_empty());
        assert_eq!(buffer.available_space(), 1024);

        // Write some data
        let data = b"Hello, World!";
        buffer.write(data).unwrap();
        assert!(!buffer.is_empty());

        // Read it back
        let read = buffer.read().unwrap();
        assert_eq!(read, data);
        assert!(buffer.is_empty());
    }
}
