//! Exercise the actual SDK codec, including bounded buffering and frame recovery.
use std::time::Duration;

use rmcp::{
    transport::{async_rw::AsyncRwTransport, Transport},
    RoleServer,
};
use tokio::{
    io::{duplex, AsyncWriteExt},
    time::timeout,
};

use crate::service::MAX_STDIO_MESSAGE_BYTES;

#[tokio::test]
async fn oversized_line_is_dropped_and_the_next_real_request_survives() {
    let (mut writer, reader) = duplex(4096);
    let mut transport = AsyncRwTransport::<RoleServer, _, _>::new_server(reader, tokio::io::sink())
        .with_max_line_length(MAX_STDIO_MESSAGE_BYTES);
    let sender = tokio::spawn(async move {
        let chunk = [b' '; 4096];
        for _ in 0..=MAX_STDIO_MESSAGE_BYTES / chunk.len() {
            writer.write_all(&chunk).await.unwrap();
        }
        writer
            .write_all(b"\n{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n")
            .await
            .unwrap();
    });
    let request = timeout(Duration::from_secs(5), transport.receive())
        .await
        .unwrap()
        .unwrap();
    let request = serde_json::to_value(request).unwrap();
    assert_eq!(request["id"], 7);
    assert_eq!(request["method"], "ping");
    sender.await.unwrap();
}

#[tokio::test]
async fn fragmented_utf8_is_preserved_until_the_full_line_arrives() {
    let (mut writer, reader) = duplex(64);
    let mut transport = AsyncRwTransport::<RoleServer, _, _>::new_server(reader, tokio::io::sink())
        .with_max_line_length(MAX_STDIO_MESSAGE_BYTES);
    let sender = tokio::spawn(async move {
        let request = "{\"jsonrpc\":\"2.0\",\"id\":8,\"method\":\"tools/call\",\"params\":{\"name\":\"memory_get\",\"arguments\":{\"key\":\"caf\u{e9}\"}}}\n";
        for chunk in request.as_bytes().chunks(1) {
            writer.write_all(chunk).await.unwrap();
            tokio::task::yield_now().await;
        }
    });
    let request = timeout(Duration::from_secs(5), transport.receive())
        .await
        .unwrap()
        .unwrap();
    let request = serde_json::to_value(request).unwrap();
    assert_eq!(request["params"]["arguments"]["key"], "café");
    sender.await.unwrap();
}

#[tokio::test]
async fn eof_does_not_dispatch_an_unterminated_request() {
    let (mut writer, reader) = duplex(128);
    let mut transport = AsyncRwTransport::<RoleServer, _, _>::new_server(reader, tokio::io::sink())
        .with_max_line_length(MAX_STDIO_MESSAGE_BYTES);
    writer
        .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}")
        .await
        .unwrap();
    drop(writer);
    assert!(timeout(Duration::from_secs(5), transport.receive())
        .await
        .unwrap()
        .is_none());
}
