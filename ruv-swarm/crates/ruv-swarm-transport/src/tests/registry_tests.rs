use crate::{
    in_process::{InProcessRegistry, InProcessTransport},
    Message, Transport, TransportBuilder, TransportConfig, TransportError, TransportRegistry,
};
use async_trait::async_trait;
use std::{sync::Arc, time::Duration};
use tokio::sync::Notify;

struct WaitingBuilder {
    entered: Arc<Notify>,
    proceed: Arc<Notify>,
}

#[async_trait]
impl TransportBuilder for WaitingBuilder {
    async fn build(
        &self,
        config: TransportConfig,
    ) -> Result<Box<dyn Transport<Message = Message, Error = TransportError>>, TransportError> {
        self.entered.notify_one();
        self.proceed.notified().await;
        Ok(Box::new(
            InProcessTransport::new("created".into(), config, InProcessRegistry::new()).await?,
        ))
    }
}

#[tokio::test]
async fn registry_replacement_can_finish_while_a_builder_is_awaiting() {
    let registry = Arc::new(TransportRegistry::new());
    let entered = Arc::new(Notify::new());
    let proceed = Arc::new(Notify::new());
    registry.register(
        "changing",
        WaitingBuilder {
            entered: entered.clone(),
            proceed: proceed.clone(),
        },
    );
    let creator_registry = registry.clone();
    let creation = tokio::spawn(async move {
        creator_registry
            .create("changing", TransportConfig::default())
            .await
    });
    entered.notified().await;
    let replacement_registry = registry.clone();
    let replacement = tokio::task::spawn_blocking(move || {
        replacement_registry.register(
            "changing",
            WaitingBuilder {
                entered: Arc::new(Notify::new()),
                proceed: Arc::new(Notify::new()),
            },
        );
    });
    let result = tokio::time::timeout(Duration::from_secs(2), replacement).await;
    // Always release the builder before asserting, including on a regression.
    proceed.notify_one();
    assert_eq!(
        creation.await.unwrap().unwrap().local_address().unwrap(),
        "inproc://created"
    );
    assert!(
        result.is_ok(),
        "A registry guard was held across the builder await"
    );
    result.unwrap().unwrap();
}
