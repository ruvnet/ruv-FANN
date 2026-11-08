//! Adapt application-owned async functions to the swarm's typed `Agent` trait.
//!
//! The handler performs the work. This crate does not create model-provider
//! sessions or execute commands on its own.

use async_trait::async_trait;
use ruv_swarm_core::{Agent, AgentMetadata, SwarmError};
use std::{fmt::Debug, future::Future, marker::PhantomData};

/// An agent backed by an asynchronous application function.
///
/// Input, output and error types stay explicit, and handler errors propagate to
/// the caller. A mutable handler can retain state between calls; `Agent::process`
/// requires exclusive access, so one agent never invokes it concurrently.
pub struct FunctionAgent<I, O, E, F> {
    id: String,
    capabilities: Vec<String>,
    metadata: AgentMetadata,
    handler: F,
    signature: PhantomData<fn(I) -> Result<O, E>>,
}

impl<I, O, E, F> FunctionAgent<I, O, E, F> {
    /// Create an adapter without invoking the handler.
    pub fn new<Fut>(
        id: impl Into<String>,
        capabilities: Vec<String>,
        handler: F,
    ) -> Result<Self, SwarmError>
    where
        F: FnMut(I) -> Fut,
        Fut: Future<Output = Result<O, E>>,
    {
        let id = id.into();
        if id.trim().is_empty() || id.contains('\0') {
            return Err(SwarmError::custom(
                "Agent ID must be nonempty and contain no null bytes",
            ));
        }
        if capabilities
            .iter()
            .any(|name| name.trim().is_empty() || name.contains('\0'))
        {
            return Err(SwarmError::custom(
                "Capability names must be nonempty and contain no null bytes",
            ));
        }
        Ok(Self {
            metadata: AgentMetadata {
                name: id.clone(),
                ..Default::default()
            },
            id,
            capabilities,
            handler,
            signature: PhantomData,
        })
    }

    /// Replace descriptive metadata without changing identity or capabilities.
    pub fn with_metadata(mut self, metadata: AgentMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

#[async_trait]
impl<I, O, E, F, Fut> Agent for FunctionAgent<I, O, E, F>
where
    I: Send + 'static,
    O: Send + 'static,
    E: Debug + Send + 'static,
    F: FnMut(I) -> Fut + Send + Sync,
    Fut: Future<Output = Result<O, E>> + Send,
{
    type Input = I;
    type Output = O;
    type Error = E;

    async fn process(&mut self, input: I) -> Result<O, E> {
        (self.handler)(input).await
    }

    fn id(&self) -> &str {
        &self.id
    }
    fn capabilities(&self) -> &[String] {
        &self.capabilities
    }
    fn metadata(&self) -> AgentMetadata {
        self.metadata.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn invokes_handler_with_state_and_preserves_errors() {
        let mut count = 0;
        let mut agent = FunctionAgent::new("counter", vec!["count".into()], move |input: i32| {
            count += 1;
            let result = if input >= 0 {
                Ok(input + count)
            } else {
                Err("negative input")
            };
            async move { result }
        })
        .unwrap();
        assert_eq!(agent.process(10).await.unwrap(), 11);
        assert_eq!(agent.process(10).await.unwrap(), 12);
        assert_eq!(agent.process(-1).await, Err("negative input"));
        assert_eq!(agent.id(), "counter");
        assert!(agent.has_capability("count"));
        assert_eq!(agent.metadata().name, "counter");
    }

    #[test]
    fn validates_identity_without_running_application_code() {
        let handler = |_: ()| async { Ok::<(), ()>(()) };
        assert!(FunctionAgent::new(" ", vec![], handler).is_err());
        assert!(FunctionAgent::new("safe", vec!["bad\0name".into()], handler).is_err());
    }
}
