//! Tests for custom task payload functionality

use crate::task::{CustomPayload, Task, TaskPayload};
use core::fmt;

#[derive(Debug, Clone)]
struct TestPayload {
    data: String,
    value: i32,
}

impl CustomPayload for TestPayload {
    fn clone_box(&self) -> Box<dyn CustomPayload> {
        Box::new(self.clone())
    }
}

#[test]
fn test_custom_payload_implementation() {
    let custom = TestPayload {
        data: "custom data".to_string(),
        value: 42,
    };
    
    let payload = TaskPayload::Custom(Box::new(custom));
    
    // Test that we can clone the payload
    let cloned = payload.clone();
    
    match cloned {
        TaskPayload::Custom(_) => {}, // Success
        _ => panic!("Expected custom payload after cloning"),
    }
}

#[test]
fn test_task_with_custom_payload() {
    let custom_data = TestPayload {
        data: "important task data".to_string(),
        value: 100,
    };
    
    let task = Task::new("custom-task", "special")
        .with_payload(TaskPayload::Custom(Box::new(custom_data)));
    
    assert_eq!(task.id.to_string(), "custom-task");
    assert_eq!(task.task_type, "special");
    
    match &task.payload {
        TaskPayload::Custom(_) => {}, // Success
        _ => panic!("Expected custom payload in task"),
    }
}

// More complex custom payload with nested data
#[derive(Debug, Clone)]
struct ComplexPayload {
    id: u64,
    nested: NestedData,
    tags: Vec<String>,
}

#[derive(Debug, Clone)]
struct NestedData {
    name: String,
    config: Vec<u8>,
}

impl CustomPayload for ComplexPayload {
    fn clone_box(&self) -> Box<dyn CustomPayload> {
        Box::new(self.clone())
    }
}

#[test]
fn test_complex_custom_payload() {
    let complex = ComplexPayload {
        id: 12345,
        nested: NestedData {
            name: "config-v2".to_string(),
            config: vec![0x01, 0x02, 0x03, 0x04],
        },
        tags: vec!["urgent".to_string(), "gpu-required".to_string()],
    };
    
    let payload = TaskPayload::Custom(Box::new(complex));
    let cloned = payload.clone();
    
    // Verify cloning works correctly
    match (payload, cloned) {
        (TaskPayload::Custom(_), TaskPayload::Custom(_)) => {}, // Success
        _ => panic!("Cloning custom payload failed"),
    }
}

// Test payload that implements Display for better debugging
#[derive(Debug, Clone)]
struct DisplayablePayload {
    message: String,
}

impl fmt::Display for DisplayablePayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisplayablePayload: {}", self.message)
    }
}

impl CustomPayload for DisplayablePayload {
    fn clone_box(&self) -> Box<dyn CustomPayload> {
        Box::new(self.clone())
    }
}

#[test]
fn test_displayable_custom_payload() {
    let displayable = DisplayablePayload {
        message: "This payload can be displayed".to_string(),
    };
    
    // Test Display implementation
    let display_string = format!("{displayable}");
    assert!(display_string.contains("This payload can be displayed"));
    
    // Use in task
    let task = Task::new("display-task", "debug")
        .with_payload(TaskPayload::Custom(Box::new(displayable)));
    
    // Verify Debug output works (required by CustomPayload trait)
    let debug_string = format!("{:?}", task.payload);
    assert!(debug_string.contains("DisplayablePayload"));
}

// Test that custom payloads work with task builder pattern
#[test]
fn test_custom_payload_with_task_builder() {
    #[derive(Debug, Clone)]
    struct BuilderPayload {
        step: u32,
    }
    
    impl CustomPayload for BuilderPayload {
        fn clone_box(&self) -> Box<dyn CustomPayload> {
            Box::new(self.clone())
        }
    }
    
    let task = Task::new("builder-task", "process")
        .with_priority(crate::task::TaskPriority::High)
        .require_capability("custom-processor")
        .with_timeout(5000)
        .with_payload(TaskPayload::Custom(Box::new(BuilderPayload { step: 1 })));
    
    assert_eq!(task.priority, crate::task::TaskPriority::High);
    assert_eq!(task.timeout_ms, Some(5000));
    assert!(task.required_capabilities.contains(&"custom-processor".to_string()));
    
    match task.payload {
        TaskPayload::Custom(_) => {}, // Success
        _ => panic!("Expected custom payload"),
    }
}

// Test thread safety of custom payloads
#[cfg(feature = "std")]
#[test]
fn test_custom_payload_thread_safety() {
    use std::sync::Arc;
    use std::thread;
    
    #[derive(Debug, Clone)]
    struct ThreadSafePayload {
        data: Arc<String>,
    }
    
    impl CustomPayload for ThreadSafePayload {
        fn clone_box(&self) -> Box<dyn CustomPayload> {
            Box::new(self.clone())
        }
    }
    
    let shared_data = Arc::new("shared data".to_string());
    let payload = ThreadSafePayload {
        data: shared_data.clone(),
    };
    
    let task_payload = TaskPayload::Custom(Box::new(payload));
    
    // Clone for thread
    let cloned_payload = task_payload.clone();
    
    // Spawn thread to verify payload can be sent across threads
    let handle = thread::spawn(move || {
        matches!(cloned_payload, TaskPayload::Custom(_))
    });
    
    let result = handle.join().unwrap();
    assert!(result);
}