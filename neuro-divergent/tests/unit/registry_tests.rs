//! The shipped metadata registry is exercised directly, without a mock registry.
use neuro_divergent::registry::{
    ModelCapabilities, ModelCategory, ModelFactory, ModelInfo, ModelRegistry, SearchCriteria,
};

fn info(name: &str, category: ModelCategory) -> ModelInfo {
    ModelInfo {
        name: name.into(),
        category,
        version: "0.2.0".into(),
        description: "Integration metadata".into(),
        parameter_types: vec!["f64".into()],
        input_size: Some(1),
        output_size: Some(2),
        capabilities: ModelCapabilities::default(),
        performance: None,
        metadata: Default::default(),
    }
}

#[test]
fn real_registry_registration_lookup_and_removal_update_indexes() {
    let registry = ModelRegistry::new();
    registry
        .register_info(info("one", ModelCategory::Recurrent))
        .unwrap();
    registry
        .register_info(info("two", ModelCategory::Custom))
        .unwrap();
    assert_eq!(registry.len(), 2);
    assert!(registry.has_model("one"));
    assert_eq!(registry.list_by_category(ModelCategory::Recurrent).len(), 1);
    assert!(registry
        .register_info(info("one", ModelCategory::Recurrent))
        .is_err());
    registry.unregister("one").unwrap();
    assert!(!registry.has_model("one"));
    assert!(registry
        .list_by_category(ModelCategory::Recurrent)
        .is_empty());
    assert_eq!(registry.get_stats().total_models, 1);
}

#[test]
fn metadata_search_and_clear_use_the_actual_registry() {
    let registry = ModelRegistry::new();
    registry
        .register_info(info("metadata-rnn", ModelCategory::Recurrent))
        .unwrap();
    let results = registry.search(&SearchCriteria {
        category: Some(ModelCategory::Recurrent),
        ..Default::default()
    });
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "metadata-rnn");
    registry.clear();
    assert!(registry.is_empty());
    assert!(registry.search(&SearchCriteria::default()).is_empty());
}

#[test]
fn absent_architecture_creators_return_errors_instead_of_fabricated_models() {
    let factory = ModelFactory::new();
    for name in [
        "NBEATS",
        "NHITS",
        "Transformer",
        "PatchTST",
        "DeepAR",
        "TCN",
        "invented",
    ] {
        assert!(
            factory.create::<f64>(name).is_err(),
            "{name} should not fabricate an executable model"
        );
    }
}
