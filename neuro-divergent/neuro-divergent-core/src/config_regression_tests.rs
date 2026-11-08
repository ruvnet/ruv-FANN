use super::*;

#[test]
fn generic_configuration_files_roundtrip_and_reject_invalid_json_or_parameters() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.json");
    let mut config = GenericModelConfig::<f64>::new("RNN")
        .with_parameter("learning_rate", ConfigParameter::Float(0.01));
    config.horizon = 3;
    config.input_size = 8;
    config.output_size = 3;
    config.save(&path).unwrap();
    let restored = GenericModelConfig::<f64>::load(&path).unwrap();
    assert_eq!(restored.horizon, 3);
    assert_eq!(restored.input_size, 8);
    assert_eq!(restored.get_float_parameter("learning_rate"), Some(0.01));
    std::fs::write(&path, b"not JSON").unwrap();
    assert!(GenericModelConfig::<f64>::load(&path).is_err());
    let mut invalid = config;
    invalid.horizon = 0;
    assert!(invalid.save(&path).is_err());
    invalid.horizon = 3;
    invalid
        .parameters
        .insert("invalid".into(), ConfigParameter::Float(f64::NAN));
    assert!(invalid.save(&path).is_err());
}

#[test]
fn system_configuration_validation_reports_actual_invalid_fields() {
    let mut manager = ConfigManager::new();
    manager.current_config = Some(SystemConfig::default());
    assert!(manager.validate().is_valid);
    let config = manager.current_config.as_mut().unwrap();
    config.parallel.num_threads = Some(0);
    config.memory.warning_threshold = Some(f64::NAN);
    config.io.buffer_size_kb = 0;
    config.performance.enable_gpu = true;
    let validation = manager.validate();
    assert!(!validation.is_valid);
    assert_eq!(validation.errors.len(), 4);
}

#[cfg(unix)]
#[test]
fn configuration_load_rejects_named_pipes_without_waiting_for_a_writer() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("pipe");
    assert!(std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .unwrap()
        .success());
    assert!(GenericModelConfig::<f64>::load(&path).is_err());
    assert!(ConfigManager::new().load_from_file(&path).is_err());
}
