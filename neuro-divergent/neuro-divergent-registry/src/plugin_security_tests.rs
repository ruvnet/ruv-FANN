use super::*;

fn descriptor() -> PluginDescriptor {
    PluginDescriptor {
        name: "test_plugin".into(),
        version: "1.0.0".into(),
        description: "Metadata fixture".into(),
        author: "Test".into(),
        license: "MIT".into(),
        homepage: None,
        repository: None,
        keywords: Vec::new(),
        min_runtime_version: "0.1.0".into(),
        max_runtime_version: None,
        models: Vec::new(),
        dependencies: Vec::new(),
        capabilities: PluginCapabilities::default(),
        metadata: HashMap::new(),
        checksum: None,
        signature: None,
    }
}

fn config(directory: &Path) -> PluginConfig {
    PluginConfig {
        allowed_directories: vec![directory.to_owned()],
        ..PluginConfig::default()
    }
}

fn write_descriptor(directory: &Path) -> PathBuf {
    let path = directory.join("plugin.toml");
    std::fs::write(&path, toml::to_string(&descriptor()).unwrap()).unwrap();
    path
}

#[test]
fn reads_only_allowed_descriptor_metadata() {
    let allowed = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let path = write_descriptor(allowed.path());
    let outside = write_descriptor(other.path());
    let plugin = Plugin::load_from_descriptor(&path, config(allowed.path())).unwrap();
    assert_eq!(plugin.get_state(), PluginState::Unloaded);
    assert!(Plugin::load_from_descriptor(&outside, config(allowed.path())).is_err());
}

#[test]
fn blocked_directory_overrides_allowed_parent() {
    let allowed = tempfile::tempdir().unwrap();
    let blocked = allowed.path().join("blocked");
    std::fs::create_dir(&blocked).unwrap();
    let path = write_descriptor(&blocked);
    let mut config = config(allowed.path());
    config.blocked_directories.push(blocked);
    assert!(Plugin::load_from_descriptor(path, config).is_err());
}

#[test]
fn descriptor_size_is_bounded_before_parsing() {
    let allowed = tempfile::tempdir().unwrap();
    let path = allowed.path().join("plugin.toml");
    std::fs::File::create(&path)
        .unwrap()
        .set_len(1024 * 1024 + 1)
        .unwrap();
    let error = Plugin::load_from_descriptor(&path, config(allowed.path()))
        .err()
        .unwrap();
    assert!(error.to_string().contains("1 MiB"));
}

#[test]
fn checksum_policy_requires_a_checksum_and_checks_contents() {
    let allowed = tempfile::tempdir().unwrap();
    let path = allowed.path().join("plugin.bin");
    std::fs::write(&path, b"abc").unwrap();
    let mut descriptor = descriptor();
    let config = config(allowed.path());
    assert!(Plugin::verify_security(&descriptor, &path, &config).is_err());
    descriptor.checksum =
        Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into());
    assert!(Plugin::verify_security(&descriptor, &path, &config).is_ok());
    std::fs::write(&path, b"altered").unwrap();
    assert!(Plugin::verify_security(&descriptor, &path, &config).is_err());
}

#[test]
fn unsupported_security_policies_fail_closed() {
    let allowed = tempfile::tempdir().unwrap();
    for level in [SecurityLevel::Signature, SecurityLevel::Sandboxed] {
        let mut config = config(allowed.path());
        config.security_level = level;
        assert!(matches!(
            Plugin::verify_security(&descriptor(), allowed.path(), &config),
            Err(RegistryError::UnsupportedOperation(_))
        ));
    }
}

#[cfg(unix)]
#[test]
fn descriptor_symlinks_cannot_escape_allowlist() {
    let allowed = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let outside = write_descriptor(other.path());
    let link = allowed.path().join("plugin.toml");
    std::os::unix::fs::symlink(outside, &link).unwrap();
    assert!(Plugin::load_from_descriptor(link, config(allowed.path())).is_err());
}

#[cfg(all(unix, feature = "plugin-system"))]
#[test]
fn native_path_authorizes_the_neighboring_descriptor_too() {
    let allowed = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let binary = allowed.path().join("plugin.bin");
    std::fs::write(&binary, b"not executed").unwrap();
    let external = write_descriptor(outside.path());
    std::os::unix::fs::symlink(external, allowed.path().join("plugin.toml")).unwrap();
    let mut policy = config(allowed.path());
    policy.security_level = SecurityLevel::None;
    let error = Plugin::load_from_path(binary, policy).err().unwrap();
    assert!(matches!(error, RegistryError::PluginError(_)));
    assert!(error.to_string().contains("outside allowed directories"));
}

#[test]
fn compatibility_compares_semantic_versions_instead_of_text() {
    let mut metadata = descriptor();
    metadata.min_runtime_version = "0.2.0".into();
    metadata.max_runtime_version = Some("0.10.0".into());
    let plugin = Plugin::new(metadata, PluginConfig::default());
    assert!(plugin.is_compatible("0.2.0"));
    assert!(plugin.is_compatible("0.10.0"));
    assert!(!plugin.is_compatible("0.1.9"));
    assert!(!plugin.is_compatible("0.10.1"));
    let mut metadata = descriptor();
    metadata.min_runtime_version = "1.10.0".into();
    let plugin = Plugin::new(metadata, PluginConfig::default());
    assert!(!plugin.is_compatible("1.9.0"));
}

#[test]
fn invalid_compatibility_versions_fail_closed() {
    let mut metadata = descriptor();
    let plugin = Plugin::new(metadata.clone(), PluginConfig::default());
    assert!(!plugin.is_compatible("not a version"));
    metadata.min_runtime_version = "invalid".into();
    let plugin = Plugin::new(metadata.clone(), PluginConfig::default());
    assert!(!plugin.is_compatible("1.0.0"));
    metadata.min_runtime_version = "0.1.0".into();
    metadata.max_runtime_version = Some("invalid".into());
    let plugin = Plugin::new(metadata, PluginConfig::default());
    assert!(!plugin.is_compatible("1.0.0"));
}
