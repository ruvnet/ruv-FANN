use super::*;

#[test]
fn checkpoint_regular_file_open_remains_bounded() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("checkpoint.json");
    std::fs::write(&path, b"{}").unwrap();
    assert!(open_checkpoint(&path)
        .unwrap()
        .metadata()
        .unwrap()
        .is_file());
    File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len((LIMIT + 1) as u64)
        .unwrap();
    assert!(
        matches!(open_checkpoint(&path), Err(Error::SerializationError(message))
        if message.contains("64 MiB"))
    );
}

#[cfg(unix)]
fn fifo(path: &Path) {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: path is a live NUL-terminated string; this creates only a private
    // test fixture inside tempfile's owned directory.
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
}

#[cfg(unix)]
fn assert_fifo_rejected_promptly(path: &Path, after_preflight: bool) {
    use std::os::unix::fs::OpenOptionsExt;
    let worker_path = path.to_owned();
    let (sender, receiver) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result = if after_preflight {
            open_after_preflight(&worker_path).map(|_| ())
        } else {
            load::<f64>(&worker_path).map(|_| ())
        };
        let _ = sender.send(result.map_err(|error| format!("{error:?}")));
    });
    let result = receiver.recv_timeout(std::time::Duration::from_secs(2));
    if result.is_err() {
        // Unblock a regressed blocking reader before joining, so a failing test
        // reports its timeout without leaving a stuck cargo test process.
        let wake = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)
            .unwrap();
        drop(wake);
    }
    worker.join().unwrap();
    assert!(
        matches!(result, Ok(Err(message)) if message.contains("SerializationError")
        && message.contains("regular file")),
        "FIFO open did not return the expected typed error promptly"
    );
}

#[cfg(unix)]
#[test]
fn checkpoint_load_rejects_fifo_without_waiting_for_a_writer() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("checkpoint");
    fifo(&path);
    assert_fifo_rejected_promptly(&path, false);
}

#[cfg(unix)]
#[test]
fn checkpoint_open_rejects_fifo_replacement_after_regular_file_preflight() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("checkpoint");
    std::fs::write(&path, b"{}").unwrap();
    validate_file(&std::fs::metadata(&path).unwrap()).unwrap();
    // Deterministically exercise the race window between metadata and open.
    std::fs::remove_file(&path).unwrap();
    fifo(&path);
    assert_fifo_rejected_promptly(&path, true);
}
