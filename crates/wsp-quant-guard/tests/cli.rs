use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};

fn invoke(bytes: &[u8]) -> Value {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ruv-fann-wsp-quant-guard"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn sequence_has_an_aggregate_verification_budget() {
    let request = json!({"op":"sequence","model":{
        "weights":vec![vec![0;16];16],"bias":vec![0;16],"scales":vec![8;16],
        "output_min":vec![-128;16],"output_max":vec![127;16],
        "weight_min":-128,"weight_max":127,"prefix_id":"test"},
        "protected":{"points":vec![vec![0;16];1024],"boxes":[]},
        "steps":vec![vec![json!({"row":0,"col":0,"value":1})];1024],
        "rollback":true});
    let response = invoke(serde_json::to_string(&request).unwrap().as_bytes());
    assert_eq!(response["ok"], false);
    assert_eq!(response["error"], "sequence aggregate work budget exceeded");
}

#[test]
fn unknown_operation_is_a_structured_error() {
    let response = invoke(br#"{"op":"execute","command":"arbitrary"}"#);
    assert_eq!(response["ok"], false);
    assert!(response["error"]
        .as_str()
        .unwrap()
        .contains("unknown variant"));
}
