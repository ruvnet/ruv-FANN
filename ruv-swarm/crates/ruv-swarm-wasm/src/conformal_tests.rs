use super::*;
fn fit_json() -> String {
    let mut records = Vec::new();
    for (i, error) in [7., 1., 5., 9., 3., 2., 8., 6., 4.].iter().enumerate() {
        for (series, scale) in [("a", 1.), ("b", 10.)] {
            for model in ["A", "B"] {
                for h in [2, 1] {
                    records.push(json!({"series":series,"model":model,"cutoff":i*2,
                "timestamp":i*2+h,"actual":0.,"prediction":-error*scale*if h==2 {3.}else{1.}}));
                }
            }
        }
    }
    json!({"as_of":18,"horizon":2,"coverages":[0.9,0.5,0.8],"records":records}).to_string()
}
fn apply_json() -> String {
    let mut records = Vec::new();
    for series in ["b", "a"] {
        for model in ["B", "A"] {
            for time in [20, 19] {
                records.push(json!({"series":series,"model":model,"origin":18,"timestamp":time,"prediction":100.}));
            }
        }
    }
    json!({"records":records}).to_string()
}
#[test]
fn shared_quantiles_and_output_are_real_sorted_intervals() {
    let calibrator = WasmConformalCalibrator::from_json(&fit_json()).unwrap();
    let groups: serde_json::Value =
        serde_json::from_str(&calibrator.groups_json().unwrap()).unwrap();
    assert_eq!(groups[0]["radii"], json!([5., 8., 9.]));
    assert_eq!(groups.as_array().unwrap().len(), 8);
    let output: serde_json::Value =
        serde_json::from_str(&calibrator.apply_json(&apply_json()).unwrap()).unwrap();
    assert_eq!(output.as_array().unwrap().len(), 24);
    assert_eq!(output[0]["series"], "a");
    assert_eq!(output[0]["model"], "A");
    assert_eq!(output[0]["horizon"], 1);
    assert_eq!(output[0]["lower"], 95.);
    assert_eq!(output[0]["upper"], 105.);
    assert_eq!(output[0]["calibration_samples"], 9);
    assert_eq!(output[0]["forecast_origin"], 18);
}
#[test]
fn exact_big_integer_and_exponent_tokens_keep_their_float64_value() {
    // 2^98 is exactly representable as Float64. Without float_roundtrip,
    // serde_json's significand/exponent parser can return one ULP less.
    let expected = 316912650057057350374175801344_u128 as f64;
    for token in ["316912650057057350374175801344", "3.1691265005705735e29"] {
        let input = format!(
            r#"{{"as_of":1,"horizon":1,"coverages":[0.5],"records":[{{"series":"a","model":"A","cutoff":0,"timestamp":1,"actual":{token},"prediction":0}}]}}"#
        );
        let calibrator = WasmConformalCalibrator::from_json(&input).unwrap();
        assert_eq!(
            calibrator.engine.groups().next().unwrap().radii[0],
            expected
        );
    }
}
#[test]
fn chronology_and_incomplete_or_misaligned_models_fail() {
    for field in ["timestamp", "cutoff", "actual", "prediction"] {
        let mut v: serde_json::Value = serde_json::from_str(&fit_json()).unwrap();
        v["records"][0][field] = match field {
            "timestamp" => json!(19),
            "cutoff" => json!(19),
            _ => serde_json::Value::Null,
        };
        assert!(WasmConformalCalibrator::from_json(&v.to_string()).is_err());
    }
    let mut v: serde_json::Value = serde_json::from_str(&fit_json()).unwrap();
    v["records"].as_array_mut().unwrap().pop();
    assert!(WasmConformalCalibrator::from_json(&v.to_string()).is_err());
    let engine = WasmConformalCalibrator::from_json(&fit_json()).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&apply_json()).unwrap();
    v["records"][0]["origin"] = json!(17);
    assert!(engine.apply_json(&v.to_string()).is_err());
}
#[test]
fn safe_integer_timestamps_and_exact_integer_literals_are_enforced() {
    for literal in [
        "9007199254740993",
        "18446744073709551617",
        "340282366920938463463374607431768211455",
    ] {
        let input = format!(
            r#"{{"as_of":1,"horizon":1,"coverages":[0.5],"records":[{{"series":"a","model":"A","cutoff":0,"timestamp":1,"actual":{literal},"prediction":0}}]}}"#
        );
        assert!(WasmConformalCalibrator::from_json(&input)
            .unwrap_err()
            .contains("exactly representable"));
    }
    let mut v: serde_json::Value = serde_json::from_str(&fit_json()).unwrap();
    v["as_of"] = json!(9007199254740992_u64);
    assert!(WasmConformalCalibrator::from_json(&v.to_string()).is_err());
    v["as_of"] = json!(18.5);
    assert!(WasmConformalCalibrator::from_json(&v.to_string()).is_err());
    let escaped = fit_json().replace("\"a\"", "\"a9007199254740993\\\"x\"");
    assert!(WasmConformalCalibrator::from_json(&escaped).is_ok());
}
#[test]
fn malformed_types_names_and_budgeted_arrays_fail() {
    for value in [json!("no"), json!(null), json!([])] {
        let mut v: serde_json::Value = serde_json::from_str(&fit_json()).unwrap();
        v["as_of"] = value;
        assert!(WasmConformalCalibrator::from_json(&v.to_string()).is_err());
    }
    let mut v: serde_json::Value = serde_json::from_str(&fit_json()).unwrap();
    v["records"][0]["series"] = json!("x".repeat(257));
    assert!(WasmConformalCalibrator::from_json(&v.to_string()).is_err());
    v["records"][0]["series"] = json!("a");
    v["coverages"] = json!(vec![0.5; 17]);
    assert!(WasmConformalCalibrator::from_json(&v.to_string()).is_err());
    assert!(WasmConformalCalibrator::from_json(&" ".repeat(INPUT_BYTES + 1)).is_err());
    let input: Result<Vec<u32>, _> = wire::parse(&format!("[{}]", vec!["0"; 10001].join(",")));
    assert!(input.is_ok()); // Generic parser budget differs; bounded record visitor is below.
    let mut v: serde_json::Value = serde_json::from_str(&fit_json()).unwrap();
    let record = v["records"][0].clone();
    v["records"] = json!(vec![record; 10001]);
    assert!(WasmConformalCalibrator::from_json(&v.to_string()).is_err());
}
#[test]
fn bounded_serializer_and_preallocation_estimate_reject_large_outputs() {
    assert!(wire::serialize(&"x".repeat(OUTPUT_BYTES)).is_err());
    let calibrator = WasmConformalCalibrator::from_json(&fit_json()).unwrap();
    let record = json!({"series":"a","model":"A","origin":18,"timestamp":19,"prediction":1.});
    let input = json!({"records":vec![record;10000]}).to_string();
    assert!(calibrator
        .apply_json(&input)
        .unwrap_err()
        .contains("byte budget"));
}
