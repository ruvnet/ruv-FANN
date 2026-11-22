//! Bounded JSONL research interface. No network access or model file writes.
use ruv_fann_wsp_quant_guard::{
    solve_one_coordinate, Certificate, Edit, Ledger, Model, Patch, Protected, Target,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, Read, Write};
use std::time::Instant;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Predict {
        model: Model,
        inputs: Vec<Vec<i32>>,
    },
    Solve {
        model: Model,
        protected: Protected,
        targets: Vec<Target>,
    },
    Check {
        model: Model,
        protected: Protected,
        edits: Vec<Edit>,
    },
    Verify {
        model: Model,
        protected: Protected,
        patch: Patch,
        certificate: Box<Certificate>,
    },
    Sequence {
        model: Model,
        protected: Protected,
        steps: Vec<Vec<Edit>>,
        #[serde(default)]
        rollback: bool,
    },
}

fn run(request: Request) -> Result<Value, String> {
    match request {
        Request::Predict { model, inputs } => {
            model.validate().map_err(|e| e.to_string())?;
            let work =
                model.weights.len() as u128 * model.weights[0].len() as u128 * inputs.len() as u128;
            if inputs.len() > 4096 || work > 10_000_000 {
                return Err("prediction work budget exceeded".into());
            }
            let outputs: Result<Vec<_>, _> = inputs.iter().map(|x| model.predict(x)).collect();
            Ok(json!({"ok":true,"outputs":outputs.map_err(|e|e.to_string())?}))
        }
        Request::Solve {
            model,
            protected,
            targets,
        } => {
            let mut ledger = Ledger::new(model, protected).map_err(|e| e.to_string())?;
            let start = Instant::now();
            let patch = solve_one_coordinate(&ledger, &targets).map_err(|e| e.to_string())?;
            let solve_us = start.elapsed().as_micros();
            match patch {
                None => Ok(json!({"ok":true,"found":false,"solve_us":solve_us,
                    "scope":"complete one changed weight; fixed bias and quantizer"})),
                Some(patch) => {
                    let start = Instant::now();
                    let certificate = ledger.certify(&patch).map_err(|e| e.to_string())?;
                    let verify_us = start.elapsed().as_micros();
                    let receipt = ledger
                        .apply(&patch, &certificate)
                        .map_err(|e| e.to_string())?;
                    Ok(
                        json!({"ok":true,"found":true,"patch":patch,"certificate":certificate,
                        "model":ledger.model(),"receipt":receipt,
                        "solve_us":solve_us,"verify_us":verify_us}),
                    )
                }
            }
        }
        Request::Check {
            model,
            protected,
            edits,
        } => {
            let mut ledger = Ledger::new(model, protected).map_err(|e| e.to_string())?;
            let patch = Patch {
                expected_parent: ledger.head().into(),
                edits,
            };
            let start = Instant::now();
            let certificate = ledger.certify(&patch).map_err(|e| e.to_string())?;
            let verify_us = start.elapsed().as_micros();
            let receipt = ledger
                .apply(&patch, &certificate)
                .map_err(|e| e.to_string())?;
            Ok(json!({"ok":true,"patch":patch,"certificate":certificate,
                "receipt":receipt,"model":ledger.model(),"verify_us":verify_us}))
        }
        Request::Verify {
            model,
            protected,
            patch,
            certificate,
        } => {
            let ledger = Ledger::new(model, protected).map_err(|e| e.to_string())?;
            ledger
                .verify(&patch, &certificate)
                .map_err(|e| e.to_string())?;
            Ok(json!({"ok":true,"verified":true}))
        }
        Request::Sequence {
            model,
            protected,
            steps,
            rollback,
        } => {
            if steps.len() > 1024 {
                return Err("sequence step budget exceeded".into());
            }
            let mut ledger = Ledger::new(model, protected).map_err(|e| e.to_string())?;
            let mut records = Vec::new();
            let domain_count = ledger.protected().points.len() + ledger.protected().boxes.len();
            let work = steps.len() as u128
                * domain_count.max(1) as u128
                * ledger.model().weights.len() as u128
                * ledger.model().weights[0].len() as u128
                * 2;
            if work > 100_000_000 {
                return Err("sequence aggregate work budget exceeded".into());
            }
            for edits in steps {
                let before_head = ledger.head().to_owned();
                let patch = Patch {
                    expected_parent: before_head.clone(),
                    edits,
                };
                let start = Instant::now();
                match ledger.certify(&patch) {
                    Ok(certificate) => {
                        let verify_us = start.elapsed().as_micros();
                        match ledger.apply(&patch, &certificate) {
                            Ok(receipt) => {
                                records.push(json!({"accepted":true,"patch":patch,
                                "certificate":certificate,"receipt":receipt,"verify_us":verify_us}))
                            }
                            Err(e) => records.push(json!({"accepted":false,"error":e.to_string(),
                                "unchanged_head":ledger.head()==before_head})),
                        }
                    }
                    Err(e) => records.push(json!({"accepted":false,"error":e.to_string(),
                        "unchanged_head":ledger.head()==before_head})),
                }
            }
            let final_model = ledger.model().clone();
            let final_head = ledger.head().to_owned();
            let mut rolled_back = 0;
            if rollback {
                let accepted = records.iter().filter(|r| r["accepted"] == true).count();
                for _ in 0..accepted {
                    ledger.rollback().map_err(|e| e.to_string())?;
                    rolled_back += 1;
                }
            }
            Ok(
                json!({"ok":true,"records":records,"final_model":final_model,
                "final_head":final_head,"rolled_back":rolled_back,
                "after_rollback":ledger.model(),"rollback_head":ledger.head(),
                "baseline_restored":ledger.model()==ledger.baseline()}),
            )
        }
    }
}

fn main() {
    const MAX_LINE: u64 = 8 * 1024 * 1024;
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for _ in 0..4096 {
        let mut bytes = Vec::new();
        let read = input
            .by_ref()
            .take(MAX_LINE + 1)
            .read_until(b'\n', &mut bytes);
        match read {
            Ok(0) => break,
            Ok(_) if bytes.len() as u64 > MAX_LINE => {
                let _ = writeln!(
                    out,
                    "{{\"ok\":false,\"error\":\"JSON line byte limit exceeded\"}}"
                );
                break;
            }
            Err(_) => break,
            _ => {}
        }
        let result = serde_json::from_slice::<Request>(&bytes)
            .map_err(|e| e.to_string())
            .and_then(run);
        let value = result.unwrap_or_else(|e| json!({"ok":false,"error":e}));
        if writeln!(out, "{value}").and_then(|_| out.flush()).is_err() {
            break;
        }
    }
}
