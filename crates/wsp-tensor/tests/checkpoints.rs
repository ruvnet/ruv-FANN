use ruv_fann_wsp_tensor::*;
use std::{fs::{self,File},path::Path};
fn fixture(path:&Path,dtype:&str) {
    let bytes=match dtype {"F32"=>[0.25f32,0.5].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>(),
        "BF16"=>[0.25f32,0.5].into_iter().flat_map(|v|half::bf16::from_f32(v).to_le_bytes()).collect(),
        _=>[0.25f32,0.5].into_iter().flat_map(|v|half::f16::from_f32(v).to_le_bytes()).collect()};
    write_raw(path,&format!(r#"{{"weight":{{"dtype":"{dtype}","shape":[2],"data_offsets":[0,{}]}}}}"#,bytes.len()),&bytes);
}
fn write_raw(path:&Path,header:&str,data:&[u8]) {
    let mut header=header.as_bytes().to_vec();while header.len() & 7 != 0 {header.push(b' ');}
    let mut bytes=(header.len() as u64).to_le_bytes().to_vec();bytes.extend(header);bytes.extend(data);fs::write(path,bytes).unwrap();
}
fn patch(path:&Path)->Patch {Patch{schema:"ruv-fann.tensor-patch/1".into(),base_sha256:fingerprint(path).unwrap(),edits:vec![Edit{tensor:"weight".into(),index:0,value:0.75}]}}
#[test] fn all_precisions_roundtrip_and_official_reader() {
    for dtype in ["F32","BF16","F16"] {
        let d=tempfile::tempdir().unwrap();let b=d.path().join("base");let c=d.path().join("candidate");let restored=d.path().join("restored");
        fixture(&b,dtype);let before=fs::read(&b).unwrap();let r=apply(&b,&c,&patch(&b)).unwrap();
        let bytes=fs::read(&c).unwrap();let parsed=safetensors::SafeTensors::deserialize(&bytes).unwrap();assert_eq!(parsed.tensor("weight").unwrap().shape(),&[2]);
        assert_ne!(r.base_sha256,r.candidate_sha256);assert_eq!(r.behavioral_validation,"not_run");
        rollback(&c,&restored,&r).unwrap();assert_eq!(fs::read(restored).unwrap(),before);assert_eq!(fs::read(&b).unwrap(),before);
    }
}
#[test] fn reject_patch_abuse_without_outputs() {
    for case in 0..10 {
        let d=tempfile::tempdir().unwrap();let b=d.path().join("b");let out=d.path().join("out");fixture(&b,"F32");let mut p=patch(&b);
        match case {0=>p.base_sha256="0".repeat(64),1=>p.edits[0].tensor="missing".into(),2=>p.edits[0].index=u64::MAX,
            3=>p.edits[0].value=f32::NAN,4=>p.edits[0].value=2.0,5=>p.edits.push(Edit{tensor:"weight".into(),index:0,value:0.4}),
            6=>p.edits.clear(),7=>p.edits[0].value=0.25,8=>p.schema="bad".into(),_=>p.edits=(0..129).map(|_|Edit{tensor:"weight".into(),index:0,value:0.5}).collect()}
        assert!(apply(&b,&out,&p).is_err(),"case {case}");assert!(!out.exists());
        assert_eq!(fs::read_dir(d.path()).unwrap().count(),1);
    }
}
#[test] fn reject_malformed_headers() {
    for h in [r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        r#"{"w":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,4]}}"#,
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"w":{"dtype":"UNKNOWN","shape":[1],"data_offsets":[0,4]}}"#] {
        let d=tempfile::tempdir().unwrap();let b=d.path().join("b");write_raw(&b,h,&[0;4]);assert!(inspect(&mut File::open(b).unwrap()).is_err());
    }
}
#[test] fn refuse_overwrite_and_forged_rollback() {
    let d=tempfile::tempdir().unwrap();let b=d.path().join("b");let c=d.path().join("c");let o=d.path().join("o");fixture(&b,"BF16");
    assert!(apply(&b,&b,&patch(&b)).is_err());let mut r=apply(&b,&c,&patch(&b)).unwrap();
    assert!(apply(&b,&c,&patch(&b)).is_err());r.edits[0].before[0]^=1;
    assert!(rollback(&c,&o,&r).is_err());assert!(!o.exists());
    r.candidate_sha256="f".repeat(64);assert!(rollback(&c,&o,&r).is_err());
}
#[test] fn dtype_rounding_is_not_an_edit() {
    let d=tempfile::tempdir().unwrap();let b=d.path().join("b");fixture(&b,"BF16");let mut p=patch(&b);p.edits[0].value=f32::from_bits(0.25f32.to_bits()+1);
    assert!(apply(&b,&d.path().join("c"),&p).is_err());
}
#[test] fn preserve_untouched_tensor_and_metadata() {
    let d=tempfile::tempdir().unwrap();let b=d.path().join("b");let c=d.path().join("c");
    write_raw(&b,r#"{"__metadata__":{"origin":"fixture"},"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"other":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,&[0.25f32,42.0].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>());
    apply(&b,&c,&patch(&b)).unwrap();let old=fs::read(b).unwrap();let new=fs::read(c).unwrap();
    assert_eq!(&old[..old.len()-8],&new[..new.len()-8]);assert_eq!(&old[old.len()-4..],&new[new.len()-4..]);
}
