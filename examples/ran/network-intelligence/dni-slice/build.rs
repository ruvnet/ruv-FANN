fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .compile(
            &[
                "proto/slice_service.proto",
                "../../shared/proto/proto/network_intelligence.proto",
                "../../shared/proto/proto/common.proto",
            ],
            &["proto", "../../shared/proto/proto"],
        )?;
    Ok(())
}