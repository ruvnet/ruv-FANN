# Rust WSP tensor checkpoint experiments

Standalone, unpublished backend for bounded direct edits to Safetensors shards.
This extends the WSP workflow concept. It does not replace or import the small
dense model schema in `ruv-fann-wsp`, and does not implement PAN2 inference.

## Build and verify

```sh
cargo test --locked
cargo build --release --locked
python3 scripts/e2e.py
python3 scripts/benchmark.py
```

The integration tests use the official Rust Safetensors reader on the generated
candidates. Python independently generates and decodes a BF16 fixture and drives
the compiled CLI through inspect, edit, rejection and byte-exact rollback.

## Inspect, edit, rollback

```sh
target/release/ruv-fann-wsp-tensor inspect original.safetensors
target/release/ruv-fann-wsp-tensor apply original.safetensors candidate.safetensors patch.json > receipt.json
target/release/ruv-fann-wsp-tensor rollback candidate.safetensors restored.safetensors receipt.json
cmp original.safetensors restored.safetensors
```

Patch format, using the exact shard SHA256 from inspect:

```json
{
  "schema": "ruv-fann.tensor-patch/1",
  "base_sha256": "REPLACE_WITH_EXACT_SHARD_SHA256",
  "edits": [{"tensor": "REPLACE_WITH_EXACT_TENSOR_NAME", "index": 0, "value": 0.75}]
}
```

Indices are flattened row-major tensor positions. Supported edits: F32, F16,
BF16. Other supported integer and F64 tensors remain byte-identical. No topology,
tensor shape, tensor name or metadata changes. At most 128 unique replacements;
each finite replacement must have absolute value at most 1,000,000 and actual
encoded change at most 1. A change that rounds to the same bytes rejects.
These are resource and mutation bounds, not evidence of behavioral safety.

Every operation writes a private temporary copy in the destination directory.
It validates the hash and structure of that copy before editing it. Persistence
uses an atomic no-clobber operation; original files and existing outputs are
never overwritten. Rollback verifies candidate hash, edited bytes and restored
baseline hash. A receipt records `behavioral_validation: not_run`.

Keep stdout receipts with their checkpoints; if output capture fails, the
candidate may exist without a receipt. The original is still intact. Hashes
provide content integrity, not authorization or trusted evaluation signatures.

## PAN2 paired inference

Use a separately provisioned Linux CUDA executor with the official PAN2
environment and dependencies already installed. No GPU is provisioned by this
project. The script runs the official launcher in its `pan` Conda environment.
It requires a clean upstream checkout pinned to
`0fb907b6e16a82ead6d691d1de2eef6299c71e9a`.

Prepare complete native-format transformer directories for baseline and
candidate. Convert the released Diffusers weights with upstream's
`scripts/convert_checkpoint.py` first. Apply the Rust patch to a selected native
Safetensors shard. Keep configuration and all other shards identical. Check
the index file if the checkpoint layout uses one; this CLI edits one shard and
does not manage shard indexes or full checkpoint directories.

```sh
python3 scripts/pan2_compare.py \
  --repo /models/PAN2 \
  --baseline /models/base/transformer \
  --candidate /models/candidate/transformer \
  --prompts /evaluation/prompts.json \
  --output /evaluation/pan2-comparison
```

The harness records checkpoint and output hashes, prompt hash, fixed seed,
upstream revision and timings. It uses fresh output directories, checks inputs
again after inference, and terminates the process group on timeout or error.
This is a paired generation harness, not a validated physics evaluator.
Prompt-referenced images and external text encoder/VAE caches must be separately
pinned and kept immutable by the operator. Those dependencies are not hashed by
this initial harness. Review upstream licensing for each component.

No automatic promotion is implemented. Score action adherence and physical
consistency against real observations on an unseen scenario set before choosing
to deploy a candidate. Generating plausible video does not prove correctness.

## Performance and deployment

No full-shard allocation or mmap is used. Header length is capped at 16 MiB;
JSON expansion is additional memory. Payload hashing uses a 1 MiB buffer. Copy
uses the operating system's optimized streaming copy where available. Cost is
linear in shard size and requires an additional shard's worth of disk space.
Use local NVMe for candidate creation; keep the large model off Workers.

Cloudflare R2 stores small validation receipts in a private bucket. No public
endpoint, model upload or remote execution API is required. Use your existing
authenticated Cloudflare tooling to upload reports; no credentials are included
here. Production checkpoint transfer should use resumable multipart upload and
verify SHA256 after download. This implementation does not automate that upload.

## Threat model and limits

Untrusted patch JSON and malformed checkpoint headers are validated. JSON input
is limited to 1 MiB, header to 16 MiB, rank to 32, dimensions checked for overflow,
and offsets checked for contiguous coverage. Checkpoints are local trusted-user
files, not a public file upload service. Use filesystem permissions and a job
sandbox before exposing this CLI to other tenants. Filesystem failures, process
termination and unavailable disk space can interrupt a run; the source remains
untouched. Receipts are unsigned. Dependency advisory scanning is a separate
release check. This crate is `publish = false` until review and GPU validation.

Acceptance: candidates load with the official reader; unchanged tensors retain
their bytes; invalid and stale edits reject; rollback restores the entire original
file hash. Full PAN2 quality and speed improvements remain unverified.

Sources: [PAN2](https://github.com/ifm-ai/PAN2),
[WSP](https://github.com/ruvnet/ruv-FANN/tree/main/packages/wsp),
[Cloudflare R2](https://developers.cloudflare.com/r2/objects/upload-objects/).
