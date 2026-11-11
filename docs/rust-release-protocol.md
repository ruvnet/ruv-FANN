# Rust registry release

The protected release defaults to `publish_npm=false`. It publishes no npm
packages in this mode and uses the separate `ruv-fann-rust-v0.2.3` GitHub tag.
The 27-crate inventory remains mandatory, including versions already public.

1. Push the final candidate to `release/**`. The complete CI workflow must finish
   successfully at that exact SHA, including all archive, browser, training,
   native, security and integration jobs. An absent, skipped, pending or failed
   job is not acceptance. Fixes require a new SHA and qualification.
2. Fast-forward that unchanged SHA to `main`. Dispatch `protected-release` with
   its full `expected_sha` and leave `publish_npm` false. The controller selects
   a successful CI run on `release/**` by source SHA and workflow identity;
   callers cannot supply a run ID. Main must still identify that SHA and there
   must be zero open issues labeled `release-blocker`.
3. Download that run's immutable 27 Cargo candidates and actual Candle oracle.
   Recheck every candidate archive, source snapshot and successful packaging
   command without registry credentials. Check **all** public versions before
   the first upload. Existing different code, APIs or executable bytes require
   new versions. Only a library differing exclusively in embedded `Cargo.lock`
   may retain its immutable public archive; both public and candidate digests
   are explicit in its receipt. Binary packages never receive this exception.
4. The protected publisher uploads the already verified archives, with at least
   360 seconds between new uploads and a 180-minute job bound. It downloads and
   compares canonical public crate bytes. No local publication command is part
   of this protocol.
5. Download all 27 canonical public archives again. Create a virgin Cargo home
   and consumer using exact public `ruv-fann=0.2.3` and
   `ruv-fann-candle=0.1.1`. Cargo metadata must resolve registry sources and the
   lock checksums must equal the public archive hashes. Compile and execute
   real Candle prepare/export, root inference, parameter/input gradients,
   three-step SGD and AdamW, and immutable frozen snapshots against the actual
   pinned Candle CPU f64 fixture. Every comparison executes. The producer records
   a separate archive-checksum-bound qualification by safely extracting the public
   root archive and executing its packaged backpropagation equivalence, single
   tile, wide overflow and allocation integration tests with the same virgin
   Cargo home. Four nonempty test summaries must report zero failures, ignored
   tests and filtered tests; retain the owned command log as a release asset.
   It emits its source-bound receipt only after all commands succeed with nonzero checks,
   zero failures and zero skips. Preserve command logs and hashes.
6. After validating all evidence and assets, the protected controller creates
   the tag at the exact accepted source SHA if it is absent, then creates the
   draft with `--verify-tag`. GitHub draft creation alone does not create its
   pending tag. An existing same-SHA tag without a release can resume; a
   conflicting tag or an existing draft requires inspection and is never changed.
   Create the Rust GitHub release as a draft with canonical public archives,
   candidate/public/CI receipts, actual fixture and command logs. The accepted
   commit must also bind the required tracked performance report
   `docs/rust-performance-2026-11-11.md` and raw measurements archive
   `benchmarks/rust-0.2.3-evidence.tar.gz`; attach both as verified assets. Check the tag's
   commit SHA and redownload **every** asset, including evidence and SHA256SUMS,
   before and after publication. Never move or overwrite an existing tag.
   A retry accepts an existing release only when its source and every asset
   match exactly. Partial failures retain evidence and require investigation;
   publication alone does not establish successful verification.

`python3 -m unittest discover -s scripts/tests -v` covers release guards and
mutations. Local path-dependency consumer checks qualify the test implementation;
only the protected post-publication virgin public consumer establishes public
registry behavior. Successful qualification is not a publication receipt.
