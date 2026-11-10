# ruv-fann story site

The static GitHub Pages site explains the published packages, the weight-editing lifecycle, and the limits of the current benchmark. The playground runs the actual published Rust/WASM implementation in the browser. It starts only after a click and sends no model inputs to a server.

From the repository root:

```sh
npm ci --prefix site --ignore-scripts
site/node_modules/.bin/playwright install chromium
node scripts/site-build.mjs
node scripts/site-serve.mjs
node scripts/site-test.mjs
```

Open `http://127.0.0.1:4173/ruv-FANN/`. Build and test outputs belong in `_site`, `_site-evidence`, and `site-candidate`; keep them outside source control using `.git/info/exclude`. The private `site/package.json` pins browser/a11y tooling; it is never published to npm.

The build downloads `ruv-fann-wsp@0.2.0`, `ruv-fann@0.1.2`, and `ruv-fann-wasm@0.1.2` from the public npm registry and checks hard-coded SHA-512 integrity before extracting only runtime files and licenses. Everything is served from the Pages origin. A strict Content Security Policy permits the hashed import map and local WebAssembly compilation. The generated source manifest records the Git commit, package integrities, and SHA-256 of every deployed file.

`Pages story` qualifies the exact pull-request head with real Chromium at desktop, mobile, narrow, reduced-motion, and JavaScript-disabled settings. It exercises the actual WASM reject/accept/promote/invalidate/rollback path, clipboard and keyboard controls, scroll narrative, and axe WCAG A/AA checks. Screenshots and the source-bound receipt are archived with a deterministic site archive. There are no skipped tests.

After the unchanged qualified commit reaches `main`, dispatch the same workflow with `expected_sha`. The protected `github-pages` environment downloads the successful exact-head artifact, checks its seal, requires current main and zero `release-blocker` issues, and deploys that archive through the official GitHub Pages actions. It then verifies all public file bytes and reruns the browser checks against the public URL. A publication receipt records the result. The workflow does not publish or change npm packages.
