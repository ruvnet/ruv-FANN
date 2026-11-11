#!/usr/bin/env python3
"""Attach canonical public Cargo bytes and evidence to an exact-source GitHub release."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_rust', ROOT / 'scripts/verify-public-rust.py')
public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)
REPO = 'ruvnet/ruv-FANN'
TAG = 'ruv-fann-rust-v0.2.3'
PERFORMANCE_ASSETS = ('docs/rust-performance-2026-11-11.md', 'benchmarks/rust-0.2.3-evidence.tar.gz')


def run(*args):
    return subprocess.check_output(['gh', *args], text=True).strip()


def tag_source():
    obj = json.loads(run('api', f'repos/{REPO}/git/ref/tags/{TAG}'))['object']
    for _ in range(5):
        if obj['type'] == 'commit':
            return obj['sha']
        if obj['type'] != 'tag':
            break
        obj = json.loads(run('api', f'repos/{REPO}/git/tags/{obj["sha"]}'))['object']
    raise ValueError('Release tag does not identify a commit')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def performance_assets(source):
    tracked = subprocess.check_output(['git', 'ls-files', '--error-unmatch', '--', *PERFORMANCE_ASSETS], cwd=ROOT, text=True).splitlines()
    if set(tracked) != set(PERFORMANCE_ASSETS):
        raise ValueError('Reviewed performance report and raw evidence must be tracked')
    paths = [ROOT / name for name in PERFORMANCE_ASSETS]
    if any(path.is_symlink() or not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise ValueError('Missing or empty reviewed performance report/raw evidence')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True):
        raise ValueError('Performance release assets require the clean accepted source checkout')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != source:
        raise ValueError('Performance release assets differ from the accepted source SHA')
    return paths


def validate_evidence(base, source):
    names = ('cargo-candidate-check', 'cargo-publication', 'public-rust-consumers', 'validated-ci')
    receipts = {name: json.loads((base / f'{name}.json').read_text()) for name in names}
    if any(row.get('sourceSha') != source for row in receipts.values()):
        raise ValueError('Release evidence source mismatch')
    public.validate(receipts['public-rust-consumers'], receipts['cargo-publication'], source)
    ci = receipts['validated-ci']
    if ci.get('schema') != 'ruv-fann.rust-candidate-ci/1' or not ci.get('jobs') or any(row.get('conclusion') != 'success' for row in ci['jobs']):
        raise ValueError('Missing or failed exact-source CI')
    crates = receipts['cargo-publication']['crates']
    if len(crates) != 27 or any(row['status'] not in ('already-verified', 'published-and-byte-verified', 'lockfile-only-drift-library') for row in crates):
        raise ValueError('Incomplete Cargo publication')
    candidates = receipts['cargo-candidate-check']['crates']
    if len(candidates) != 27 or len({row['name'] for row in candidates}) != 27:
        raise ValueError('Incomplete candidate evidence')
    by_name = {row['name']: row for row in candidates}
    for row in crates:
        expected = by_name[row['name']]
        if row['version'] != expected['version'] or row['sha256'] != expected['sha256']:
            # Immutable library archives may differ only in their ignored lock.
            if row['status'] != 'lockfile-only-drift-library' or row.get('candidate_sha256') != expected.get('candidate_sha256', expected['sha256']):
                raise ValueError('Candidate/public digest mismatch')
    return receipts


def main():
    base = Path(sys.argv[1]).resolve()
    source = os.environ['GITHUB_SHA']
    if (os.environ.get('GITHUB_ACTIONS'), os.environ.get('GITHUB_WORKFLOW'), os.environ.get('GITHUB_REF')) != ('true', 'protected-release', 'refs/heads/main'):
        raise ValueError('GitHub releases require the protected main workflow')
    reviewed_performance = performance_assets(source)
    if json.loads(run('api', f'repos/{REPO}/git/ref/heads/main'))['object']['sha'] != source:
        raise ValueError('Main moved since candidate acceptance')
    if json.loads(run('api', f'repos/{REPO}/issues?labels=release-blocker&state=open&per_page=1')):
        raise ValueError('Open release blockers')
    receipts = validate_evidence(base, source)
    assets = base / 'github-release-assets'
    assets.mkdir()
    for path in reviewed_performance:
        shutil.copyfile(path, assets / path.name)
    for row in receipts['cargo-publication']['crates']:
        archive = base / 'cargo-public-archives' / f"{row['name']}-{row['version']}.crate"
        if digest(archive) != row['sha256']:
            raise ValueError('Missing canonical public archive: ' + row['name'])
        shutil.copyfile(archive, assets / archive.name)
    for name in receipts:
        shutil.copyfile(base / f'{name}.json', assets / f'{name}.json')
    for name in ('receipt.json', 'receipt.sha256', 'publication-order.json', 'snapshot-files.json'):
        shutil.copyfile(base / 'cargo-candidates' / name, assets / ('candidate-' + name))
    fixture = Path(os.environ['RUV_FANN_CANDLE_FIXTURE'])
    if digest(fixture) != receipts['public-rust-consumers']['fixtureSha256']:
        raise ValueError('Candle fixture changed')
    shutil.copyfile(fixture, assets / 'candle-f64-fixture.json')
    for index, command in enumerate(receipts['public-rust-consumers']['commands']):
        log = base / f'public-rust-{index}.log'
        if digest(log) != command['logSha256']:
            raise ValueError('Public consumer command log changed')
        shutil.copyfile(log, assets / log.name)
    packed_log = base / 'public-rust-packed.log'
    if digest(packed_log) != receipts['public-rust-consumers']['packedArtifactQualification']['command']['logSha256']:
        raise ValueError('Packed public artifact command log changed')
    shutil.copyfile(packed_log, assets / packed_log.name)
    manifest = {path.name: digest(path) for path in assets.iterdir()}
    (assets / 'SHA256SUMS').write_text(''.join(f'{value}  {name}\n' for name, value in sorted(manifest.items())))
    manifest['SHA256SUMS'] = digest(assets / 'SHA256SUMS')
    def verify(folder):
        if tag_source() != source:
            raise ValueError('Release tag source mismatch')
        run('release', 'download', TAG, '--repo', REPO, '--dir', str(folder))
        if {p.name: digest(p) for p in folder.iterdir()} != manifest:
            raise ValueError('GitHub release asset byte mismatch')
    existing = subprocess.run(['gh', 'api', f'repos/{REPO}/releases/tags/{TAG}'], capture_output=True, text=True)
    ref = subprocess.run(['gh', 'api', f'repos/{REPO}/git/ref/tags/{TAG}'], capture_output=True, text=True)
    if existing.returncode == 0 or ref.returncode == 0:
        if existing.returncode or json.loads(existing.stdout).get('draft') is not False:
            raise ValueError('Existing tag or draft requires inspection; never overwrite')
        verify(base / 'github-existing-download')
        status = 'already-released'
    else:
        # A lookup failure other than an actual absence cannot authorize creation.
        if '404' not in existing.stderr or '404' not in ref.stderr:
            raise ValueError('Could not establish release/tag absence')
        notes = base / 'rust-release-notes.md'
        notes.write_text(f'Rust registry release from `{source}`.\n\n'
            'ruv-fann 0.2.3 and ruv-fann-candle 0.1.1; all 27 Cargo candidates verified. '
            'Canonical public archives, candidate/public receipts and the actual Candle fixture are attached. '
            'A fresh public Cargo consumer passed inference and three-step SGD/AdamW gradients and updates. '
            'Library-only Cargo.lock differences retain their immutable public bytes and candidate digests in the receipts.\n')
        run('release', 'create', TAG, '--repo', REPO, '--target', source, '--draft', '--title',
            'ruv-FANN Rust 0.2.3 / Candle 0.1.1', '--notes-file', str(notes), *[str(p) for p in sorted(assets.iterdir())])
        verify(base / 'github-draft-download')
        run('release', 'edit', TAG, '--repo', REPO, '--draft=false')
        verify(base / 'github-public-download')
        status = 'published-and-byte-verified'
    with (base / 'github-release.json').open('x') as stream:
        json.dump({'schema': 'ruv-fann.rust-github-publication/1', 'sourceSha': source, 'tagSourceSha': source,
                   'tag': TAG, 'status': status, 'url': f'https://github.com/{REPO}/releases/tag/{TAG}', 'assets': manifest}, stream, indent=2)


if __name__ == '__main__':
    main()
