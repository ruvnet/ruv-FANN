#!/usr/bin/env python3
"""Create the coordinated GitHub release only after public consumers pass."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(*args):
    return subprocess.check_output(['gh', *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = Path(sys.argv[1]).resolve()
    source = os.environ['GITHUB_SHA']
    if (os.environ.get('GITHUB_ACTIONS'), os.environ.get('GITHUB_WORKFLOW'), os.environ.get('GITHUB_REF')) != ('true', 'protected-release', 'refs/heads/main'):
        raise ValueError('GitHub releases require the protected main workflow')
    receipts = {}
    for name in ('cargo-publication', 'npm-publication', 'public-consumers'):
        receipts[name] = json.loads((base / (name + '.json')).read_text())
        if receipts[name]['sourceSha'] != source:
            raise ValueError('Publication source mismatch')
    crates = receipts['cargo-publication']['crates']
    packages = receipts['npm-publication']['packages']
    if len(crates) != 27 or any(r['status'] not in ('already-verified', 'published-and-byte-verified') for r in crates):
        raise ValueError('Incomplete Cargo publication')
    if len(packages) != 2 or any(r['status'] != 'published-and-byte-verified' for r in packages):
        raise ValueError('Incomplete npm publication')
    consumer = receipts['public-consumers']
    if consumer['status'] != 'passed' or any(consumer.get(k) is not True for k in ('npmNormalInstall', 'coreAndSimdTraining', 'persistentCli', 'cudaPublicCompiler', 'cargoPublicDenseTraining')):
        raise ValueError('Public consumer checks did not pass')
    assets = base / 'github-release-assets'
    assets.mkdir()
    candidates = list((base / 'cargo-candidates').rglob('*.crate'))
    candidates += [Path(r['archive']) for r in packages]
    for row in crates + packages:
        matches = [p for p in candidates if digest(p) == row['sha256']]
        if not matches:
            raise ValueError('Missing exact published archive: ' + row['name'])
        target = assets / matches[0].name
        if target.exists():
            raise ValueError('Duplicate asset name')
        shutil.copyfile(matches[0], target)
    for name in receipts:
        shutil.copyfile(base / (name + '.json'), assets / (name + '.json'))
    manifest = {p.name: digest(p) for p in assets.iterdir()}
    (assets / 'SHA256SUMS').write_text(''.join(f'{value}  {name}\n' for name, value in sorted(manifest.items())))
    manifest['SHA256SUMS'] = digest(assets / 'SHA256SUMS')
    notes = base / 'release-notes.md'
    notes.write_text(f'''Coordinated release from PR #202, commit `{source}`.

Includes dense autograd training, the Candle adapter, WASM training, and SQLite/MCP fixes. Publishes 27 verified Cargo crates, `ruv-swarm@2.0.0` and `cuda-wasm@2.0.0`. The native Node addon remains a local platform build.

All attached package archives are the exact bytes verified from the public registries. Fresh public installs passed core/SIMD training, persistent CLI, CUDA compiler and Rust dense training checks. See the attached publication receipts and SHA256SUMS.
''')
    repo, tag = 'ruvnet/ruv-FANN', 'v0.2.2'
    # Never overwrite an existing release or move a tag.
    exists = subprocess.run(['gh', 'api', f'repos/{repo}/git/ref/tags/{tag}'], capture_output=True)
    if exists.returncode == 0:
        raise ValueError('Release tag already exists; inspect before retrying')
    run('release', 'create', tag, '--repo', repo, '--target', source, '--draft', '--title', 'ruv-FANN 0.2.2 — dense training and coordinated packages', '--notes-file', str(notes), *[str(p) for p in sorted(assets.iterdir())])
    def verify(folder):
        run('release', 'download', tag, '--repo', repo, '--dir', str(folder))
        if {p.name: digest(p) for p in folder.iterdir()} != manifest:
            raise ValueError('GitHub release asset byte mismatch')
    verify(base / 'github-draft-download')
    run('release', 'edit', tag, '--repo', repo, '--draft=false')
    verify(base / 'github-public-download')
    (base / 'github-release.json').write_text(json.dumps({'sourceSha': source, 'tag': tag, 'status': 'published-and-byte-verified', 'url': f'https://github.com/{repo}/releases/tag/{tag}', 'assets': manifest}, indent=2))


if __name__ == '__main__':
    main()
