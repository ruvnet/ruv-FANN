#!/usr/bin/env python3
"""Create the coordinated GitHub release only after public consumers pass."""
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys


def run(*args):
    return subprocess.check_output(['gh', *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ARCHIVE_SUFFIXES = ('.crate', '.tgz')


def write_receipt(base, source, repo, tag, status, manifest):
    (base / 'github-release.json').write_text(json.dumps({'sourceSha': source, 'tag': tag, 'status': status, 'url': f'https://github.com/{repo}/releases/tag/{tag}', 'assets': manifest}, indent=2))


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
    if len(crates) != 27 or any(r['status'] not in ('already-verified', 'published-and-byte-verified', 'lockfile-only-drift-library') for r in crates):
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
    versions = {r['name']: r.get('version') for r in packages}
    swarm = versions.get('ruv-swarm')
    if not isinstance(swarm, str) or not re.fullmatch(r'\d+\.\d+\.\d+', swarm):
        raise ValueError('Release tag requires the published ruv-swarm npm version')
    # One release per ruv-swarm npm version. The npm byte check already
    # forces a new ruv-swarm version whenever its packaged bytes change.
    repo, tag = 'ruvnet/ruv-FANN', f'ruv-fann-registry-v{swarm}'
    archives = {name: value for name, value in manifest.items() if name.endswith(ARCHIVE_SUFFIXES)}
    # Never overwrite an existing release or move a tag. A rerun of the same
    # version with identical package archives is already released; receipts
    # are excluded from the comparison because they embed the source commit.
    release = subprocess.run(['gh', 'release', 'view', tag, '--repo', repo, '--json', 'isDraft'], capture_output=True, text=True)
    ref = subprocess.run(['gh', 'api', f'repos/{repo}/git/ref/tags/{tag}'], capture_output=True)
    if release.returncode == 0 or ref.returncode == 0:
        if release.returncode != 0 or json.loads(release.stdout).get('isDraft') is not False:
            raise ValueError(f'Tag {tag} exists without a published release; inspect before retrying')
        existing = base / 'github-existing-download'
        run('release', 'download', tag, '--repo', repo, '--dir', str(existing))
        published = {p.name: digest(p) for p in existing.iterdir() if p.name.endswith(ARCHIVE_SUFFIXES)}
        if published != archives:
            raise ValueError(f'Release {tag} exists with different package archives; bump the ruv-swarm npm version')
        write_receipt(base, source, repo, tag, 'already-released', manifest)
        return
    published_now = [r for r in crates if r['status'] == 'published-and-byte-verified']
    listing = ', '.join(f"`{r['name']}@{r.get('version', '?')}`" for r in packages)
    new_crates = ', '.join(f"`{r['name']} {r.get('version', '?')}`" for r in published_now) or 'none (all already on crates.io)'
    kept = [r for r in crates if r['status'] == 'lockfile-only-drift-library']
    lock_note = (' Library crates whose only difference from crates.io is the embedded Cargo.lock were not republished (cargo ignores a dependency\'s lockfile); their public bytes are attached: '
                 + ', '.join(f"`{r['name']} {r.get('version', '?')}`" for r in kept) + '.') if kept else ''
    notes = base / 'release-notes.md'
    notes.write_text(f'''Coordinated registry release from commit `{source}`.

npm packages: {listing}. Cargo crates verified: {len(crates)}; newly published: {new_crates}.{lock_note}

All attached package archives are the exact bytes verified from the public registries. Fresh public installs passed core/SIMD training, persistent CLI, CUDA compiler and Rust dense training checks. See the attached publication receipts and SHA256SUMS.
''')
    run('release', 'create', tag, '--repo', repo, '--target', source, '--draft', '--title', f'ruv-FANN registry release (ruv-swarm {swarm})', '--notes-file', str(notes), *[str(p) for p in sorted(assets.iterdir())])
    def verify(folder):
        run('release', 'download', tag, '--repo', repo, '--dir', str(folder))
        if {p.name: digest(p) for p in folder.iterdir()} != manifest:
            raise ValueError('GitHub release asset byte mismatch')
    verify(base / 'github-draft-download')
    run('release', 'edit', tag, '--repo', repo, '--draft=false')
    verify(base / 'github-public-download')
    write_receipt(base, source, repo, tag, 'published-and-byte-verified', manifest)


if __name__ == '__main__':
    main()
