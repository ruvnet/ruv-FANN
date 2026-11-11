#!/usr/bin/env python3
"""Publish verified, immutable Cargo archives through the documented registry API.

Only the protected-release Actions workflow may upload. Default mode is read-only.
https://doc.rust-lang.org/cargo/reference/registry-web-api.html#publish
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tarfile
import time
import tomllib
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
API = 'https://crates.io/api/v1'
USER_AGENT = 'ruv-fann-release (https://github.com/ruvnet/ruv-FANN)'
# static.crates.io can take several minutes to serve a newly uploaded crate.
VISIBILITY_TIMEOUT = 600
VISIBILITY_DELAYS = (5, 10, 20, 30, 60)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def metadata(archive):
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        manifests = [n for n in names if n.count('/') == 1 and n.endswith('/Cargo.toml')]
        if len(manifests) != 1:
            raise ValueError('Expected exactly one normalized manifest')
        prefix = manifests[0].rsplit('/', 1)[0]
        document = tomllib.loads(tar.extractfile(manifests[0]).read().decode())
        package = document['package']
        if prefix != package['name'] + '-' + package['version']:
            raise ValueError('Archive identity mismatch')
        result = {'name': package['name'], 'vers': package['version'],
                  'deps': [], 'features': document.get('features', {}),
                  'authors': package.get('authors', []), 'keywords': package.get('keywords', []),
                  'categories': package.get('categories', []), 'badges': document.get('badges', {})}
        for key in ['description', 'documentation', 'homepage', 'license', 'repository', 'links']:
            result[key] = package.get(key)
        result['license_file'] = package.get('license-file')
        result['rust_version'] = package.get('rust-version')
        readme = package.get('readme')
        result['readme_file'] = readme if isinstance(readme, str) else None
        result['readme'] = tar.extractfile(prefix + '/' + readme).read().decode() if isinstance(readme, str) else None
        for target, table in [(None, document), *document.get('target', {}).items()]:
            for section, kind in [('dependencies', 'normal'), ('dev-dependencies', 'dev'), ('build-dependencies', 'build')]:
                for name, dep in table.get(section, {}).items():
                    dep = {'version': dep} if isinstance(dep, str) else dep
                    if 'version' not in dep or 'path' in dep or 'git' in dep:
                        raise ValueError('Unpublishable dependency: ' + name)
                    result['deps'].append({'name': dep.get('package', name), 'version_req': dep['version'],
                        'features': dep.get('features', []), 'optional': dep.get('optional', False),
                        'default_features': dep.get('default-features', True), 'target': target, 'kind': kind,
                        'registry': dep.get('registry-index'), 'explicit_name_in_toml': name if 'package' in dep else None})
        return result


def candidates(directory):
    receipt_path = directory / 'receipt.json'
    expected = (directory / 'receipt.sha256').read_text().split()[0]
    if sha(receipt_path.read_bytes()) != expected:
        raise ValueError('Receipt hash mismatch')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('verified') is not True or not receipt.get('commands') or any(c['exit_code'] != 0 for c in receipt['commands']):
        raise ValueError('Missing or failed Cargo verification')
    if receipt['live_lock_sha256_before'] != receipt['live_lock_sha256_after']:
        raise ValueError('Source lock changed during verification')
    for entry in json.loads((directory / 'snapshot-files.json').read_text()):
        path = ROOT / entry['path']
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Source path escapes repository')
        if entry['sha256'].startswith('symlink:'):
            actual = 'symlink:' + os.readlink(path)
        else:
            actual = sha(path.read_bytes())
        if actual != entry['sha256']:
            raise ValueError('Source changed after verification: ' + entry['path'])
    order = json.loads((directory / 'publication-order.json').read_text())
    records = {r['name']: r for r in receipt['archives']}
    if len(records) != 27 or len(order) != len(records) or {r['name'] for r in order} != set(records):
        raise ValueError('Incomplete or duplicate publication inventory')
    result = []
    for row in order:
        record = records[row['name']]
        archive = directory / record['archive']
        if not archive.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Archive path escapes candidate directory')
        if record.get('verified') is not True or sha(archive.read_bytes()) != row['archive_sha256'] or record['sha256'] != row['archive_sha256']:
            raise ValueError('Unverified or changed archive')
        data = metadata(archive)
        if (data['name'], data['vers']) != (row['name'], row['version']):
            raise ValueError('Publication identity mismatch')
        result.append((archive, data, row['archive_sha256']))
    return result


def downloaded(name, version):
    url = f'https://static.crates.io/crates/{name}/{name}-{version}.crate'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': USER_AGENT}), timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            return None
        raise


def wait_public(name, version, timeout=VISIBILITY_TIMEOUT, sleep=time.sleep, clock=time.monotonic):
    """Return the public crate bytes, or None if static.crates.io still lacks them past the deadline."""
    deadline = clock() + timeout
    attempt = 0
    while True:
        public = downloaded(name, version)
        if public is not None:
            return public
        remaining = deadline - clock()
        if remaining <= 0:
            return None
        sleep(min(VISIBILITY_DELAYS[min(attempt, len(VISIBILITY_DELAYS) - 1)], remaining))
        attempt += 1


def preflight(items):
    """Read-only: fail before any upload if a candidate's version is already public with other bytes."""
    states, drifted = [], []
    for archive, data, digest in items:
        existing = downloaded(data['name'], data['vers'])
        if existing is None:
            states.append('verified-candidate')
        elif sha(existing) == digest:
            states.append('already-verified')
        else:
            states.append('published-with-different-bytes')
            drifted.append(f"{data['name']} {data['vers']}")
    if drifted:
        raise ValueError('Published versions exist with different bytes (bump them before releasing; nothing was uploaded): ' + ', '.join(drifted))
    return states


def publish(archive, data, digest, token):
    existing = downloaded(data['name'], data['vers'])
    if existing is not None:
        if sha(existing) != digest:
            raise ValueError('Published version exists with different bytes: ' + data['name'])
        return 'already-verified'
    meta = json.dumps(data).encode()
    payload = archive.read_bytes()
    if sha(payload) != digest:
        raise ValueError('Archive changed before upload')
    body = struct.pack('<I', len(meta)) + meta + struct.pack('<I', len(payload)) + payload
    request = urllib.request.Request(API + '/crates/new', data=body, method='PUT', headers={
        'User-Agent': USER_AGENT, 'Authorization': token, 'Content-Type': 'application/octet-stream', 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode().replace(token, '[REDACTED]')
        raise RuntimeError(f'Crates upload HTTP {error.code}: {detail}') from None
    if result.get('errors'):
        raise RuntimeError(str(result['errors']).replace(token, '[REDACTED]'))
    public = wait_public(data['name'], data['vers'])
    if public is None:
        raise RuntimeError(f"Published, but {data['name']} {data['vers']} was not publicly downloadable within {VISIBILITY_TIMEOUT}s; rerun the failed job to verify the bytes")
    if sha(public) != digest:
        raise ValueError('Public bytes differ from verified candidate')
    return 'published-and-byte-verified'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != args.expected_sha or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT):
        raise ValueError('Publication requires the clean reviewed SHA')
    items = candidates(args.artifacts)
    if args.publish and (os.environ.get('GITHUB_ACTIONS') != 'true' or os.environ.get('GITHUB_WORKFLOW') != 'protected-release' or os.environ.get('GITHUB_REF') != 'refs/heads/main' or os.environ.get('GITHUB_SHA') != head):
        raise ValueError('Uploads require the protected main-branch workflow')
    token = os.environ.get('CARGO_REGISTRY_TOKEN')
    if args.publish and not token:
        raise ValueError('Missing Cargo registry token')
    report = {'sourceSha': head, 'publicationRequested': args.publish, 'crates': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    states = None if args.publish else preflight(items)
    with args.output.open('x') as stream:
        for index, (archive, data, digest) in enumerate(items):
            state = publish(archive, data, digest, token) if args.publish else states[index]
            row = {'name': data['name'], 'version': data['vers'], 'sha256': digest, 'status': state}
            print(json.dumps(row), flush=True)
            report['crates'].append(row)
            stream.seek(0); json.dump(report, stream, indent=2); stream.truncate(); stream.flush()
            if args.publish: time.sleep(2)


if __name__ == '__main__':
    main()
