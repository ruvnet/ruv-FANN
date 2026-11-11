#!/usr/bin/env python3
"""Download canonical Cargo bytes and exercise real public APIs in a virgin Cargo home."""
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cargo_publisher', ROOT / 'scripts/publish-candidates.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
REQUIRED_CASES = ['candle-public-inference', 'sgd-three-step-gradients', 'adamw-three-step-gradients']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(receipt, publication, source):
    if receipt.get('schema') != 'ruv-fann.public-rust-consumer/1' or receipt.get('sourceSha') != source or publication.get('sourceSha') != source:
        raise ValueError('Public Rust consumer source/schema mismatch')
    commands, result = receipt.get('commands', []), receipt.get('result', {})
    if (len(commands) != 3 or any(row.get('exit_code') != 0 or not row.get('logSha256') for row in commands)
            or result.get('cases') != REQUIRED_CASES or result.get('tests') != 3
            or not isinstance(result.get('checks'), int) or result['checks'] < 100
            or result.get('failures') != 0 or result.get('skipped') != 0):
        raise ValueError('Missing, skipped, empty or failed actual public Rust checks')
    if (commands[0].get('command') != ['cargo', 'generate-lockfile']
            or commands[1].get('command') != ['cargo', 'metadata', '--locked', '--format-version=1']
            or commands[2].get('command', [])[:6] != ['cargo', 'run', '--locked', '--release', '--quiet', '--']
            or len(commands[2]['command']) != 7
            or receipt.get('consumerSourceSha256') != digest(ROOT / 'scripts/fixtures/public-rust-consumer.rs')
            or any(not re.fullmatch('[a-f0-9]{64}', row['logSha256']) for row in commands)):
        raise ValueError('Public consumer command/source identity mismatch')
    rows = receipt.get('crates', [])
    published = publication.get('crates', [])
    if len(rows) != 27 or len({row['name'] for row in rows}) != 27 or {
        (r['name'], r['version'], r['sha256']) for r in rows
    } != {(r['name'], r['version'], r['sha256']) for r in published}:
        raise ValueError('Incomplete canonical public Cargo archive identity')
    expected = {'ruv-fann': '0.2.3', 'ruv-fann-candle': '0.1.1'}
    if receipt.get('versions') != expected or not re.fullmatch('[a-f0-9]{64}', receipt.get('fixtureSha256', '')):
        raise ValueError('Wrong public consumer versions or missing actual Candle fixture')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    base = args.directory.resolve()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    publication = json.loads((base / 'cargo-publication.json').read_text())
    if publication['sourceSha'] != source:
        raise ValueError('Cargo publication source mismatch')
    fixture = Path(os.environ['RUV_FANN_CANDLE_FIXTURE']).resolve()
    oracle = json.loads(fixture.read_text())
    if oracle['oracle'] != 'candle-core/candle-nn 0.11.1 CPU f64':
        raise ValueError('Expected actual pinned Candle CPU f64 fixture')
    archives = base / 'cargo-public-archives'
    archives.mkdir()
    rows = []
    for row in publication['crates']:
        if row['status'] not in ('already-verified', 'published-and-byte-verified', publisher.LOCKFILE_ONLY_DRIFT):
            raise ValueError('Unpublished Cargo candidate')
        payload = publisher.downloaded(row['name'], row['version'])
        if payload is None or publisher.sha(payload) != row['sha256']:
            raise ValueError('Canonical public crate differs: ' + row['name'])
        archive = archives / f"{row['name']}-{row['version']}.crate"
        archive.write_bytes(payload)
        data = publisher.metadata(archive)
        if (data['name'], data['vers']) != (row['name'], row['version']):
            raise ValueError('Public crate identity mismatch')
        rows.append({key: row[key] for key in ('name', 'version', 'sha256')})
    rust = base / 'rust-public-consumer'
    (rust / 'src').mkdir(parents=True)
    (rust / 'Cargo.toml').write_text('''[package]
name="ruv-fann-public-release-consumer"
version="0.0.0"
edition="2021"
[dependencies]
ruv-fann={version="=0.2.3",default-features=false,features=["std","serde"]}
ruv-fann-candle="=0.1.1"
candle-core={version="=0.11.1",default-features=false}
candle-nn={version="=0.11.1",default-features=false}
serde_json={version="=1.0.154",features=["float_roundtrip"]}
''')
    shutil.copyfile(ROOT / 'scripts/fixtures/public-rust-consumer.rs', rust / 'src/main.rs')
    env = os.environ.copy()
    env.update(CARGO_HOME=str(base / 'public-cargo-home'), CARGO_TARGET_DIR=str(base / 'public-cargo-target'), CARGO_BUILD_JOBS='2')
    env.pop('CARGO_REGISTRY_TOKEN', None)
    commands = []
    def run(argv):
        result = subprocess.run(argv, cwd=rust, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log = base / f'public-rust-{len(commands)}.log'
        log.write_text(result.stdout + '\n' + result.stderr)
        commands.append({'command': argv, 'exit_code': result.returncode, 'logSha256': digest(log)})
        if result.returncode:
            raise RuntimeError('Public Rust consumer failed; see ' + str(log))
        return result.stdout
    run(['cargo', 'generate-lockfile'])
    metadata = json.loads(run(['cargo', 'metadata', '--locked', '--format-version=1']))
    lock = tomllib.loads((rust / 'Cargo.lock').read_text())
    versions = {'ruv-fann': '0.2.3', 'ruv-fann-candle': '0.1.1'}
    for name, version in versions.items():
        package = [p for p in metadata['packages'] if p['name'] == name]
        locked = [p for p in lock['package'] if p['name'] == name]
        public = next(r for r in rows if r['name'] == name and r['version'] == version)
        if (len(package) != 1 or package[0]['version'] != version
                or package[0]['source'] != 'registry+https://github.com/rust-lang/crates.io-index'
                or len(locked) != 1 or locked[0].get('checksum') != public['sha256']):
            raise ValueError('Consumer did not resolve exact canonical public bytes: ' + name)
    result = json.loads(run(['cargo', 'run', '--locked', '--release', '--quiet', '--', str(fixture)]))
    receipt = {'schema': 'ruv-fann.public-rust-consumer/1', 'sourceSha': source,
               'fixtureSha256': digest(fixture), 'consumerSourceSha256': digest(rust / 'src/main.rs'),
               'lockSha256': digest(rust / 'Cargo.lock'), 'versions': versions,
               'commands': commands, 'result': result, 'crates': rows}
    validate(receipt, publication, source)
    with (base / 'public-rust-consumers.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)


if __name__ == '__main__':
    main()
