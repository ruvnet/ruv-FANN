#!/usr/bin/env python3
"""Source-bound tensor crate qualification and protected exact-archive publication."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
import tomllib
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'crates/wsp-tensor'
REPO = 'ruvnet/ruv-FANN'
NAME, VERSION = 'ruv-fann-wsp-tensor', '0.1.0'
ARCHIVE = f'{NAME}-{VERSION}.crate'
TAG = f'{NAME}-v{VERSION}'
WORKFLOW = '.github/workflows/tensor-wsp.yml'

def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

registry = load('tensor_registry', 'publish-candidates.py')
archive_tools = load('tensor_archive', 'verify-release.py')
ci_tools = load('tensor_ci', 'select-rust-candidate.py')
sha = archive_tools.digest

def gh(endpoint):
    return ci_tools.gh(endpoint)

def identity(source):
    if not re.fullmatch('[a-f0-9]{40}', source):
        raise ValueError('Full source SHA required')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != source:
        raise ValueError('Checkout SHA mismatch')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT):
        raise ValueError('Clean immutable checkout required')

def protected(source):
    identity(source)
    keys = ('GITHUB_ACTIONS', 'GITHUB_WORKFLOW', 'GITHUB_REF', 'GITHUB_SHA')
    if tuple(os.environ.get(k) for k in keys) != ('true', 'protected-tensor-release', 'refs/heads/main', source):
        raise ValueError('Only protected tensor workflow may publish')
    if gh(f'repos/{REPO}/git/ref/heads/main')['object']['sha'] != source:
        raise ValueError('Main moved')
    if gh(f'repos/{REPO}/issues?state=open&labels=release-blocker&per_page=1'):
        raise ValueError('Open release blockers')

def command_plan(base):
    packed = base / 'extracted' / f'{NAME}-{VERSION}'
    target = base / 'build'
    return [
        ('fmt', ['cargo', 'fmt', '--check'], PACKAGE, None),
        ('clippy', ['cargo', 'clippy', '--locked', '--all-targets', '--', '-D', 'warnings'], PACKAGE, None),
        ('source-tests', ['cargo', 'test', '--locked', '--test', 'checkpoints', '--', '--test-threads=1'], PACKAGE, 'rust'),
        ('msrv', ['cargo', '+1.81.0', 'check', '--locked', '--all-targets'], PACKAGE, None),
        ('audit', ['cargo', 'audit', '--file', 'Cargo.lock', '--deny', 'warnings'], PACKAGE, None),
        ('python-tests', ['python3', '-m', 'unittest', 'discover', '-s', 'scripts', '-p', 'test_*.py', '-v'], PACKAGE, 'python'),
        ('package', ['cargo', 'package', '--locked', '--no-verify', '--target-dir', str(base / 'package')], PACKAGE, None),
        ('packed-tests', ['cargo', 'test', '--locked', '--test', 'checkpoints', '--', '--test-threads=1'], packed, 'rust'),
        ('packed-build', ['cargo', 'build', '--release', '--locked'], packed, None),
        ('packed-cli', ['python3', 'scripts/e2e.py'], packed, 'cli'),
    ]

def evidence(output, kind):
    if kind == 'rust':
        summaries = re.findall(r'^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out;', output, re.M)
        if len(summaries) != 1 or int(summaries[0][0]) < 10 or any(map(int, summaries[0][1:])):
            raise ValueError('Nonzero complete tensor integration suite required')
        return {'testsPassed': int(summaries[0][0])}
    if kind == 'python':
        rows = re.findall(r'^Ran (\d+) tests? in ', output, re.M)
        if len(rows) != 1 or int(rows[0]) == 0 or not re.search(r'^OK\s*$', output, re.M):
            raise ValueError('Nonzero successful Python tests with no skips required')
        return {'testsPassed': int(rows[0])}
    if kind == 'cli':
        data = json.loads(output)
        if (data.get('schema') != 'ruv-fann.tensor-e2e/1' or data.get('roundtrip') != 'passed'
                or data.get('overwrite_rejected') is not True or data.get('pan2_inference') != 'not_run'):
            raise ValueError('Actual independent CLI roundtrip missing')
        return data
    return {}

def portable(value, base):
    text = str(value)
    for prefix, alias in ((base, '${CANDIDATE}'), (ROOT, '${SOURCE}')):
        if text == str(prefix) or text.startswith(str(prefix) + '/'):
            return alias + text[len(str(prefix)):]
    return text

def execute(role, argv, cwd, kind, base, commands):
    env = dict(os.environ, CARGO_TARGET_DIR=str(base / 'build'), CARGO_BUILD_JOBS='2', CARGO_INCREMENTAL='0',
               WSP_TENSOR_BIN=str(base / 'build/release' / NAME), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = base / f'{role}.log'
    log.write_text(result.stdout)
    if result.returncode:
        print(result.stdout[-12000:], flush=True)
        raise ValueError(f'{role} failed ({result.returncode})')
    commands.append({'role': role, 'argv': [portable(x, base) for x in argv], 'cwd': portable(cwd, base),
                     'log': log.name, 'logSha256': sha(log), 'exitCode': result.returncode,
                     'evidence': evidence(result.stdout, kind)})
    print(f'{role}: passed', flush=True)

def normalized_manifest(actual):
    source = tomllib.loads((PACKAGE / 'Cargo.toml').read_text())
    expected = {key: value for key, value in source.items() if key != 'workspace'}
    expected['package'] = dict(source['package'], build=False, autolib=False, autobins=False,
                               autoexamples=False, autotests=False, autobenches=False)
    for section in ('dependencies', 'dev-dependencies', 'build-dependencies'):
        if section in expected:
            expected[section] = {key: {'version': value} if isinstance(value, str) else value
                                 for key, value in expected[section].items()}
    expected['lib'] = {'name': NAME.replace('-', '_'), 'path': 'src/lib.rs'}
    expected['bin'] = [{'name': NAME, 'path': 'src/main.rs'}]
    expected['test'] = [{'name': path.stem, 'path': 'tests/' + path.name}
                        for path in sorted((PACKAGE / 'tests').glob('*.rs'))]
    if actual != expected:
        raise ValueError('Normalized Cargo manifest differs from reviewed package semantics')


def archive_binding(archive, source):
    info = archive_tools.archive_info(archive)
    if (info['name'], info['version']) != (NAME, VERSION):
        raise ValueError('Package generation mismatch')
    with tarfile.open(archive) as stream:
        entries = stream.getmembers()
        if len({m.name for m in entries}) != len(entries):
            raise ValueError('Duplicate Cargo archive member')
        members = {m.name: m for m in entries if m.isfile()}
        prefix = f'{NAME}-{VERSION}/'
        normalized_manifest(tomllib.loads(stream.extractfile(prefix + 'Cargo.toml').read().decode()))
        vcs = json.load(stream.extractfile(prefix + '.cargo_vcs_info.json'))
        if vcs['git'].get('sha1') != source or vcs['git'].get('dirty', False) or vcs.get('path_in_vcs') != 'crates/wsp-tensor':
            raise ValueError('Archive VCS identity mismatch')
        for name, member in members.items():
            relative = name.removeprefix(prefix)
            if relative in ('Cargo.toml', '.cargo_vcs_info.json'):
                continue
            original = 'Cargo.toml' if relative == 'Cargo.toml.orig' else relative
            path = PACKAGE / original
            if not path.is_file() or path.is_symlink() or stream.extractfile(member).read() != path.read_bytes():
                raise ValueError('Archive/source payload mismatch: ' + relative)
        tracked = subprocess.check_output(['git', 'ls-files', '--', 'crates/wsp-tensor'], cwd=ROOT, text=True).splitlines()
        expected = {p.removeprefix('crates/wsp-tensor/') for p in tracked
                    if p.startswith(('crates/wsp-tensor/src/', 'crates/wsp-tensor/tests/', 'crates/wsp-tensor/scripts/'))
                    or p.rsplit('/', 1)[-1] in ('Cargo.lock', 'README.md', 'LICENSE-MIT', 'LICENSE-APACHE')}
        actual = {name.removeprefix(prefix) for name in members}
        if not expected.issubset(actual):
            raise ValueError('Tracked package source absent from archive')
    return info

def validate(base, source):
    identity(source)
    receipt = json.loads((base / 'candidate.json').read_text())
    if receipt.get('schema') != 'ruv-fann.tensor-candidate/1' or receipt.get('sourceSha') != source:
        raise ValueError('Candidate source/schema mismatch')
    if receipt.get('name') != NAME or receipt.get('version') != VERSION or receipt.get('archive') != ARCHIVE:
        raise ValueError('Candidate package mismatch')
    archive = base / ARCHIVE
    archive_binding(archive, source)
    if sha(archive) != receipt['archiveSha256']:
        raise ValueError('Archive seal mismatch')
    commands = receipt['commands']
    plan = command_plan(base)
    if len(commands) != len(plan):
        raise ValueError('Incomplete command proof')
    for row, (role, argv, cwd, kind) in zip(commands, plan):
        if type(row.get('exitCode')) is not int:
            raise ValueError('Command exit must be an integer')
        if (row.get('role'), row.get('argv'), row.get('cwd'), row.get('exitCode'), row.get('log')) != (
                role, [portable(x, base) for x in argv], portable(cwd, base), 0, f'{role}.log'):
            raise ValueError('Command proof mismatch')
        log = base / row['log']
        if sha(log) != row['logSha256'] or evidence(log.read_text(), kind) != row['evidence']:
            raise ValueError('Log evidence mismatch')
    return receipt

def seal(base, source):
    identity(source)
    base.mkdir(parents=True, exist_ok=False)
    commands = []
    for role, argv, cwd, kind in command_plan(base):
        execute(role, argv, cwd, kind, base, commands)
        if role == 'package':
            shutil.copyfile(base / 'package/package' / ARCHIVE, base / ARCHIVE)
            archive_binding(base / ARCHIVE, source)
            archive_tools.extract_root(base / ARCHIVE, base / 'extracted')
    identity(source)
    receipt = {'schema': 'ruv-fann.tensor-candidate/1', 'sourceSha': source, 'name': NAME, 'version': VERSION,
               'archive': ARCHIVE, 'archiveSha256': sha(base / ARCHIVE), 'commands': commands,
               'pan2GpuInference': 'not_run', 'behavioralImprovement': 'not_established'}
    (base / 'candidate.json').write_text(json.dumps(receipt, indent=2) + '\n')
    validate(base, source)

def validate_ci(run, jobs, artifacts, source):
    if (run.get('head_sha') != source or run.get('event') != 'push' or not run.get('head_branch', '').startswith('release/')
            or run.get('path') != WORKFLOW or run.get('status') != 'completed' or run.get('conclusion') != 'success'):
        raise ValueError('Successful exact-source release-branch tensor workflow required')
    names = {j.get('name') for j in jobs}
    if not {'qualify', 'portability (windows-latest)', 'portability (macos-latest)'}.issubset(names) or any(
            j.get('status') != 'completed' or j.get('conclusion') != 'success' for j in jobs):
        raise ValueError('Missing/pending/failed tensor qualification job')
    matches = [a for a in artifacts if a.get('name') == f'tensor-candidate-{source}' and not a.get('expired')]
    if len(matches) != 1 or not re.fullmatch('sha256:[a-f0-9]{64}', matches[0].get('digest', '')):
        raise ValueError('Unique digest-bound candidate required')
    return matches[0]

def download(base, source):
    protected(source)
    runs = ci_tools.pages(f'repos/{REPO}/actions/workflows/tensor-wsp.yml/runs?head_sha={source}&event=push', 'workflow_runs')
    runs = [r for r in runs if r.get('head_branch', '').startswith('release/') and r.get('conclusion') == 'success']
    if not runs:
        raise ValueError('No successful tensor preflight')
    run = max(runs, key=lambda r: r['id'])
    jobs = ci_tools.pages(f'repos/{REPO}/actions/runs/{run["id"]}/jobs?filter=latest', 'jobs')
    artifacts = ci_tools.pages(f'repos/{REPO}/actions/runs/{run["id"]}/artifacts', 'artifacts')
    item = validate_ci(run, jobs, artifacts, source)
    # Preserve the repository-wide pre-main regression gate as well.
    core_runs = ci_tools.pages(f'repos/{REPO}/actions/workflows/ci.yml/runs?head_sha={source}&event=push', 'workflow_runs')
    core_runs = [r for r in core_runs if r.get('head_branch', '').startswith('release/') and r.get('conclusion') == 'success']
    if not core_runs:
        raise ValueError('No full exact-source repository CI')
    core = max(core_runs, key=lambda r: r['id'])
    core_receipt = ci_tools.validate_run(core,
        ci_tools.pages(f'repos/{REPO}/actions/runs/{core["id"]}/jobs?filter=latest', 'jobs'),
        ci_tools.pages(f'repos/{REPO}/actions/runs/{core["id"]}/artifacts', 'artifacts'), source)
    base.mkdir(parents=True, exist_ok=False)
    path = base / 'candidate.zip'
    with path.open('xb') as stream:
        subprocess.run(['gh', 'api', f'repos/{REPO}/actions/artifacts/{item["id"]}/zip?release_probe={time.time_ns()}'], stdout=stream, check=True)
    if 'sha256:' + sha(path) != item['digest']:
        raise ValueError('GitHub ZIP seal mismatch')
    with zipfile.ZipFile(path) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n for n in names):
            raise ValueError('Unsafe candidate ZIP')
        stream.extractall(base / 'candidate')
    validate(base / 'candidate', source)
    (base / 'ci.json').write_text(json.dumps({'sourceSha': source, 'run': run, 'jobs': jobs, 'artifact': item, 'repositoryCi': core_receipt}, indent=2))

def validated_inputs(base, source):
    protected(source)
    ci = json.loads((base / 'ci.json').read_text())
    if ci.get('sourceSha') != source:
        raise ValueError('Downloaded CI source mismatch')
    run_id = ci['run']['id']
    run = gh(f'repos/{REPO}/actions/runs/{run_id}')
    item = validate_ci(run,
        ci_tools.pages(f'repos/{REPO}/actions/runs/{run_id}/jobs?filter=latest', 'jobs'),
        ci_tools.pages(f'repos/{REPO}/actions/runs/{run_id}/artifacts', 'artifacts'), source)
    if item['id'] != ci['artifact']['id'] or item['digest'] != ci['artifact']['digest']:
        raise ValueError('Downloaded artifact differs from accepted CI')
    archive = base / 'candidate.zip'
    if 'sha256:' + sha(archive) != item['digest']:
        raise ValueError('Original GitHub ZIP seal changed')
    expected = {ARCHIVE, 'candidate.json', *[role + '.log' for role, *_ in command_plan(base / 'candidate')]}
    with zipfile.ZipFile(archive) as stream:
        if set(stream.namelist()) != expected or len(stream.namelist()) != len(expected):
            raise ValueError('Unexpected candidate ZIP inventory')
        for name in expected:
            path = base / 'candidate' / name
            if path.is_symlink() or path.read_bytes() != stream.read(name):
                raise ValueError('Candidate bytes changed after accepted CI download')
    core_id = ci['repositoryCi']['runId']
    ci_tools.validate_run(gh(f'repos/{REPO}/actions/runs/{core_id}'),
        ci_tools.pages(f'repos/{REPO}/actions/runs/{core_id}/jobs?filter=latest', 'jobs'),
        ci_tools.pages(f'repos/{REPO}/actions/runs/{core_id}/artifacts', 'artifacts'), source)
    return validate(base / 'candidate', source)


def publish(base, source):
    receipt = validated_inputs(base, source)
    archive = base / 'candidate' / ARCHIVE
    digest = receipt['archiveSha256']
    public = registry.downloaded(NAME, VERSION)
    if public is not None and registry.sha(public) != digest:
        raise ValueError('Immutable version already has different bytes')
    if public is None:
        registry.confirm_version_absent(NAME, VERSION)
    token = os.environ.get('CARGO_REGISTRY_TOKEN')
    if not token:
        raise ValueError('Missing Cargo token')
    status = registry.publish(archive, registry.metadata(archive), digest, token)
    if status not in ('already-verified', 'published-and-byte-verified'):
        raise ValueError('Exact byte publication required')
    (base / 'publication.json').write_text(json.dumps({'sourceSha': source, 'archiveSha256': digest, 'name': NAME, 'version': VERSION, 'status': status}, indent=2))

def verify_public(base, source):
    receipt = validated_inputs(base, source)
    public = registry.downloaded(NAME, VERSION)
    if public is None or registry.sha(public) != receipt['archiveSha256']:
        raise ValueError('Canonical public archive mismatch')
    (base / 'public.crate').write_bytes(public)
    fresh = base / 'fresh'
    fresh.mkdir()
    packed = archive_tools.extract_root(base / 'public.crate', fresh / 'extracted')
    commands = []
    for role, argv, kind in [
        ('public-tests', ['cargo', 'test', '--locked', '--test', 'checkpoints', '--', '--test-threads=1'], 'rust'),
        ('public-build', ['cargo', 'build', '--release', '--locked'], None),
        ('public-cli', ['python3', 'scripts/e2e.py'], 'cli')]:
        execute(role, argv, packed, kind, fresh, commands)
    # Resolve as a real public registry consumer in a fresh Cargo home, then verify its lock checksum.
    consumer = fresh / 'consumer'
    consumer.mkdir()
    (consumer / 'src').mkdir()
    (consumer / 'Cargo.toml').write_text(f'[package]\nname="tensor-public-consumer"\nversion="0.0.0"\nedition="2021"\n[dependencies]\n{NAME}="={VERSION}"\n')
    fixture = fresh / 'registry-fixture.safetensors'
    import struct
    header = json.dumps({'w': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    header += b' ' * (-len(header) % 8)
    fixture.write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<f', 0.25))
    (consumer / 'src/main.rs').write_text('''use ruv_fann_wsp_tensor::{apply,fingerprint,rollback,Edit,Patch};
use std::{path::Path,fs};
fn main(){let args:Vec<_>=std::env::args().collect();let base=Path::new(&args[1]);let candidate=base.with_extension("candidate");let restored=base.with_extension("restored");
let original=fs::read(base).unwrap();let p=Patch{schema:"ruv-fann.tensor-patch/1".into(),base_sha256:fingerprint(base).unwrap(),edits:vec![Edit{tensor:"w".into(),index:0,value:0.75}]};
let receipt=apply(base,&candidate,&p).unwrap();let bytes=fs::read(&candidate).unwrap();assert_eq!(&bytes[bytes.len()-4..],&0.75f32.to_le_bytes());
assert_eq!(fs::read(base).unwrap(),original);assert!(apply(base,&candidate,&p).is_err());rollback(&candidate,&restored,&receipt).unwrap();assert_eq!(fs::read(restored).unwrap(),original);
println!("Public tensor API consumer passed");}''')
    env = dict(os.environ, CARGO_HOME=str(fresh / 'cargo-home'), CARGO_TARGET_DIR=str(fresh / 'consumer-target'))
    result = subprocess.run(['cargo', 'run', '--manifest-path', str(consumer / 'Cargo.toml'), '--', str(fixture)], cwd=consumer, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (fresh / 'registry-consumer.log').write_text(result.stdout)
    if result.returncode or result.stdout.splitlines().count('Public tensor API consumer passed') != 1:
        raise ValueError('Fresh registry consumer failed')
    import tomllib
    rows = tomllib.loads((consumer / 'Cargo.lock').read_text())['package']
    rows = [r for r in rows if r['name'] == NAME]
    if len(rows) != 1 or rows[0].get('checksum') != receipt['archiveSha256'] or rows[0].get('source') != 'registry+https://github.com/rust-lang/crates.io-index':
        raise ValueError('Fresh registry checksum/source mismatch')
    installed = fresh / 'installed'
    result = subprocess.run(['cargo', 'install', '--locked', '--root', str(installed), '--version', '=' + VERSION, NAME],
        cwd=fresh, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (fresh / 'registry-install.log').write_text(result.stdout)
    if result.returncode:
        raise ValueError('Fresh registry CLI install failed')
    result = subprocess.run(['python3', str(packed / 'scripts/e2e.py')], env=dict(env, WSP_TENSOR_BIN=str(installed / 'bin' / NAME)),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (fresh / 'registry-cli.log').write_text(result.stdout)
    if result.returncode:
        raise ValueError('Fresh installed registry CLI failed')
    installed_evidence = evidence(result.stdout, 'cli')
    (base / 'public-consumers.json').write_text(json.dumps({'sourceSha': source, 'archiveSha256': receipt['archiveSha256'], 'commands': commands,
        'registryConsumer': {'exitCode': 0, 'logSha256': sha(fresh / 'registry-consumer.log'), 'checksum': rows[0]['checksum']},
        'installedCli': {'exitCode': 0, 'installLogSha256': sha(fresh / 'registry-install.log'), 'logSha256': sha(fresh / 'registry-cli.log'), 'evidence': installed_evidence}}, indent=2))

def validate_public(base, source):
    receipt = validated_inputs(base, source)
    digest = receipt['archiveSha256']
    published = json.loads((base / 'publication.json').read_text())
    proof = json.loads((base / 'public-consumers.json').read_text())
    if (published.get('sourceSha') != source or published.get('archiveSha256') != digest
            or published.get('name') != NAME or published.get('version') != VERSION
            or published.get('status') not in ('already-verified', 'published-and-byte-verified')
            or proof.get('sourceSha') != source or proof.get('archiveSha256') != digest):
        raise ValueError('Publication proof mismatch')
    if sha(base / 'public.crate') != digest:
        raise ValueError('Public archive changed')
    fresh = base / 'fresh'
    roles = [('public-tests', 'rust'), ('public-build', None), ('public-cli', 'cli')]
    if len(proof['commands']) != len(roles):
        raise ValueError('Incomplete public qualification')
    for row, (role, kind) in zip(proof['commands'], roles):
        log = fresh / (role + '.log')
        if (row.get('role') != role or type(row.get('exitCode')) is not int or row['exitCode'] != 0
                or row.get('log') != log.name or sha(log) != row['logSha256']
                or evidence(log.read_text(), kind) != row['evidence']):
            raise ValueError('Public command proof mismatch')
    registry_proof = proof['registryConsumer']
    if (type(registry_proof.get('exitCode')) is not int or registry_proof['exitCode'] != 0
            or registry_proof.get('checksum') != digest
            or sha(fresh / 'registry-consumer.log') != registry_proof['logSha256']
            or (fresh / 'registry-consumer.log').read_text().splitlines().count('Public tensor API consumer passed') != 1):
        raise ValueError('Public registry consumer proof mismatch')
    cli = proof['installedCli']
    if (type(cli.get('exitCode')) is not int or cli['exitCode'] != 0
            or sha(fresh / 'registry-install.log') != cli['installLogSha256']
            or sha(fresh / 'registry-cli.log') != cli['logSha256']
            or evidence((fresh / 'registry-cli.log').read_text(), 'cli') != cli['evidence']):
        raise ValueError('Public installed CLI proof mismatch')
    return receipt


def validate_empty_log_assets(assets, originals, source, archive_sha):
    empty = sorted(name for name, (path, _) in originals.items() if path.stat().st_size == 0)
    encoding, container = assets / 'empty-log-encoding.json', assets / 'empty-logs.zip'
    if not empty:
        if encoding.exists() or container.exists():
            raise ValueError('Unexpected empty-log encoding')
        return
    entries = [{'asset': name, 'origin': originals[name][1], 'size': 0, 'sha256': hashlib.sha256(b'').hexdigest(),
                'container': container.name, 'member': name}
               for name in empty]
    expected = {'schema': 'ruv-fann.tensor-release-empty-logs/1', 'sourceSha': source,
                'archiveSha256': archive_sha, 'containerSha256': sha(container), 'entries': entries}
    if json.loads(encoding.read_text()) != expected:
        raise ValueError('Empty-log encoding/source/hash mismatch')
    with zipfile.ZipFile(container) as stream:
        if (stream.namelist() != empty or any(stream.read(name) != b'' for name in empty)
                or any((info.external_attr >> 16) & 0o170000 != 0o100000 for info in stream.infolist())):
            raise ValueError('Empty-log container does not preserve original bytes')
    if any((assets / name).exists() for name in empty):
        raise ValueError('Zero-length log must use the lossless container')


def github_assets(base, source, receipt):
    assets = base / 'release-assets'
    assets.mkdir()
    shutil.copyfile(base / 'public.crate', assets / ARCHIVE)
    for name in ('ci.json', 'publication.json', 'public-consumers.json'):
        shutil.copyfile(base / name, assets / name)
    originals = {}
    for folder, prefix in [('candidate', 'candidate-'), ('fresh', '')]:
        for path in sorted((base / folder).glob('*.log')):
            name = prefix + path.name
            if name in originals:
                raise ValueError('Colliding release log name: ' + name)
            originals[name] = (path, folder + '/' + path.name)
            shutil.copyfile(path, assets / name)
    shutil.copyfile(base / 'candidate/candidate.json', assets / 'candidate.json')
    empty = sorted(name for name, (path, _) in originals.items() if path.stat().st_size == 0)
    if empty:
        container = assets / 'empty-logs.zip'
        # Stored entries and explicit metadata keep the lossless ZIP byte deterministic.
        with zipfile.ZipFile(container, 'w', compression=zipfile.ZIP_STORED) as stream:
            for name in empty:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100600 << 16
                stream.writestr(info, originals[name][0].read_bytes())
                (assets / name).unlink()
        entries = [{'asset': name, 'origin': originals[name][1], 'size': 0,
                    'sha256': hashlib.sha256(b'').hexdigest(), 'container': container.name, 'member': name} for name in empty]
        (assets / 'empty-log-encoding.json').write_text(json.dumps({
            'schema': 'ruv-fann.tensor-release-empty-logs/1', 'sourceSha': source,
            'archiveSha256': receipt['archiveSha256'], 'containerSha256': sha(container), 'entries': entries}, indent=2) + '\n')
    validate_empty_log_assets(assets, originals, source, receipt['archiveSha256'])
    manifest = {p.name: sha(p) for p in assets.iterdir()}
    (assets / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(manifest.items())))
    manifest['SHA256SUMS'] = sha(assets / 'SHA256SUMS')
    if any(not p.stat().st_size for p in assets.iterdir()):
        raise ValueError('Cannot upload a zero-length release asset')
    return assets, manifest


def github_asset_bytes(identifier):
    # gh's terminal sanitizer otherwise rejects legitimate ANSI bytes inside captured logs.
    # Capture bytes only; never render untrusted release content to the terminal.
    return subprocess.check_output(['gh', 'api', '--allow-escape-sequences', '-H', 'Accept: application/octet-stream',
                                    f'repos/{REPO}/releases/assets/{identifier}'])


def github_inventory(identifier, source):
    release = gh(f'repos/{REPO}/releases/{identifier}')
    if (release.get('id') != identifier or release.get('tag_name') != TAG
            or release.get('target_commitish') != source or type(release.get('draft')) is not bool):
        raise ValueError('GitHub release source/identity changed')
    rows = []
    for page in range(1, 11):
        batch = gh(f'repos/{REPO}/releases/{identifier}/assets?per_page=100&page={page}')
        if not isinstance(batch, list):
            raise ValueError('Invalid GitHub asset inventory')
        rows.extend(batch)
        if len(batch) < 100:
            break
    else:
        raise ValueError('GitHub asset inventory exceeded bound')
    if len({row['name'] for row in rows}) != len(rows):
        raise ValueError('Duplicate GitHub release assets')
    return release, rows


def github_checked_assets(identifier, source, manifest, *, draft=False, complete=False, cache=None, folder=None):
    release, rows = github_inventory(identifier, source)
    if draft and release['draft'] is not True:
        raise ValueError('Draft became public; refuse mutation')
    if (complete or not release['draft']) and {row['name'] for row in rows} != set(manifest):
        raise ValueError('GitHub release asset inventory differs')
    if folder is not None:
        folder.mkdir()
    if cache is None:
        cache = {}
    for row in rows:
        name = row['name']
        if name not in manifest or row.get('state') != 'uploaded':
            raise ValueError('Unknown or incomplete GitHub asset: ' + name)
        fingerprint = tuple(row.get(k) for k in ('id', 'name', 'size', 'digest', 'state', 'updated_at'))
        if fingerprint not in cache:
            cache[fingerprint] = github_asset_bytes(row['id'])
        data = cache[fingerprint]
        if len(data) != row.get('size') or hashlib.sha256(data).hexdigest() != manifest[name]:
            raise ValueError('Conflicting GitHub release asset: ' + name)
        if folder is not None:
            (folder / name).write_bytes(data)
    return release, rows


def github_upload(identifier, path):
    data = path.read_bytes()
    if not data:
        raise ValueError('Zero-length GitHub upload refused')
    request = urllib.request.Request(
        f'https://uploads.github.com/repos/{REPO}/releases/{identifier}/assets?name=' + urllib.parse.quote(path.name, safe=''),
        data=data, method='POST', headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
        'Content-Type': 'application/octet-stream', 'Accept': 'application/vnd.github+json'})
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=120) as response:
        if response.status != 201:
            raise ValueError('GitHub upload did not return 201')
        row = json.load(response)
    if row.get('name') != path.name or row.get('state') != 'uploaded' or row.get('size') != len(data):
        raise ValueError('GitHub uploaded asset metadata differs')


def github_resume(identifier, source, assets, manifest, guard):
    cache = {}
    for _ in range(len(manifest) + 1):
        guard()
        _, rows = github_checked_assets(identifier, source, manifest, draft=True, cache=cache)
        missing = sorted(set(manifest) - {row['name'] for row in rows})
        if not missing:
            return
        github_upload(identifier, assets / missing[0])
    raise ValueError('GitHub draft upload exceeded bounded progress')


def github_release(base, source):
    receipt = validate_public(base, source)
    def canonical():
        public = registry.downloaded(NAME, VERSION)
        if public is None or registry.sha(public) != receipt['archiveSha256']:
            raise ValueError('Canonical bytes changed before GitHub release')
    canonical()
    assets, manifest = github_assets(base, source, receipt)
    release_tools = load('tensor_github', 'create-rust-github-release.py')
    release_tools.TAG = TAG
    existing = release_tools.existing_release()
    if existing and (existing.get('tag_name') != TAG or existing.get('target_commitish') != source
                     or type(existing.get('draft')) is not bool):
        raise ValueError('Existing release identifies another source')
    ref = subprocess.run(['gh', 'api', f'repos/{REPO}/git/ref/tags/{TAG}'], capture_output=True, text=True)
    if ref.returncode:
        if '404' not in ref.stderr:
            raise ValueError('Could not establish immutable tag absence')
        if existing:
            raise ValueError('Release exists without the exact tag')
        protected(source)
        subprocess.run(['gh', 'api', '--method', 'POST', f'repos/{REPO}/git/refs', '-f', f'ref=refs/tags/{TAG}', '-f', f'sha={source}'], check=True, stdout=subprocess.DEVNULL)
    def guard():
        protected(source)
        if release_tools.tag_source() != source:
            raise ValueError('Existing tag identifies another source')
        canonical()
    guard()
    if not existing:
        notes = base / 'release-notes.md'
        notes.write_text(f"Bounded F32/F16/BF16 Safetensors checkpoint editing and byte-exact rollback.\n\nSource: {source}\nArchive SHA-256: {receipt['archiveSha256']}\n\nQualified source and packed integration tests, cross-platform CLI roundtrips, Rust 1.81, audit with warnings denied, and fresh public registry consumers.\n\nPAN2 GPU inference has not run; no behavioral improvement or automatic promotion is claimed.\n")
        subprocess.run(['gh', 'release', 'create', TAG, '--repo', REPO, '--target', source, '--draft', '--verify-tag',
                        '--title', f'{NAME} {VERSION}', '--notes-file', str(notes)], check=True)
        existing = release_tools.existing_release()
        if not existing:
            raise ValueError('Created draft not found')
    identifier = existing['id']
    initial, _ = github_checked_assets(identifier, source, manifest)
    if initial['draft']:
        github_resume(identifier, source, assets, manifest, guard)
    before, _ = github_checked_assets(identifier, source, manifest, complete=True, folder=base / 'github-before')
    if before['draft'] is not initial['draft']:
        raise ValueError('Release state changed during upload')
    guard()
    if before['draft']:
        github_checked_assets(identifier, source, manifest, draft=True, complete=True)
        subprocess.run(['gh', 'api', '--method', 'PATCH', f'repos/{REPO}/releases/{identifier}', '-F', 'draft=false'], check=True, stdout=subprocess.DEVNULL)
    metadata, _ = github_checked_assets(identifier, source, manifest, complete=True, folder=base / 'github-after')
    if metadata['draft'] is not False or release_tools.tag_source() != source:
        raise ValueError('Release remains draft or tag changed')
    (base / 'github-release.json').write_text(json.dumps({'sourceSha': source, 'tag': TAG, 'url': metadata['html_url'], 'assets': manifest, 'verified': True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('seal', 'validate', 'download', 'publish', 'verify-public', 'github-release'))
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    globals()[{'verify-public': 'verify_public', 'github-release': 'github_release'}.get(args.mode, args.mode)](args.directory.resolve(), args.expected_sha)

if __name__ == '__main__':
    main()
