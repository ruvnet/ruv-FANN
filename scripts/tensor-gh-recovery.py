#!/usr/bin/env python3
"""Recover one failed tensor GitHub draft; never upload a Cargo package or move refs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = 'ruvnet/ruv-FANN'
SOURCE = 'fc575c8c6cdbd18cca92fb07ddfdcd87c60d34a8'
REF = 'refs/heads/ops/tensor-release-recovery'
WORKFLOW = '.github/workflows/protected-tensor-release.yml'
TAG = 'ruv-fann-wsp-tensor-v0.1.0'
ARCHIVE = 'ruv-fann-wsp-tensor-0.1.0.crate'
CRATE_SHA = 'ab361ddadc64f8a87c132cd4a21333c16cad5c888c01a091334b0d75f685d269'
FAILED_RUN, DRAFT = 43137765102, 442401777
PUBLICATION_ID = 13907489615
PUBLICATION_SHA = '322df02118afce6fde80e49683e713cbfea74905ad5867562d7b83e8308913a0'
CANDIDATE_ID = 13903115912
CANDIDATE_SHA = 'fc02a4c0a3a5f616eede5f7a492dd5927b6b9a5d5f2cf0bd40dee28f98fa3b3a'
FILES = ('scripts/tensor-gh-recovery.py', 'scripts/tests/test_tensor_recovery.py', WORKFLOW)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*argv):
    return subprocess.check_output(list(argv), text=True)


def gh(endpoint):
    return json.loads(run('gh', 'api', endpoint))


def pages(endpoint, field=None):
    result = []
    for page in range(1, 21):
        row = gh(endpoint + ('&' if '?' in endpoint else '?') + f'per_page=100&page={page}')
        rows = row[field] if field else row
        require(isinstance(rows, list), 'Invalid API inventory')
        result.extend(rows)
        if len(rows) < 100:
            return result
    raise ValueError('API inventory exceeded bound')


def checkout_identity(path, expected):
    require(re.fullmatch('[a-f0-9]{40}', expected), 'Full immutable SHA required')
    require(run('git', '-C', str(path), 'rev-parse', 'HEAD').strip() == expected, 'Checkout SHA mismatch')
    require(not run('git', '-C', str(path), 'status', '--porcelain'), 'Checkout must be clean')


def controller_guard(controller):
    checkout_identity(ROOT, controller)
    expected = {'GITHUB_ACTIONS': 'true', 'GITHUB_EVENT_NAME': 'workflow_dispatch',
                'GITHUB_WORKFLOW': 'protected-tensor-release', 'GITHUB_REF': REF,
                'GITHUB_SHA': controller, 'GITHUB_REPOSITORY': REPO,
                'GITHUB_WORKFLOW_REF': f'{REPO}/{WORKFLOW}@{REF}'}
    require(all(os.environ.get(k) == v for k, v in expected.items()), 'Protected recovery controller identity mismatch')
    require(gh(f'repos/{REPO}/git/ref/heads/ops/tensor-release-recovery')['object']['sha'] == controller,
            'Recovery controller ref moved')


def source_module(path):
    path = path.resolve()
    require(path != ROOT and not path.is_relative_to(ROOT), 'Source must be a separate immutable checkout')
    checkout_identity(path, SOURCE)
    spec = importlib.util.spec_from_file_location('original_tensor_release', path / 'scripts/tensor-release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_source():
    require(gh(f'repos/{REPO}/git/ref/heads/main')['object']['sha'] == SOURCE, 'Main source moved')
    obj = gh(f'repos/{REPO}/git/ref/tags/{TAG}')['object']
    for _ in range(5):
        if obj.get('type') == 'commit':
            break
        require(obj.get('type') == 'tag', 'Tag is not a commit')
        obj = gh(f'repos/{REPO}/git/tags/{obj["sha"]}')['object']
    require(obj.get('type') == 'commit' and obj.get('sha') == SOURCE, 'Immutable source tag changed')
    require(not gh(f'repos/{REPO}/issues?state=open&labels=release-blocker&per_page=1'), 'Open release blockers')


def validate_failed_run(row, jobs):
    require((row.get('id'), row.get('head_sha'), row.get('head_branch'), row.get('event'), row.get('path'),
             row.get('status'), row.get('conclusion'), row.get('run_attempt')) ==
            (FAILED_RUN, SOURCE, 'main', 'workflow_dispatch', WORKFLOW, 'completed', 'failure', 1),
            'Original failed publication run differs')
    require(len(jobs) == 1 and jobs[0].get('name') == 'publish' and jobs[0].get('conclusion') == 'failure'
            and jobs[0].get('status') == 'completed', 'Original failed job differs')
    steps = jobs[0]['steps']
    names = [s['name'] for s in steps]
    success = ('Revalidate exact source and download the sealed pre-main candidate',
               'Publish only the qualified immutable Cargo archive',
               'Verify canonical bytes and fresh public Cargo and CLI consumers without registry credentials')
    for name in success:
        require(names.count(name) == 1 and next(s for s in steps if s['name'] == name)['conclusion'] == 'success',
                'Earlier publication gate did not succeed: ' + name)
    failure = [s for s in steps if s.get('conclusion') == 'failure']
    require(len(failure) == 1 and failure[0]['name'] == 'Create exact-source release and verify every downloaded asset',
            'Unexpected publication failure')


def artifact_identity(item, identifier, digest, run_id, source, name):
    require((item.get('id'), item.get('digest'), item.get('expired'), item.get('name')) ==
            (identifier, 'sha256:' + digest, False, name), 'Artifact identity/digest changed')
    require(item.get('workflow_run', {}).get('id') == run_id
            and item['workflow_run'].get('head_sha') == source, 'Artifact run/source changed')


def safe_extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as stream:
        entries = stream.infolist()
        require(len(entries) <= 100 and sum(i.file_size for i in entries) <= 10_000_000, 'Oversized evidence ZIP')
        names = [i.filename for i in entries]
        require(len(names) == len(set(names)), 'Duplicate ZIP member')
        for info in entries:
            path = PurePosixPath(info.filename)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in info.filename
                    and ':' not in info.filename and path.as_posix() == info.filename.rstrip('/'), 'Unsafe ZIP path')
            mode = (info.external_attr >> 16) & 0o170000
            require(mode in (0, stat.S_IFREG, stat.S_IFDIR), 'Nonregular ZIP member')
        stream.extractall(destination)


def download_artifact(identifier, digest, output):
    with output.open('xb') as stream:
        subprocess.run(['gh', 'api', f'repos/{REPO}/actions/artifacts/{identifier}/zip?recovery={time.time_ns()}'], stdout=stream, check=True)
    require(sha(output) == digest, 'Original artifact ZIP checksum mismatch')


def sums(files):
    return ''.join(f'{sha(path)}  {name}\n' for name, path in sorted(files.items())).encode()


def validate_original(pub, candidate, module):
    receipt = module.validate(candidate, SOURCE)
    require(receipt['archiveSha256'] == CRATE_SHA and sha(pub / 'public.crate') == CRATE_SHA, 'Original canonical crate differs')
    assets = pub / 'release-assets'
    expected = {ARCHIVE, 'candidate.json', 'ci.json', 'publication.json', 'public-consumers.json', 'SHA256SUMS'}
    expected.update('candidate-' + row['log'] for row in receipt['commands'])
    public_roles = [('public-tests', 'rust', ['cargo', 'test', '--locked', '--test', 'checkpoints', '--', '--test-threads=1']),
                    ('public-build', None, ['cargo', 'build', '--release', '--locked']),
                    ('public-cli', 'cli', ['python3', 'scripts/e2e.py'])]
    logs = [r[0] + '.log' for r in public_roles] + ['registry-consumer.log', 'registry-install.log', 'registry-cli.log']
    expected.update(logs)
    require({p.name for p in assets.iterdir()} == expected, 'Original release asset inventory differs')
    require((assets / 'SHA256SUMS').read_bytes() == sums({n: assets / n for n in expected - {'SHA256SUMS'}}), 'Original asset checksum manifest changed')
    for name, original in [('candidate.json', candidate / 'candidate.json'), (ARCHIVE, candidate / ARCHIVE),
                           ('ci.json', pub / 'ci.json'), ('publication.json', pub / 'publication.json'),
                           ('public-consumers.json', pub / 'public-consumers.json')]:
        require((assets / name).read_bytes() == original.read_bytes(), 'Original asset/source mismatch: ' + name)
    require((pub / 'candidate/candidate.json').read_bytes() == (candidate / 'candidate.json').read_bytes(), 'Candidate receipt changed')
    for row in receipt['commands']:
        require((assets / ('candidate-' + row['log'])).read_bytes() == (candidate / row['log']).read_bytes(), 'Candidate log changed')
    for log in logs:
        require((assets / log).read_bytes() == (pub / 'fresh' / log).read_bytes(), 'Public log changed')
    publication = json.loads((pub / 'publication.json').read_text())
    proof = json.loads((pub / 'public-consumers.json').read_text())
    require(publication == {'sourceSha': SOURCE, 'archiveSha256': CRATE_SHA, 'name': module.NAME,
                            'version': module.VERSION, 'status': 'published-and-byte-verified'}, 'Original publication receipt differs')
    require(proof.get('sourceSha') == SOURCE and proof.get('archiveSha256') == CRATE_SHA, 'Public proof identity differs')
    require(len(proof['commands']) == len(public_roles), 'Missing public command')
    for row, (role, kind, argv) in zip(proof['commands'], public_roles):
        log = pub / 'fresh' / (role + '.log')
        require(row.get('role') == role and row.get('argv') == argv
                and row.get('cwd') == '${CANDIDATE}/extracted/' + module.NAME + '-' + module.VERSION
                and type(row.get('exitCode')) is int and row['exitCode'] == 0 and row.get('log') == log.name
                and row.get('logSha256') == sha(log) and row.get('evidence') == module.evidence(log.read_text(), kind),
                'Public command/log differs: ' + role)
    registry = proof['registryConsumer']
    require(type(registry.get('exitCode')) is int and registry['exitCode'] == 0 and registry.get('checksum') == CRATE_SHA
            and registry.get('logSha256') == sha(pub / 'fresh/registry-consumer.log')
            and (pub / 'fresh/registry-consumer.log').read_text().splitlines().count('Public tensor API consumer passed') == 1,
            'Registry consumer proof differs')
    cli = proof['installedCli']
    require(type(cli.get('exitCode')) is int and cli['exitCode'] == 0
            and cli.get('installLogSha256') == sha(pub / 'fresh/registry-install.log')
            and cli.get('logSha256') == sha(pub / 'fresh/registry-cli.log')
            and cli.get('evidence') == module.evidence((pub / 'fresh/registry-cli.log').read_text(), 'cli'), 'Installed CLI proof differs')
    top_expected = {'candidate/candidate.json', 'ci.json', 'public.crate', 'publication.json', 'public-consumers.json'}
    top_expected.update('fresh/' + name for name in logs)
    top_expected.update('release-assets/' + name for name in expected)
    require({p.relative_to(pub).as_posix() for p in pub.rglob('*') if p.is_file()} == top_expected, 'Original evidence inventory differs')
    return receipt


def prepare(base, module):
    base.mkdir(parents=True, exist_ok=False)
    validate_failed_run(gh(f'repos/{REPO}/actions/runs/{FAILED_RUN}'), pages(f'repos/{REPO}/actions/runs/{FAILED_RUN}/jobs?filter=latest', 'jobs'))
    artifact_identity(gh(f'repos/{REPO}/actions/artifacts/{PUBLICATION_ID}'), PUBLICATION_ID, PUBLICATION_SHA, FAILED_RUN, SOURCE, 'tensor-publication-' + SOURCE)
    artifact_identity(gh(f'repos/{REPO}/actions/artifacts/{CANDIDATE_ID}'), CANDIDATE_ID, CANDIDATE_SHA, 43124955022, SOURCE, 'tensor-candidate-' + SOURCE)
    for identifier, digest, name in [(PUBLICATION_ID, PUBLICATION_SHA, 'publication'), (CANDIDATE_ID, CANDIDATE_SHA, 'candidate')]:
        download_artifact(identifier, digest, base / (name + '.zip'))
        safe_extract(base / (name + '.zip'), base / name)
    receipt = validate_original(base / 'publication', base / 'candidate', module)
    ci = json.loads((base / 'publication/ci.json').read_text())
    require(ci.get('sourceSha') == SOURCE and ci['run']['id'] == 43124955022 and ci['repositoryCi']['runId'] == 43124954982, 'Original CI source/run differs')
    item = module.validate_ci(gh(f'repos/{REPO}/actions/runs/43124955022'), pages(f'repos/{REPO}/actions/runs/43124955022/jobs?filter=latest', 'jobs'), pages(f'repos/{REPO}/actions/runs/43124955022/artifacts', 'artifacts'), SOURCE)
    require(item['id'] == CANDIDATE_ID and item['digest'] == 'sha256:' + CANDIDATE_SHA and ci['artifact']['id'] == CANDIDATE_ID and ci['artifact']['digest'] == item['digest'], 'Original candidate CI seal differs')
    module.ci_tools.validate_run(gh(f'repos/{REPO}/actions/runs/43124954982'), pages(f'repos/{REPO}/actions/runs/43124954982/jobs?filter=latest', 'jobs'), pages(f'repos/{REPO}/actions/runs/43124954982/artifacts', 'artifacts'), SOURCE)
    canonical(module)
    return receipt


def canonical(module):
    public = module.registry.downloaded(module.NAME, module.VERSION)
    require(public is not None and hashlib.sha256(public).hexdigest() == CRATE_SHA, 'Live canonical archive differs')


def controller_files():
    return {name: sha(ROOT / name) for name in FILES}


def test_evidence(output):
    counts = re.findall(r'^Ran (\d+) tests? in ', output, re.M)
    require(len(counts) == 1 and int(counts[0]) >= 8 and re.search(r'^OK\s*$', output, re.M), 'Nonzero unskipped controller guards required')
    return int(counts[0])


def qualify(base, controller, module):
    live_source()
    base.mkdir(parents=True, exist_ok=False)
    argv = ['python3', '-m', 'unittest', 'discover', '-s', 'scripts/tests', '-p', 'test_tensor_recovery.py', '-v']
    result = subprocess.run(argv, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True)
    log = base / 'qualification.log'
    log.write_text(result.stdout + result.stderr)
    require(result.returncode == 0, 'Recovery guard command failed: ' + log.read_text()[-8000:])
    count = test_evidence(log.read_text())
    prepare(base / 'evidence', module)
    receipt = {'schema': 'ruv-fann.tensor-recovery-qualification/1', 'controllerSha': controller, 'sourceSha': SOURCE,
               'files': controller_files(), 'command': argv, 'exitCode': result.returncode, 'tests': count,
               'logSha256': sha(log), 'publicationZipSha256': PUBLICATION_SHA, 'candidateZipSha256': CANDIDATE_SHA,
               'archiveSha256': CRATE_SHA}
    (base / 'qualification.json').write_text(json.dumps(receipt, indent=2) + '\n')


def qualification(base, controller, run_id):
    row = gh(f'repos/{REPO}/actions/runs/{run_id}')
    require((row.get('head_sha'), row.get('head_branch'), row.get('event'), row.get('path'), row.get('status'), row.get('conclusion')) ==
            (controller, REF.removeprefix('refs/heads/'), 'workflow_dispatch', WORKFLOW, 'completed', 'success'), 'Controller qualifier run differs')
    jobs = pages(f'repos/{REPO}/actions/runs/{run_id}/jobs?filter=latest', 'jobs')
    require(len(jobs) == 1 and jobs[0].get('name') == 'controller' and jobs[0].get('status') == 'completed' and jobs[0].get('conclusion') == 'success', 'Controller qualifier job did not succeed')
    artifacts = pages(f'repos/{REPO}/actions/runs/{run_id}/artifacts', 'artifacts')
    matches = [a for a in artifacts if a.get('name') == 'tensor-recovery-qualification-' + controller and a.get('expired') is False]
    require(len(matches) == 1 and re.fullmatch('sha256:[a-f0-9]{64}', matches[0].get('digest', '')), 'Controller qualifier seal missing')
    item = matches[0]
    download_artifact(item['id'], item['digest'].removeprefix('sha256:'), base / 'qualification.zip')
    safe_extract(base / 'qualification.zip', base / 'qualification')
    folder = base / 'qualification'
    require({p.name for p in folder.iterdir()} == {'qualification.json', 'qualification.log'}, 'Unexpected qualifier files')
    proof = json.loads((folder / 'qualification.json').read_text())
    expected = {'schema': 'ruv-fann.tensor-recovery-qualification/1', 'controllerSha': controller, 'sourceSha': SOURCE,
                'files': controller_files(), 'command': ['python3', '-m', 'unittest', 'discover', '-s', 'scripts/tests', '-p', 'test_tensor_recovery.py', '-v'],
                'exitCode': 0, 'tests': test_evidence((folder / 'qualification.log').read_text()), 'logSha256': sha(folder / 'qualification.log'),
                'publicationZipSha256': PUBLICATION_SHA, 'candidateZipSha256': CANDIDATE_SHA, 'archiveSha256': CRATE_SHA}
    require(type(proof.get('exitCode')) is int and proof == expected, 'Qualifier receipt/log/source differs')
    return {'runId': run_id, 'artifactId': item['id'], 'zipSha256': sha(base / 'qualification.zip')}


def make_assets(base, controller, admission):
    original = base / 'evidence/publication/release-assets'
    assets = base / 'assets'; assets.mkdir()
    empty = []
    for path in sorted(original.iterdir()):
        if path.name == 'SHA256SUMS':
            shutil.copyfile(path, assets / 'original-SHA256SUMS')
        elif path.stat().st_size:
            shutil.copyfile(path, assets / path.name)
        else:
            require(path.name.endswith('.log'), 'Unexpected empty original asset')
            empty.append({'originalAsset': path.name, 'size': 0, 'sha256': sha(path), 'container': 'original-candidate.zip', 'member': 'fmt.log'})
    require(empty == [{'originalAsset': 'candidate-fmt.log', 'size': 0, 'sha256': hashlib.sha256(b'').hexdigest(), 'container': 'original-candidate.zip', 'member': 'fmt.log'}], 'Original empty log differs')
    with zipfile.ZipFile(base / 'evidence/candidate.zip') as stream:
        require(stream.read('fmt.log') == b'', 'Original candidate ZIP does not preserve empty fmt output')
    (assets / 'empty-log-encoding.json').write_text(json.dumps({'schema': 'ruv-fann.tensor-empty-log-encoding/1', 'sourceSha': SOURCE, 'publicationZipSha256': PUBLICATION_SHA, 'entries': empty}, indent=2) + '\n')
    for old, name in [('evidence/publication.zip', 'original-publication.zip'), ('evidence/candidate.zip', 'original-candidate.zip'), ('qualification.zip', 'controller-qualification.zip')]:
        shutil.copyfile(base / old, assets / name)
    binding = {'schema': 'ruv-fann.tensor-recovery-binding/1', 'controllerSha': controller, 'controllerFiles': controller_files(),
               'sourceSha': SOURCE, 'originalFailedRun': FAILED_RUN, 'publicationArtifactId': PUBLICATION_ID,
               'publicationZipSha256': PUBLICATION_SHA, 'candidateArtifactId': CANDIDATE_ID, 'candidateZipSha256': CANDIDATE_SHA,
               'archiveSha256': CRATE_SHA, 'tag': TAG, 'releaseId': DRAFT, 'qualification': admission}
    (assets / 'recovery-binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    (assets / 'SHA256SUMS').write_bytes(sums({p.name: p for p in assets.iterdir()}))
    return {p.name: sha(p) for p in assets.iterdir()}


def release_inventory():
    release = gh(f'repos/{REPO}/releases/{DRAFT}')
    require(release.get('id') == DRAFT and release.get('tag_name') == TAG and release.get('target_commitish') == SOURCE
            and type(release.get('draft')) is bool, 'Known release identity changed')
    rows = pages(f'repos/{REPO}/releases/{DRAFT}/assets')
    require(len({r['name'] for r in rows}) == len(rows), 'Duplicate release assets')
    return release, rows


def asset_bytes(identifier):
    return subprocess.check_output(['gh', 'api', '-H', 'Accept: application/octet-stream', f'repos/{REPO}/releases/assets/{identifier}'])


def classify_assets(rows, assets, original_sums, is_draft, fetch=asset_bytes):
    """Read every existing asset before allowing any change; only old sums may differ."""
    replacement = None
    for row in rows:
        name = row['name']
        require(name in assets and row.get('state') == 'uploaded', 'Unknown or incomplete release asset: ' + name)
        data = fetch(row['id'])
        require(len(data) == row.get('size'), 'Existing asset size differs')
        if hashlib.sha256(data).hexdigest() == sha(assets[name]):
            continue
        require(is_draft and name == 'SHA256SUMS' and data == original_sums, 'Existing asset bytes conflict: ' + name)
        replacement = row['id']
    if not is_draft:
        require({r['name'] for r in rows} == set(assets), 'Published release inventory differs')
    return replacement


def checked_draft(assets, original_sums, cache, complete=False):
    """Re-read mutable metadata; reuse bytes only for unchanged immutable asset IDs."""
    release, rows = release_inventory()
    require(release['draft'] is True, 'Draft became public; refuse further mutation')
    fingerprints = {row['id']: tuple(row.get(k) for k in
                    ('id', 'name', 'size', 'digest', 'state', 'updated_at')) for row in rows}

    def fetch(identifier):
        fingerprint = fingerprints[identifier]
        if fingerprint not in cache:
            cache[fingerprint] = asset_bytes(identifier)
        return cache[fingerprint]

    replacement = classify_assets(rows, assets, original_sums, True, fetch)
    if complete:
        require(replacement is None and {row['name'] for row in rows} == set(assets),
                'Final draft manifest changed before publication')
    return rows, replacement


def upload_asset(path):
    """POST raw bytes to the pinned release ID, never resolve another release by tag."""
    data = path.read_bytes()
    require(data, 'Cannot upload a zero-length asset')
    request = urllib.request.Request(
        f'https://uploads.github.com/repos/{REPO}/releases/{DRAFT}/assets?name=' + urllib.parse.quote(path.name, safe=''),
        data=data, method='POST', headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
        'Content-Type': 'application/octet-stream', 'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'})
    # This endpoint returns 201 directly. Never forward authorization on a redirect.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=120) as response:
        require(response.status == 201, 'Asset upload did not return 201')
        row = json.load(response)
    require(row.get('name') == path.name and row.get('state') == 'uploaded'
            and row.get('size') == len(data), 'Uploaded asset metadata differs')


def repair_draft(assets, original_sums, guard):
    cache = {}
    # One deletion and at most one upload per final asset; unexpected lack of progress stops.
    for _ in range(len(assets) + 2):
        guard()
        rows, replacement = checked_draft(assets, original_sums, cache)
        if replacement is not None:
            subprocess.run(['gh', 'api', '--method', 'DELETE', f'repos/{REPO}/releases/assets/{replacement}'], check=True)
            continue
        missing = sorted(set(assets) - {row['name'] for row in rows})
        if not missing:
            return
        upload_asset(assets[missing[0]])
    raise ValueError('Draft repair exceeded bounded progress')


def publish_draft(assets, original_sums, guard):
    guard()
    # No cached bytes here: check the complete manifest again immediately before PATCH.
    checked_draft(assets, original_sums, {}, complete=True)
    subprocess.run(['gh', 'api', '--method', 'PATCH', f'repos/{REPO}/releases/{DRAFT}', '-F', 'draft=false'],
                   check=True, stdout=subprocess.DEVNULL)


def verify_assets(base, manifest, stage):
    release, rows = release_inventory()
    require({r['name'] for r in rows} == set(manifest), 'Final release asset inventory differs')
    folder = base / stage; folder.mkdir()
    for row in rows:
        require(row.get('state') == 'uploaded', 'Release asset incomplete')
        data = asset_bytes(row['id'])
        require(hashlib.sha256(data).hexdigest() == manifest[row['name']] and len(data) == row['size'], 'Downloaded release bytes differ')
        (folder / row['name']).write_bytes(data)
    live_source()
    return release


def recover(base, controller, module, qualifier_run):
    live_source()
    base.mkdir(parents=True, exist_ok=False)
    admission = qualification(base, controller, qualifier_run)
    prepare(base / 'evidence', module)
    manifest = make_assets(base, controller, admission)
    assets = {name: base / 'assets' / name for name in manifest}
    existing, rows = release_inventory()
    original_sums = (base / 'evidence/publication/release-assets/SHA256SUMS').read_bytes()
    classify_assets(rows, assets, original_sums, existing['draft'])
    # All admission and all existing-byte checks precede the first mutation.
    def guard():
        controller_guard(controller); live_source(); canonical(module)
    guard()
    if existing['draft']:
        repair_draft(assets, original_sums, guard)
    before = verify_assets(base, manifest, 'before-publication')
    require(before['draft'] is existing['draft'], 'Release draft state changed during recovery')
    if existing['draft']:
        publish_draft(assets, original_sums, guard)
    after = verify_assets(base, manifest, 'after-publication')
    require(after['draft'] is False, 'Release is not public')
    proof = {'schema': 'ruv-fann.tensor-github-recovery/1', 'controllerSha': controller, 'sourceSha': SOURCE,
             'originalFailedRun': FAILED_RUN, 'publicationArtifactId': PUBLICATION_ID, 'publicationZipSha256': PUBLICATION_SHA,
             'candidateArtifactId': CANDIDATE_ID, 'candidateZipSha256': CANDIDATE_SHA, 'archiveSha256': CRATE_SHA,
             'qualification': admission, 'recoveryRunId': int(os.environ['GITHUB_RUN_ID']), 'tag': TAG, 'releaseId': DRAFT,
             'url': after['html_url'], 'assets': manifest, 'beforeDownloadHashes': manifest, 'afterDownloadHashes': manifest}
    (base / 'recovery.json').write_text(json.dumps(proof, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('qualify', 'recover'))
    parser.add_argument('--expected-controller-sha', required=True)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--qualification-run', type=int)
    args = parser.parse_args()
    controller_guard(args.expected_controller_sha)
    module = source_module(args.source_directory)
    if args.mode == 'qualify':
        qualify(args.directory.resolve(), args.expected_controller_sha, module)
    else:
        require(args.qualification_run is not None, 'Exact successful controller qualifier run required')
        recover(args.directory.resolve(), args.expected_controller_sha, module, args.qualification_run)


if __name__ == '__main__':
    main()
