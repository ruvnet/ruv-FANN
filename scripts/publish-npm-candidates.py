#!/usr/bin/env python3
"""Seal tested npm archives, publish only in Actions, and exercise public installs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
# npm can take several minutes before a new tarball is served publicly.
VISIBILITY_TIMEOUT = 600
VISIBILITY_DELAYS = (5, 10, 20, 30, 60)


def wait_public(url, timeout=VISIBILITY_TIMEOUT, sleep=time.sleep, clock=time.monotonic):
    """Return the public tarball bytes, or None if it stays 404 past the deadline."""
    deadline = clock() + timeout
    attempt = 0
    while True:
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        remaining = deadline - clock()
        if remaining <= 0:
            return None
        sleep(min(VISIBILITY_DELAYS[min(attempt, len(VISIBILITY_DELAYS) - 1)], remaining))
        attempt += 1


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, cwd, env=None):
    subprocess.run(argv, cwd=cwd, env=env, check=True, timeout=1800)


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def consumer(directory, specs, packages, compiler=None):
    directory.mkdir()
    (directory / 'package.json').write_text('{"private":true,"type":"module"}')
    # Consumer lifecycle code never inherits registry publishing credentials.
    env = {k: v for k, v in os.environ.items() if k not in ('NODE_AUTH_TOKEN', 'NPM_TOKEN', 'CARGO_REGISTRY_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN')}
    (directory / '.npmrc').write_text('registry=https://registry.npmjs.org\n')
    env['NPM_CONFIG_USERCONFIG'] = str(directory / '.npmrc')
    cuda = [r['version'] for r in packages if r['name'] == 'cuda-wasm']
    if len(cuda) != 1: raise ValueError('Expected one cuda-wasm candidate')
    # The consumer checks the exact version that this release candidate carries.
    env['CUDA_WASM_VERSION'] = cuda[0]
    run(['npm', 'install', '--registry=https://registry.npmjs.org', '--no-audit', '--no-fund', *specs], directory, env)
    script = r'''
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {execFileSync} from 'node:child_process';
import {RuvSwarm} from 'ruv-swarm';
const require=createRequire(import.meta.url);
const runtime=await RuvSwarm.initialize({useSIMD:false});
const json=JSON.stringify({schema:'ruv-fann.dense-layers/1',layers:[{inputs:1,outputs:1,weights:[0],bias:[0],activation:'Linear',steepness:1}]});
for (const simd of [false,true]) {
 const r=await RuvSwarm.initialize({useSIMD:simd});
 const m=r.createTrainableNetwork(json,2);
 const x=Float64Array.of(1,2),y=Float64Array.of(3,5);
 const first=m.mse_backward(x,y,2);
 for(let i=0;i<500;i++){m.sgd_step(0.05);m.mse_backward(x,y,2);}
 assert(m.mse_backward(x,y,2)<first/10000);m.free();
}
const cli=new URL('./node_modules/ruv-swarm/bin/ruv-swarm-secure.js',import.meta.url);
const call=args=>JSON.parse(execFileSync(process.execPath,[cli.pathname,...args],{encoding:'utf8',env:{...process.env,RUV_SWARM_DB_PATH:new URL('./state.sqlite',import.meta.url).pathname}}));
call(['init','mesh','2']);call(['spawn','coder','Registry consumer']);assert.equal(call(['status']).total_agents,1);
const cjs=require('cuda-wasm');const esm=await import('cuda-wasm');assert.equal(cjs.getVersion(),process.env.CUDA_WASM_VERSION);assert.equal(esm.getVersion(),process.env.CUDA_WASM_VERSION);
if(process.env.CUDA2RUST_BINARY){
 const source='__global__ void copy(float* a, float* b) { int i = threadIdx.x; b[i] = a[i]; }';
 assert.match((await cjs.transpileCuda(source)).code,/fn copy/);
 assert.match((await esm.transpileCuda(source,{target:'wgsl'})).code,/@compute/);
}
console.log('Public package training, SIMD, persistent CLI and CUDA entrypoints passed');
'''
    (directory / 'verify.mjs').write_text(script)
    if compiler:
        env['CUDA2RUST_BINARY'] = str(compiler)
    run(['node', 'verify.mjs'], directory, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--publish', action='store_true')
    mode.add_argument('--verify-public', action='store_true')
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    base = args.directory.resolve()
    seal = base / 'npm-candidate.json'
    if args.prepare:
        sqlite = json.loads((base / 'npm-swarm/receipt.json').read_text())
        training = json.loads((base / 'npm-training/receipt.json').read_text())
        if sqlite['status'] != 'passed' or training['status'] != 'passed':
            raise ValueError('Actual archive consumer checks did not pass')
        candidates = []
        for folder, expected in [('npm-swarm','ruv-swarm'),('npm-cuda','cuda-wasm')]:
            archives = list((base / folder).glob('*.tgz'))
            if len(archives) != 1: raise ValueError('Expected one npm archive')
            path = archives[0]
            with tarfile.open(path) as tar:
                package = json.load(tar.extractfile('package/package.json'))
            if package['name'] != expected or package.get('private'): raise ValueError('Unexpected npm package')
            candidates.append({'name': expected, 'version': package['version'], 'archive': str(path), 'sha256': sha(path)})
        if candidates[0]['sha256'] != sqlite['tarballSha256']:
            raise ValueError('SQLite-tested archive differs')
        training_archive = list((base / 'npm-training').glob('*.tgz'))
        if len(training_archive) != 1 or sha(training_archive[0]) != candidates[0]['sha256']:
            raise ValueError('Training-tested archive differs')
        consumer(base / 'npm-local-consumer', [r['archive'] for r in candidates], candidates)
        save(seal, {'sourceSha': head(), 'packages': candidates})
        return
    record = json.loads(seal.read_text())
    if record['sourceSha'] != head(): raise ValueError('Candidate source changed')
    for row in record['packages']:
        if sha(Path(row['archive'])) != row['sha256']: raise ValueError('Candidate archive changed')
    if args.publish:
        if os.environ.get('GITHUB_ACTIONS') != 'true' or os.environ.get('GITHUB_WORKFLOW') != 'protected-release' or os.environ.get('GITHUB_REF') != 'refs/heads/main' or os.environ.get('GITHUB_SHA') != head():
            raise ValueError('npm uploads require protected main workflow')
        if not os.environ.get('NODE_AUTH_TOKEN'): raise ValueError('Missing npm publishing credential')
        published = []
        report = base / 'npm-publication.json'
        with report.open('x') as stream:
            for row in record['packages']:
                url = f"https://registry.npmjs.org/{row['name']}/-/{row['name']}-{row['version']}.tgz"
                try:
                    public = urllib.request.urlopen(url, timeout=60).read()
                except urllib.error.HTTPError as error:
                    if error.code != 404: raise
                    run(['npm','publish',row['archive'],'--ignore-scripts','--access','public','--provenance','--registry=https://registry.npmjs.org'],ROOT)
                    public = wait_public(url)
                    if public is None:
                        raise RuntimeError(f'Published, but {url} was not publicly downloadable within {VISIBILITY_TIMEOUT}s; rerun the failed job to verify the bytes')
                if hashlib.sha256(public).hexdigest() != row['sha256']:
                    raise ValueError('Published npm bytes do not match candidate: '+row['name'])
                published.append({**row, 'status':'published-and-byte-verified'})
                stream.seek(0);json.dump({'sourceSha':head(),'packages':published},stream,indent=2);stream.truncate();stream.flush()
        return
    # Install the Cargo compiler and dense example from public crates.io too.
    run(['cargo','install','cuda-rust-wasm','--version','=0.2.1','--locked','--bin','cuda2rust','--root',str(base/'public-compiler')],base)
    consumer(base/'npm-public-consumer',[r['name']+'@'+r['version'] for r in record['packages']],record['packages'],base/'public-compiler/bin/cuda2rust')
    rust = base/'rust-public-consumer';(rust/'src').mkdir(parents=True)
    (rust/'Cargo.toml').write_text('[package]\nname="release-consumer"\nversion="0.0.0"\nedition="2021"\n[dependencies]\nruv-fann={version="=0.2.3",default-features=false,features=["std","serde"]}\nruv-fann-candle="=0.1.1"\n')
    (rust/'src/main.rs').write_text((ROOT/'examples/dense_autograd.rs').read_text())
    run(['cargo','run','--release'],rust)
    save(base/'public-consumers.json',{'sourceSha':head(),'status':'passed','npmNormalInstall':True,'coreAndSimdTraining':True,'persistentCli':True,'cudaPublicCompiler':True,'cargoPublicDenseTraining':True})


if __name__=='__main__':main()
