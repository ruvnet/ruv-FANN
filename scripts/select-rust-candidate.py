#!/usr/bin/env python3
"""Select exact-source completed release-branch CI; never accept a caller's run ID."""
import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = 'ruvnet/ruv-FANN'
ARTIFACTS = ('verified-cargo-release-archives', 'candle-native-oracle')


def gh(endpoint):
    return json.loads(subprocess.check_output(['gh', 'api', endpoint], text=True))


def pages(endpoint, field):
    result = []
    for page in range(1, 101):
        rows = gh(f'{endpoint}{"&" if "?" in endpoint else "?"}per_page=100&page={page}')[field]
        result.extend(rows)
        if len(rows) < 100:
            return result
    raise ValueError('GitHub pagination exceeded the explicit bound')


def validate_run(run, jobs, artifacts, source):
    if (run.get('head_sha') != source or run.get('event') != 'push'
            or not run.get('head_branch', '').startswith('release/')
            or run.get('path') != '.github/workflows/ci.yml'
            or run.get('status') != 'completed' or run.get('conclusion') != 'success'):
        raise ValueError('Candidate must be successful exact-SHA release-branch CI')
    if not jobs or any(job.get('status') != 'completed' or job.get('conclusion') != 'success' for job in jobs):
        raise ValueError('CI contains missing, skipped, pending or unsuccessful jobs')
    names = [row['name'] for row in artifacts if not row.get('expired')]
    for name in ARTIFACTS:
        matches = [row for row in artifacts if row['name'] == name and not row.get('expired')]
        if names.count(name) != 1 or not re.fullmatch(r'sha256:[a-f0-9]{64}', matches[0].get('digest', '')):
            raise ValueError('Missing, duplicate, expired or unsealed candidate artifact: ' + name)
    return {'schema': 'ruv-fann.rust-candidate-ci/1', 'sourceSha': source,
            'runId': run['id'], 'runUrl': run['html_url'],
            'jobs': [{'name': job['name'], 'conclusion': job['conclusion']} for job in jobs],
            'artifacts': [row for row in artifacts if row['name'] in ARTIFACTS]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch('[a-f0-9]{40}', args.expected_sha):
        raise ValueError('Expected a full immutable source SHA')
    if gh(f'repos/{REPO}/git/ref/heads/main')['object']['sha'] != args.expected_sha:
        raise ValueError('Current main differs from reviewed source')
    if gh(f'repos/{REPO}/issues?labels=release-blocker&state=open&per_page=1'):
        raise ValueError('Open release blockers')
    runs = pages(f'repos/{REPO}/actions/workflows/ci.yml/runs?head_sha={args.expected_sha}&event=push', 'workflow_runs')
    eligible = [row for row in runs if row.get('head_branch', '').startswith('release/') and row.get('conclusion') == 'success']
    if not eligible:
        raise ValueError('No exact-SHA successful pre-main release-branch CI artifact')
    run = max(eligible, key=lambda row: row['id'])
    receipt = validate_run(run, pages(f'repos/{REPO}/actions/runs/{run["id"]}/jobs?filter=latest', 'jobs'),
                           pages(f'repos/{REPO}/actions/runs/{run["id"]}/artifacts', 'artifacts'), args.expected_sha)
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(run['id'])


if __name__ == '__main__':
    main()
