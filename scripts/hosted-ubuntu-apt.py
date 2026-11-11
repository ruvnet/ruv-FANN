#!/usr/bin/env python3
"""Install CI's native prerequisites using Ubuntu's canonical archive mirror."""
import argparse
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit, urlunsplit

PACKAGES = ('mesa-vulkan-drivers', 'libopencv-dev', 'libclang-dev')


def require_hosted_linux(environment):
    expected = {'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux'}
    if any(environment.get(key) != value for key, value in expected.items()):
        raise ValueError('Apt configuration is restricted to ephemeral GitHub-hosted Linux runners')


def canonical_mirror(text):
    """Change only exact Azure Ubuntu archive URLs; retain source syntax/options."""
    def replace(match):
        original = match.group()
        url = urlsplit(original)
        if url.hostname != 'azure.archive.ubuntu.com':
            return original
        if url.username or url.password or url.port or url.path not in ('/ubuntu', '/ubuntu/'):
            raise ValueError('Unexpected Azure Ubuntu archive URL')
        return urlunsplit(('https', 'archive.ubuntu.com', url.path, url.query, url.fragment))
    return re.sub(r'https?://[^\s\]\"\'<>]+', replace, text)


def install(packages, environment=os.environ, apt_root=Path('/etc/apt')):
    require_hosted_linux(environment)
    if not packages or any(package not in PACKAGES for package in packages):
        raise ValueError('Only the reviewed CI native prerequisite packages are allowed')
    files = [apt_root / 'sources.list']
    files += sorted((apt_root / 'sources.list.d').glob('*.list'))
    files += sorted((apt_root / 'sources.list.d').glob('*.sources'))
    for path in files:
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(apt_root.resolve()):
            raise ValueError('Apt source path escapes its configuration directory')
        original = path.read_text()
        rewritten = canonical_mirror(original)
        if rewritten != original:
            subprocess.run(['sudo', 'tee', str(path)], input=rewritten, text=True,
                           stdout=subprocess.DEVNULL, check=True)
            print('Configured canonical Ubuntu archive in ' + str(path), flush=True)
    subprocess.run(['sudo', 'apt-get', '-o', 'Acquire::Retries=3',
                    '-o', 'APT::Update::Error-Mode=any', 'update'], check=True)
    subprocess.run(['sudo', 'apt-get', '-o', 'Acquire::Retries=3',
                    'install', '--yes', *packages], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', nargs='+', choices=PACKAGES, required=True)
    args = parser.parse_args()
    install(args.packages)


if __name__ == '__main__':
    main()
