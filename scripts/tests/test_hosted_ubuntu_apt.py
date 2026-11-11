import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('hosted_apt', Path(__file__).resolve().parents[1] / 'hosted-ubuntu-apt.py')
apt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apt)
HOSTED = {'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'github-hosted', 'RUNNER_OS': 'Linux'}


class HostedUbuntuAptTests(unittest.TestCase):
    def test_legacy_sources_preserve_options_security_and_other_origins(self):
        source = ('deb [arch=amd64 signed-by=/usr/share/keyrings/ubuntu.gpg] http://azure.archive.ubuntu.com/ubuntu noble main\n'
                  'deb https://security.ubuntu.com/ubuntu noble-security main\n'
                  'deb https://azure.archive.ubuntu.com.example/ubuntu noble main\n')
        expected = source.replace('http://azure.archive.ubuntu.com/ubuntu ', 'https://archive.ubuntu.com/ubuntu ')
        self.assertEqual(apt.canonical_mirror(source), expected)
        self.assertEqual(apt.canonical_mirror(expected), expected)

    def test_deb822_preserves_signed_by_suites_and_other_uris(self):
        source = ('Types: deb deb-src\nURIs: http://azure.archive.ubuntu.com/ubuntu/ https://security.ubuntu.com/ubuntu\n'
                  'Suites: noble noble-updates noble-backports\nComponents: main universe\n'
                  'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n\n'
                  'Types: deb\nURIs: https://packages.microsoft.com/repos/azure-cli\nSuites: noble\n')
        self.assertEqual(apt.canonical_mirror(source), source.replace('http://azure.archive.ubuntu.com/ubuntu/', 'https://archive.ubuntu.com/ubuntu/'))

    def test_local_self_hosted_and_other_operating_systems_never_execute_commands(self):
        with patch.object(apt.subprocess, 'run') as run:
            for environment in ({}, dict(HOSTED, GITHUB_ACTIONS='false'), dict(HOSTED, RUNNER_ENVIRONMENT='self-hosted'), dict(HOSTED, RUNNER_OS='Windows')):
                with self.subTest(environment=environment), self.assertRaises(ValueError):
                    apt.install(['mesa-vulkan-drivers'], environment)
            run.assert_not_called()

    def test_only_reviewed_package_names_are_accepted_without_execution(self):
        with patch.object(apt.subprocess, 'run') as run:
            for packages in ([], ['--allow-unauthenticated'], ['libopencv-dev; command']):
                with self.assertRaises(ValueError):
                    apt.install(packages, HOSTED)
            run.assert_not_called()

    def test_mocked_install_rewrites_only_sources_and_keeps_real_package_requirements(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(apt.subprocess, 'run') as run:
            root = Path(directory)
            (root / 'sources.list.d').mkdir()
            source = root / 'sources.list.d/ubuntu.sources'
            source.write_text('URIs: http://azure.archive.ubuntu.com/ubuntu\nSigned-By: /key.gpg\n')
            apt.install(['libopencv-dev', 'libclang-dev'], HOSTED, root)
            self.assertEqual(run.call_args_list[0].args[0], ['sudo', 'tee', str(source)])
            self.assertEqual(run.call_args_list[0].kwargs['input'], 'URIs: https://archive.ubuntu.com/ubuntu\nSigned-By: /key.gpg\n')
            self.assertEqual(run.call_args_list[-1].args[0], ['sudo', 'apt-get', '-o', 'Acquire::Retries=3', 'install', '--yes', 'libopencv-dev', 'libclang-dev'])
            self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))
            self.assertIn('APT::Update::Error-Mode=any', run.call_args_list[-2].args[0])

    def test_failed_source_rewrite_prevents_apt_install(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(apt.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['sudo', 'tee'])) as run:
            root = Path(directory)
            (root / 'sources.list').write_text('deb http://azure.archive.ubuntu.com/ubuntu noble main\n')
            with self.assertRaises(subprocess.CalledProcessError):
                apt.install(['mesa-vulkan-drivers'], HOSTED, root)
            self.assertEqual(run.call_count, 1)
