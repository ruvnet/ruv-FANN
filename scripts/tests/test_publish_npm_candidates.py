import importlib.util
from pathlib import Path
import unittest
import urllib.error
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('publish_npm', Path(__file__).resolve().parents[1] / 'publish-npm-candidates.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def not_found(url):
    return urllib.error.HTTPError(url, 404, 'Not Found', {}, None)


class Response:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


class VisibilityTests(unittest.TestCase):
    def test_waits_past_two_minutes_with_backoff(self):
        clock = Clock()
        def urlopen(url, timeout):
            if clock.now < 300:
                raise not_found(url)
            return Response(b'public bytes')
        with patch.object(module.urllib.request, 'urlopen', side_effect=urlopen):
            data = module.wait_public('https://registry.example/pkg.tgz', sleep=clock.sleep, clock=clock.time)
        self.assertEqual(data, b'public bytes')
        self.assertEqual(clock.sleeps[:5], [5, 10, 20, 30, 60])
        self.assertTrue(all(delay <= 60 for delay in clock.sleeps))

    def test_gives_up_at_the_deadline(self):
        clock = Clock()
        with patch.object(module.urllib.request, 'urlopen', side_effect=lambda url, timeout: (_ for _ in ()).throw(not_found(url))):
            data = module.wait_public('https://registry.example/pkg.tgz', sleep=clock.sleep, clock=clock.time)
        self.assertIsNone(data)
        self.assertEqual(clock.now, module.VISIBILITY_TIMEOUT)

    def test_other_http_errors_are_not_retried(self):
        clock = Clock()
        error = urllib.error.HTTPError('u', 500, 'Server Error', {}, None)
        with patch.object(module.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaises(urllib.error.HTTPError):
                module.wait_public('https://registry.example/pkg.tgz', sleep=clock.sleep, clock=clock.time)
        self.assertEqual(clock.sleeps, [])


if __name__ == '__main__':
    unittest.main()
