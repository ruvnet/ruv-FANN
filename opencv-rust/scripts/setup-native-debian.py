#!/usr/bin/env python3
"""Extract authenticated Debian 13 development packages into a user prefix.

This is a build helper for Debian 13 hosts with the usual compiler/runtime
libraries, not a replacement for apt's dependency solver. It changes no system
files. Prefer libopencv-dev/libclang-dev installed by your system administrator.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import lzma

MIRROR = "https://deb.debian.org/debian/"
MODULES = ["core", "imgproc", "imgcodecs", "videoio", "highgui", "features2d",
           "flann", "objdetect", "calib3d", "ml"]


def download(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--software-vulkan", action="store_true")
    args = parser.parse_args()
    release = platform.freedesktop_os_release()
    if (release.get("ID"), release.get("VERSION_ID")) != ("debian", "13"):
        parser.error("This helper supports Debian 13; use your distribution's package manager")
    for tool in ("dpkg-deb", "dpkg"):
        if not shutil.which(tool):
            parser.error(f"Required program missing: {tool}")
    architecture = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
    prefix = args.prefix.resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    cache = prefix / ".debian-package-cache"
    cache.mkdir(exist_ok=True)
    index_name = f"main/binary-{architecture}/Packages.xz"
    release_data = download(MIRROR + "dists/trixie/InRelease")
    (cache / "InRelease").write_bytes(release_data)
    keyring = "/usr/share/keyrings/debian-archive-keyring.gpg"
    if shutil.which("gpgv"):
        verifier = ["gpgv", "--keyring", keyring]
        subprocess.run(verifier + [str(cache / "InRelease")], check=True)
    elif shutil.which("gpg"):
        with tempfile.TemporaryDirectory() as gnupg_directory:
            subprocess.run(["gpg", "--batch", "--no-options", "--homedir", gnupg_directory,
                            "--no-default-keyring", "--keyring", keyring,
                            "--no-auto-key-retrieve", "--verify", str(cache / "InRelease")], check=True)
    else:
        raise RuntimeError("gpgv or gpg is required to authenticate the Debian index")
    checksum_section = release_data.decode().split("SHA256:\n", 1)[1].split("\nSHA512:", 1)[0]
    checksum, length, _ = next(line.split() for line in checksum_section.splitlines()
                               if line.strip().endswith(" " + index_name))
    index_data = download(MIRROR + "dists/trixie/" + index_name)
    if len(index_data) != int(length) or hashlib.sha256(index_data).hexdigest() != checksum:
        raise RuntimeError("Signed package index checksum/length mismatch")
    packages = {}
    for block in lzma.decompress(index_data).decode().split("\n\n"):
        item = dict(line.split(": ", 1) for line in block.splitlines()
                    if line and not line[0].isspace() and ": " in line)
        if "Package" in item:
            packages[item["Package"]] = item
    names = ["libopencv-dev", "libclang-common-19-dev"]
    for module in MODULES:
        names.extend([f"libopencv-{module}-dev", f"libopencv-{module}410"])
    if args.software_vulkan:
        names.extend(["mesa-vulkan-drivers", "vulkan-tools"])

    def prepare(name):
        item = packages[name]
        destination = cache / (name + ".deb")
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == item["SHA256"]:
            data = None
        else:
            data = download(MIRROR + item["Filename"])
            if hashlib.sha256(data).hexdigest() != item["SHA256"]:
                raise RuntimeError("Package checksum mismatch: " + name)
            destination.write_bytes(data)
        return item, destination

    evidence = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        # Serialize extraction: archives may contain common directories.
        for item, archive in pool.map(prepare, names):
            subprocess.run(["dpkg-deb", "-x", str(archive), str(prefix)], check=True)
            evidence.append({key: item[key] for key in ("Package", "Version", "SHA256", "Filename")})
            print(item["Package"], item["Version"], flush=True)
    (prefix / "native-package-provenance.json").write_text(json.dumps({
        "mirror": MIRROR, "suite": "trixie", "architecture": architecture,
        "signed_index_sha256": checksum, "packages": evidence,
    }, indent=2) + "\n")
    libraries = prefix / "usr/lib" / subprocess.check_output(["gcc", "-dumpmachine"], text=True).strip()
    missing = set()
    for module in MODULES:
        result = subprocess.run(["ldd", str(libraries / f"libopencv_{module}.so")],
                                text=True, capture_output=True)
        missing.update(line.strip() for line in result.stdout.splitlines() if "not found" in line)
    if missing:
        raise RuntimeError("Additional system runtime dependencies required: " + "; ".join(sorted(missing)))
    print(f"Ready. Source scripts/native-env.sh with prefix {prefix}")


if __name__ == "__main__":
    main()
