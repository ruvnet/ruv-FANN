#!/usr/bin/env python3
"""Extract acceleration window features from the byte-pinned UCI MHEALTH archive."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


ARCHIVE_BYTES = 75_567_983
ARCHIVE_SHA256 = "16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0"
SCHEMA = "ruv-fann.mhealth-features/1"
WINDOW = 100
LOG_BYTES = (
    30_019_680, 24_446_574, 22_788_438, 21_833_916, 22_285_024,
    18_243_582, 19_460_454, 24_087_426, 25_148_835, 18_500_501,
)
POSITIONS = {"chest": 0, "wrist": 14, "ankle": 5}


def mean_and_sd(values):
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def features(samples):
    axes = list(zip(*samples, strict=True))
    means, sds = zip(*(mean_and_sd(axis) for axis in axes), strict=True)
    magnitude = [math.hypot(*sample) for sample in samples]
    rms_differences = [
        math.sqrt(math.fsum((right - left) ** 2 for left, right in zip(axis, axis[1:])) / (len(axis) - 1))
        for axis in axes
    ]
    return [*means, *sds, *mean_and_sd(magnitude), *rms_differences]


def subject_windows(lines, subject):
    previous = 0
    ordinal = 0
    samples = {position: [] for position in POSITIONS}
    for index, line in enumerate(lines):
        columns = line.split()
        if len(columns) != 24:
            raise ValueError(f"subject {subject} sample {index} must contain 24 columns")
        activity = int(columns[23])
        if activity < 0 or activity > 12:
            raise ValueError(f"subject {subject} sample {index} has invalid activity")
        if activity != previous:
            previous = activity
            ordinal = 0
            for series in samples.values():
                series.clear()
        if activity == 0:
            continue
        for position, column in POSITIONS.items():
            sample = tuple(float(value) for value in columns[column:column + 3])
            if not all(math.isfinite(value) for value in sample):
                raise ValueError(f"subject {subject} sample {index} has nonfinite acceleration")
            samples[position].append(sample)
        if len(samples["chest"]) == WINDOW:
            yield {
                "subject": subject,
                "start": index + 1 - WINDOW,
                "activity": activity,
                "ordinal": ordinal,
                **{position: features(series) for position, series in samples.items()},
            }
            ordinal += 1
            for series in samples.values():
                series.clear()


def archive_windows(source):
    if not source.is_file():
        raise ValueError("expected a regular MHEALTH archive")
    with source.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != ARCHIVE_BYTES:
            raise ValueError("expected the original 75,567,983 byte MHEALTH archive")
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        if sha256 != ARCHIVE_SHA256:
            raise ValueError("MHEALTH archive SHA256 mismatch")
        stream.seek(0)
        with ZipFile(stream) as archive:
            expected = {f"MHEALTHDATASET/mHealth_subject{subject}.log": size
                        for subject, size in enumerate(LOG_BYTES, start=1)}
            members = archive.infolist()
            names = {member.filename for member in members}
            if (len(names) != len(members)
                    or names != expected.keys() | {"MHEALTHDATASET/", "MHEALTHDATASET/README.txt"}):
                raise ValueError("unexpected MHEALTH archive members")
            for subject, (name, size) in enumerate(expected.items(), start=1):
                member = archive.getinfo(name)
                if member.file_size != size or member.compress_type != ZIP_DEFLATED or member.flag_bits & 1:
                    raise ValueError(f"unexpected subject {subject} log metadata")
                with archive.open(member) as lines:
                    yield from subject_windows(lines, subject)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output path already exists")
    try:
        windows = list(archive_windows(args.archive))
        if not windows:
            raise ValueError("no complete activity windows")
        data = {"schema": SCHEMA, "archiveSha256": ARCHIVE_SHA256, "windows": windows}
        with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
            generated = Path(directory) / "output.json"
            with generated.open("w", encoding="utf-8") as output:
                json.dump(data, output, allow_nan=False, separators=(",", ":"))
                output.write("\n")
            args.output.hardlink_to(generated)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"features: {len(windows)} paired windows from the pinned UCI MHEALTH archive")


if __name__ == "__main__":
    main()
