#!/usr/bin/env python3
"""Retain benchmark samples, compute ratios and plot measured WASM throughput.

Uses NumPy/Matplotlib only for offline reporting, not as runtime dependencies.
Confidence intervals resample warmed timing blocks; they do not cover different
machines, workloads, cold starts or correlated host noise.
"""
import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def ratio(numerator, denominator, paired=True):
    a, b = np.array(numerator), np.array(denominator)
    if len(a) != len(b) or not np.all(a > 0) or not np.all(b > 0):
        raise ValueError("Expected positive equal-length sample arrays")
    # Reset for every comparison so a report is independent of traversal order.
    rng = np.random.default_rng(20261108)
    indices = rng.integers(0, len(a), size=(5000, len(a)))
    other = indices if paired else rng.integers(0, len(b), size=indices.shape)
    draws = a[indices].mean(axis=1) / b[other].mean(axis=1)
    return {"mean_ratio": float(a.mean() / b.mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "paired_blocks": paired}


def key(case):
    return (tuple(case["layers"]), case["activation"], case["rows"])


def samples(case, api):
    return case["timings"][api]["samples_ns"]


def enrich(data):
    before = {run["variant"]: run for run in data["results"] if run["revision"] == "before"}
    for run in data["results"]:
        if run["revision"] != "current":
            continue
        old_inference = {key(case): case for case in before[run["variant"]]["inference"]}
        old_calibration = {(c["kind"], c["levels"]): c for c in before[run["variant"]]["calibration"]}
        for case in run["inference"]:
            if case["maximum_absolute_error"] != 0:
                raise ValueError("Expected exact output agreement for measured fixtures")
            denominator = samples(case, "prepared_batch")
            case["ratios"] = {
                "same_binary_single": ratio(samples(case, "prepared_single_loop"), denominator),
                "same_binary_legacy": ratio(samples(case, "legacy_loop"), denominator),
                "previous_legacy": ratio(samples(old_inference[key(case)], "legacy_loop"), denominator, False),
            }
        for case in run["calibration"]:
            if not case["exact_before_after_output_validated"]:
                raise ValueError("Missing exact calibration JSON comparison")
            old = old_calibration[(case["kind"], case["levels"])]
            case["ratios"] = {api: ratio(samples(old, api), samples(case, api), False)
                              for api in ["fit_and_free", "apply_json"]}
    return data


def compact(data, destination):
    # One case per line keeps all raw samples while remaining reviewable.
    metadata = {k: v for k, v in data.items() if k != "results"}
    lines = ["{", '  "metadata": ' + json.dumps(metadata, separators=(",", ":")) + ",", '  "results": [']
    for index, run in enumerate(data["results"]):
        meta = {k: v for k, v in run.items() if k not in ["inference", "calibration"]}
        lines.append('    {"artifact":' + json.dumps(meta, separators=(",", ":")) + ',"inference":[')
        for i, case in enumerate(run["inference"]):
            lines.append("      " + json.dumps(case, separators=(",", ":")) + ("," if i + 1 < len(run["inference"]) else ""))
        lines.append('    ],"calibration":[')
        for i, case in enumerate(run["calibration"]):
            lines.append("      " + json.dumps(case, separators=(",", ":")) + ("," if i + 1 < len(run["calibration"]) else ""))
        lines.append("    ]}" + ("," if index + 1 < len(data["results"]) else ""))
    lines.extend(["  ]", "}"])
    destination.write_text("\n".join(lines) + "\n")


def plot(datasets, destination):
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "2026-11-08-batch",
                         "font.family": "DejaVu Sans", "font.size": 10})
    shapes = [(5, 7, 3), (100, 50, 25, 10), (784, 128, 64, 10)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), sharex=True, sharey=True)
    colors = {"core": "#365faa", "simd": "#087f73"}
    for row, data in enumerate(datasets):
        for col, shape in enumerate(shapes):
            ax = axes[row, col]
            for run in data["results"]:
                if run["revision"] != "current":
                    continue
                for activation, style in [("Linear", "-"), ("Sigmoid", "--")]:
                    cases = [c for c in run["inference"] if tuple(c["layers"]) == shape and c["activation"] == activation]
                    ratios = [c["ratios"]["same_binary_single"] for c in cases]
                    y = np.array([r["mean_ratio"] for r in ratios])
                    bounds = np.array([r["ci95"] for r in ratios]).T
                    ax.errorbar([c["rows"] for c in cases], y,
                                yerr=np.maximum(0, [y - bounds[0], bounds[1] - y]),
                                color=colors[run["variant"]], linestyle=style, marker="o", markersize=3,
                                linewidth=1.6, capsize=2, label=f'{run["variant"].upper()} · {activation}')
            ax.axhline(1, color="#777", linewidth=.9)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_xticks([1, 3, 8, 32, 128], labels=[1, 3, 8, 32, 128])
            ax.set_yticks([.5, 1, 2, 4, 8, 16, 32], labels=["0.5×", "1×", "2×", "4×", "8×", "16×", "32×"])
            ax.grid(True, alpha=.15)
            ax.set_title(" → ".join(map(str, shape)))
            if col == 0:
                ax.set_ylabel(f'{data["runtime"]}\nThroughput ratio')
            if row == 1:
                ax.set_xlabel("Rows per batch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Measured Float64 WASM batching: warmed public APIs", fontsize=15)
    fig.text(.5, .925, "Relative to repeated prepared single calls · validation and copies included · 20 blocks · 95% bootstrap intervals", ha="center", color="#444")
    fig.tight_layout(rect=(0, .06, 1, .91))
    fig.savefig(destination / "batch-throughput.svg", metadata={"Title": "Measured WASM batch throughput", "Date": "2026-11-08",
                "Description": "Node and Chromium; core and SIMD; Linear and Sigmoid. Ratios above one improve on repeated prepared single calls. Error bars resample timing blocks, not hardware or workload diversity."})
    fig.savefig(destination / "batch-throughput.png", dpi=180)
    plt.close(fig)
    svg = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg)
    document = ET.fromstring((destination / "batch-throughput.svg").read_text())
    document.set("role", "img")
    document.set("aria-labelledby", "batch-title batch-desc")
    for index, (tag, text) in enumerate([
        ("title", "Measured WASM batch throughput"),
        ("desc", "Node and Chromium. Independent Float64 sample batches compared with repeated prepared single calls, including small-batch regressions and timing-block confidence intervals.")
    ]):
        element = ET.Element(f"{{{svg}}}{tag}", {"id": f"batch-{'title' if tag == 'title' else 'desc'}"})
        element.text = text
        document.insert(index, element)
    motion = ET.Element(f"{{{svg}}}style")
    motion.text = "@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}"
    document.insert(2, motion)
    # Compact generated markup, keeping text, paths, metadata and vector detail.
    for element in document.iter():
        if element.text is not None and not element.text.strip():
            element.text = None
        if element.tail is not None and not element.tail.strip():
            element.tail = None
    (destination / "batch-throughput.svg").write_bytes(ET.tostring(document, encoding="utf-8", xml_declaration=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="Directory containing node/browser-speed-throughput.json and native receipts")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    out = args.docs / "benchmarks"
    assets = args.docs / "assets"
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    datasets = []
    summary = {"schema_version": 1, "bootstrap": {"replicates": 5000, "seed": 20261108,
               "scope": "Timing-block sampling only; same-binary API ratios paired, sequential previous/current artifacts unpaired. Host drift and workload diversity not represented."}, "runtimes": []}
    for runtime in ["node", "browser"]:
        source = args.evidence / f"{runtime}-speed-throughput.json"
        data = enrich(json.loads(source.read_text()))
        data["raw_evidence_sha256"] = sha(source)
        datasets.append(data)
        compact(data, out / f"{runtime}-2026-11-08.json")
        size = json.loads((args.evidence / f"{runtime}-throughput.json").read_text())
        comparisons = []
        for run in data["results"]:
            if run["revision"] != "current":
                continue
            old = next(r for r in size["results"] if r["revision"] == "current" and r["variant"] == run["variant"])
            ratios = [a["timings"]["prepared_batch"]["mean_ns"] / b["timings"]["prepared_batch"]["mean_ns"]
                      for a, b in zip(old["inference"], run["inference"])]
            comparisons.append({"variant": run["variant"], "cases": len(ratios),
                                "o3_over_size_min_median_max": [min(ratios), float(np.median(ratios)), max(ratios)]})
        summary["runtimes"].append({"runtime": data["runtime"], "raw_sha256": sha(source),
                                    "exact_calibration_json_cases": data["exact_before_after_calibration_cases"], "profile_comparison": comparisons})
    for source, target in [("prepared/native-final.csv", "native-batch-2026-11-08.csv"),
                           ("calibration/baseline.csv", "native-calibration-before-2026-11-08.csv"),
                           ("calibration/final.csv", "native-calibration-after-2026-11-08.csv")]:
        shutil.copyfile(args.evidence / source, out / target)
    plot(datasets, assets)
    names = ["node-2026-11-08.json", "browser-2026-11-08.json", "native-batch-2026-11-08.csv",
             "native-calibration-before-2026-11-08.csv", "native-calibration-after-2026-11-08.csv"]
    summary["files"] = {name: sha(out / name) for name in names}
    (out / "summary-2026-11-08.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
