#!/usr/bin/env python3
"""Pinned, benchmark-only ExtraTrees comparison. No file/network/eval inputs."""
import json
import platform
import sys
import time

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

EXPECTED = {"numpy": "1.26.4", "scipy": "1.12.0", "scikit-learn": "1.7.2", "joblib": "1.4.2", "threadpoolctl": "3.5.0"}
VERSIONS = {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__}
if VERSIONS != EXPECTED:
    raise ValueError("Tree benchmark requires exact versions from packages/wsp/benchmarks/requirements.lock")
raw = sys.stdin.buffer.read(1_000_001)
if len(raw) > 1_000_000:
    raise ValueError("Tree benchmark input exceeds byte bound")
request = json.loads(raw)
if set(request) != {"schema", "jobs"} or request["schema"] != "ruv-fann.tree-benchmark-input/1" or len(request["jobs"]) != 10:
    raise ValueError("Invalid tree benchmark plan")
results = []
for job in request["jobs"]:
    if set(job) != {"id", "seed", "train", "validation", "test"} or job["seed"] not in [17, 43, 101, 211, 509]:
        raise ValueError("Invalid job shape or seed")
    parts = {}
    width = None
    for name in ["train", "validation", "test"]:
        part = job[name]
        if set(part) != {"features", "labels"}:
            raise ValueError("Invalid partition fields")
        if not isinstance(part["labels"], list) or any(type(label) is not int or label not in [0, 1, 2] for label in part["labels"]):
            raise ValueError("Labels must be integer class identifiers")
        features, labels = np.asarray(part["features"], dtype=np.float64), np.asarray(part["labels"], dtype=np.int64)
        if features.ndim != 2 or features.shape[1] not in [4, 13] or not 1 <= features.shape[0] <= 200 or labels.shape != (features.shape[0],):
            raise ValueError("Partition dimensions exceed plan")
        width = features.shape[1] if width is None else width
        if features.shape[1] != width:
            raise ValueError("Inconsistent feature dimensions")
        if not np.isfinite(features).all() or not np.isin(labels, [0, 1, 2]).all():
            raise ValueError("Invalid numeric input")
        parts[name] = features, labels
    candidates = []
    trained = []
    for depth in [3, None]:
        estimator = ExtraTreesClassifier(n_estimators=128, max_depth=depth, random_state=job["seed"], n_jobs=1)
        started = time.perf_counter()
        estimator.fit(*parts["train"])
        fit_ms = (time.perf_counter() - started) * 1000
        probabilities = estimator.predict_proba(parts["validation"][0])
        accuracy = accuracy_score(parts["validation"][1], estimator.classes_[np.argmax(probabilities, axis=1)])
        mse = mean_squared_error(np.eye(3)[parts["validation"][1]], probabilities)
        candidates.append({"maxDepth": depth, "trees": 128, "validationAccuracy": float(accuracy), "validationMSE": float(mse), "fitMs": fit_ms})
        trained.append(estimator)
    selected = min(range(2), key=lambda index: (-candidates[index]["validationAccuracy"], candidates[index]["validationMSE"], index))
    estimator = trained[selected]
    started = time.perf_counter()
    scores = estimator.predict_proba(parts["test"][0])
    prediction_ms = (time.perf_counter() - started) * 1000
    predicted = estimator.classes_[np.argmax(scores, axis=1)]
    results.append({"id": job["id"], "seed": job["seed"], "selected": selected, "candidates": candidates,
                    "testScores": scores.ravel().tolist(), "testPredictionMs": prediction_ms,
                    "oracle": {"accuracy": float(accuracy_score(parts["test"][1], predicted)),
                               "macroF1": float(f1_score(parts["test"][1], predicted, average="macro", labels=[0, 1, 2], zero_division=0)),
                               "mse": float(mean_squared_error(np.eye(3)[parts["test"][1]], scores))}})
print(json.dumps({"schema": "ruv-fann.tree-benchmark/1", "versions": VERSIONS, "python": platform.python_version(),
                  "estimator": "sklearn.ensemble.ExtraTreesClassifier", "nJobs": 1, "results": results}, allow_nan=False))
