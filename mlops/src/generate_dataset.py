"""Generate a mock Kubernetes failure dataset for MLOps demo purposes."""

import argparse
import random

import numpy as np
import pandas as pd


FAILURE_CLASSES = [
    "healthy",
    "crash_loop",
    "oom_killed",
    "image_pull_error",
    "scheduling_failure",
    "probe_failure",
]

# Each class: (mean, std) per numeric feature, and categorical distribution
CLASS_PROFILES = {
    "healthy": {
        "restart_count":              (0.3, 0.5),
        "cpu_usage_pct":              (25,  12),
        "memory_usage_pct":           (40,  12),
        "pod_ready":                  (1,   0),     # always ready
        "last_exit_code":             (0,   0),
        "oom_killed_count":           (0,   0),
        "image_pull_errors":          (0,   0),
        "failed_scheduling_events":   (0,   0),
        "readiness_probe_failures":   (0.2, 0.4),
        "waiting_reasons":            ["None"],
    },
    "crash_loop": {
        "restart_count":              (8,   3),
        "cpu_usage_pct":              (78,  10),
        "memory_usage_pct":           (70,  10),
        "pod_ready":                  (0,   0),
        "last_exit_code":             (1,   0.5),
        "oom_killed_count":           (0,   0),
        "image_pull_errors":          (0,   0),
        "failed_scheduling_events":   (0,   0),
        "readiness_probe_failures":   (5,   2),
        "waiting_reasons":            ["CrashLoopBackOff", "Error"],
    },
    "oom_killed": {
        "restart_count":              (3,   1.5),
        "cpu_usage_pct":              (55,  12),
        "memory_usage_pct":           (95,  4),
        "pod_ready":                  (0,   0),
        "last_exit_code":             (137, 0),
        "oom_killed_count":           (3,   1.5),
        "image_pull_errors":          (0,   0),
        "failed_scheduling_events":   (0,   0),
        "readiness_probe_failures":   (2,   1),
        "waiting_reasons":            ["OOMKilled", "Error"],
    },
    "image_pull_error": {
        "restart_count":              (0.2, 0.4),
        "cpu_usage_pct":              (10,  5),
        "memory_usage_pct":           (16,  5),
        "pod_ready":                  (0,   0),
        "last_exit_code":             (0,   0.5),
        "oom_killed_count":           (0,   0),
        "image_pull_errors":          (5,   2),
        "failed_scheduling_events":   (0,   0),
        "readiness_probe_failures":   (0.2, 0.4),
        "waiting_reasons":            ["ImagePullBackOff", "ErrImagePull"],
    },
    "scheduling_failure": {
        "restart_count":              (0.2, 0.4),
        "cpu_usage_pct":              (8,   5),
        "memory_usage_pct":           (15,  6),
        "pod_ready":                  (0,   0),
        "last_exit_code":             (0,   0),
        "oom_killed_count":           (0,   0),
        "image_pull_errors":          (0,   0),
        "failed_scheduling_events":   (6,   2),
        "readiness_probe_failures":   (0.2, 0.4),
        "waiting_reasons":            ["Pending", "Unschedulable"],
    },
    "probe_failure": {
        "restart_count":              (2,   1),
        "cpu_usage_pct":              (48,  12),
        "memory_usage_pct":           (60,  10),
        "pod_ready":                  (0,   0),
        "last_exit_code":             (0.5, 0.5),
        "oom_killed_count":           (0,   0),
        "image_pull_errors":          (0,   0),
        "failed_scheduling_events":   (0,   0),
        "readiness_probe_failures":   (6,   2),
        "waiting_reasons":            ["Running", "Error"],
    },
}


def sample_class(label: str, rng: np.random.Generator) -> dict:
    p = CLASS_PROFILES[label]

    def gauss_int(key, lo=0, hi=None):
        mean, std = p[key]
        val = int(round(rng.normal(mean, std)))
        val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    def gauss_float(key, lo=0.0, hi=100.0):
        mean, std = p[key]
        val = float(round(rng.normal(mean, std), 1))
        return max(lo, min(hi, val))

    return {
        "restart_count":            gauss_int("restart_count"),
        "cpu_usage_pct":            gauss_float("cpu_usage_pct"),
        "memory_usage_pct":         gauss_float("memory_usage_pct"),
        "pod_ready":                int(p["pod_ready"][0]),
        "last_exit_code":           gauss_int("last_exit_code"),
        "waiting_reason":           random.choice(p["waiting_reasons"]),
        "oom_killed_count":         gauss_int("oom_killed_count"),
        "image_pull_errors":        gauss_int("image_pull_errors"),
        "failed_scheduling_events": gauss_int("failed_scheduling_events"),
        "readiness_probe_failures": gauss_int("readiness_probe_failures"),
        "label":                    label,
    }


def generate(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows_per_class = rows // len(FAILURE_CLASSES)
    records = []
    for label in FAILURE_CLASSES:
        for _ in range(rows_per_class):
            records.append(sample_class(label, rng))
    # fill remainder evenly
    remaining = rows - len(records)
    for i in range(remaining):
        label = FAILURE_CLASSES[i % len(FAILURE_CLASSES)]
        records.append(sample_class(label, rng))

    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate mock Kubernetes failure dataset.")
    parser.add_argument("--rows",   type=int, default=500,                          help="Total rows to generate.")
    parser.add_argument("--output", default="data/kubernetes_failures.csv",          help="Output CSV path.")
    parser.add_argument("--seed",   type=int, default=42,                            help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = generate(args.rows, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} rows → {args.output}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
