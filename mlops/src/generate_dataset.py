"""Generate a mock Kubernetes failure dataset for MLOps demo purposes."""

import argparse
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

CLASS_WEIGHTS = {
    "healthy": 0.42,
    "crash_loop": 0.14,
    "oom_killed": 0.12,
    "image_pull_error": 0.10,
    "scheduling_failure": 0.10,
    "probe_failure": 0.12,
}

# Each class: (mean, std) per numeric feature, and categorical distribution
CLASS_PROFILES = {
    "healthy": {
        "restart_count":              (0.3, 0.5),
        "cpu_usage_pct":              (25,  12),
        "memory_usage_pct":           (40,  12),
        "pod_ready_probability":      0.97,
        "last_exit_code":             (0,   0),
        "oom_killed_count":           (0,   0),
        "image_pull_errors":          (0,   0),
        "failed_scheduling_events":   (0,   0),
        "readiness_probe_failures":   (0.2, 0.4),
        "waiting_reasons":            ["Running", "Running", "Running", "Unknown"],
    },
    "crash_loop": {
        "restart_count":              (7,   3.5),
        "cpu_usage_pct":              (66,  16),
        "memory_usage_pct":           (67,  15),
        "pod_ready_probability":      0.08,
        "last_exit_code":             (1,   1),
        "oom_killed_count":           (0,   0),
        "image_pull_errors":          (0,   0),
        "failed_scheduling_events":   (0,   0),
        "readiness_probe_failures":   (4,   2.5),
        "waiting_reasons":            ["CrashLoopBackOff", "Error", "Running"],
    },
    "oom_killed": {
        "restart_count":              (3,   1.5),
        "cpu_usage_pct":              (55,  12),
        "memory_usage_pct":           (91,  8),
        "pod_ready_probability":      0.12,
        "last_exit_code":             (137, 3),
        "oom_killed_count":           (2.5, 1.5),
        "image_pull_errors":          (0,   0),
        "failed_scheduling_events":   (0,   0),
        "readiness_probe_failures":   (2,   1),
        "waiting_reasons":            ["OOMKilled", "Error", "CrashLoopBackOff"],
    },
    "image_pull_error": {
        "restart_count":              (0.2, 0.4),
        "cpu_usage_pct":              (10,  5),
        "memory_usage_pct":           (16,  5),
        "pod_ready_probability":      0.03,
        "last_exit_code":             (0,   0.5),
        "oom_killed_count":           (0,   0),
        "image_pull_errors":          (5,   2),
        "failed_scheduling_events":   (0,   0),
        "readiness_probe_failures":   (0.2, 0.4),
        "waiting_reasons":            ["ImagePullBackOff", "ErrImagePull", "Pending"],
    },
    "scheduling_failure": {
        "restart_count":              (0.2, 0.4),
        "cpu_usage_pct":              (8,   5),
        "memory_usage_pct":           (15,  6),
        "pod_ready_probability":      0.02,
        "last_exit_code":             (0,   0),
        "oom_killed_count":           (0,   0),
        "image_pull_errors":          (0,   0),
        "failed_scheduling_events":   (6,   2),
        "readiness_probe_failures":   (0.2, 0.4),
        "waiting_reasons":            ["Pending", "Unschedulable", "ContainerCreating"],
    },
    "probe_failure": {
        "restart_count":              (2,   1),
        "cpu_usage_pct":              (48,  12),
        "memory_usage_pct":           (60,  10),
        "pod_ready_probability":      0.18,
        "last_exit_code":             (0.5, 0.5),
        "oom_killed_count":           (0,   0),
        "image_pull_errors":          (0,   0),
        "failed_scheduling_events":   (0,   0),
        "readiness_probe_failures":   (5,   2.5),
        "waiting_reasons":            ["Running", "Error", "CrashLoopBackOff"],
    },
}

POST_UPGRADE_PROBE_PROFILE = {
    "restart_count": (8, 1.8),
    "cpu_usage_pct": (76, 7),
    "memory_usage_pct": (72, 8),
    "pod_ready_probability": 0.02,
    "last_exit_code": (1, 0.4),
    "oom_killed_count": (0, 0),
    "image_pull_errors": (0, 0),
    "failed_scheduling_events": (0, 0),
    "readiness_probe_failures": (4, 1.2),
    "waiting_reasons": ["CrashLoopBackOff", "CrashLoopBackOff", "Error"],
}


def sample_class(
    label: str,
    rng: np.random.Generator,
    ambiguity: float,
    profile=None,
) -> dict:
    p = profile or CLASS_PROFILES[label]
    healthy = CLASS_PROFILES["healthy"]

    def gauss_int(key, lo=0, hi=None):
        mean, std = p[key]
        if label != "healthy" and rng.random() < ambiguity:
            healthy_mean, _ = healthy[key]
            mean = 0.65 * mean + 0.35 * healthy_mean
            std *= 1.25
        val = int(round(rng.normal(mean, std)))
        val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    def gauss_float(key, lo=0.0, hi=100.0):
        mean, std = p[key]
        if label != "healthy" and rng.random() < ambiguity:
            healthy_mean, _ = healthy[key]
            mean = 0.65 * mean + 0.35 * healthy_mean
            std *= 1.25
        val = float(round(rng.normal(mean, std), 1))
        return max(lo, min(hi, val))

    return {
        "restart_count":            gauss_int("restart_count"),
        "cpu_usage_pct":            gauss_float("cpu_usage_pct"),
        "memory_usage_pct":         gauss_float("memory_usage_pct"),
        "pod_ready":                int(rng.random() < p["pod_ready_probability"]),
        "last_exit_code":           gauss_int("last_exit_code"),
        "waiting_reason":           str(rng.choice(p["waiting_reasons"])),
        "oom_killed_count":         gauss_int("oom_killed_count"),
        "image_pull_errors":        gauss_int("image_pull_errors"),
        "failed_scheduling_events": gauss_int("failed_scheduling_events"),
        "readiness_probe_failures": gauss_int("readiness_probe_failures"),
        "label":                    label,
    }


def generate(
    rows: int,
    seed: int,
    ambiguity: float,
    label_noise: float,
    concept_version: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift_rows = max(1, rows // 5) if concept_version == "post-upgrade" else 0
    if drift_rows >= rows:
        raise ValueError("Post-upgrade generation requires at least two rows.")
    baseline_rows = rows - drift_rows
    labels = rng.choice(
        FAILURE_CLASSES,
        size=baseline_rows,
        p=[CLASS_WEIGHTS[label] for label in FAILURE_CLASSES],
    )
    records = [sample_class(str(label), rng, ambiguity) for label in labels]

    if concept_version == "post-upgrade":
        records.extend(
            sample_class(
                "probe_failure",
                rng,
                ambiguity=0.0,
                profile=POST_UPGRADE_PROBE_PROFILE,
            )
            for _ in range(drift_rows)
        )

    noisy_rows = rng.random(baseline_rows) < label_noise
    for index in np.flatnonzero(noisy_rows):
        current = records[index]["label"]
        alternatives = [label for label in FAILURE_CLASSES if label != current]
        records[index]["label"] = str(rng.choice(alternatives))

    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate mock Kubernetes failure dataset.")
    parser.add_argument("--rows", type=int, default=6000, help="Total rows to generate.")
    parser.add_argument("--output", default="data/kubernetes_failures.csv",          help="Output CSV path.")
    parser.add_argument("--seed",   type=int, default=42,                            help="Random seed.")
    parser.add_argument(
        "--ambiguity",
        type=float,
        default=0.18,
        help="Fraction of failure samples blended toward healthy behavior.",
    )
    parser.add_argument(
        "--concept-version",
        choices=["baseline", "post-upgrade"],
        default="baseline",
        help="Generate the original or post-platform-upgrade feature/label relationship.",
    )
    parser.add_argument(
        "--label-noise",
        type=float,
        default=0.02,
        help="Fraction of rows assigned a different label to simulate imperfect incidents.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.ambiguity <= 1 or not 0 <= args.label_noise <= 1:
        raise ValueError("--ambiguity and --label-noise must be between 0 and 1.")
    df = generate(
        args.rows,
        args.seed,
        args.ambiguity,
        args.label_noise,
        args.concept_version,
    )
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} rows → {args.output}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
