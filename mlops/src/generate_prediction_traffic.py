"""Send repeatable production-like workload observations to the prediction API."""

import argparse
import json
import random
import time
import urllib.request


BASELINE_SCENARIOS = {
    "checkout": {
        "expected_failure_type": "healthy",
        "restart_count": 0,
        "cpu_usage_pct": 31.0,
        "memory_usage_pct": 43.0,
        "pod_ready": 1,
        "last_exit_code": 0,
        "waiting_reason": "Running",
        "oom_killed_count": 0,
        "image_pull_errors": 0,
        "failed_scheduling_events": 0,
        "readiness_probe_failures": 0,
    },
    "payments": {
        "expected_failure_type": "oom_killed",
        "restart_count": 3,
        "cpu_usage_pct": 61.0,
        "memory_usage_pct": 94.0,
        "pod_ready": 0,
        "last_exit_code": 137,
        "waiting_reason": "OOMKilled",
        "oom_killed_count": 2,
        "image_pull_errors": 0,
        "failed_scheduling_events": 0,
        "readiness_probe_failures": 2,
    },
    "catalog": {
        "expected_failure_type": "probe_failure",
        "restart_count": 2,
        "cpu_usage_pct": 49.0,
        "memory_usage_pct": 63.0,
        "pod_ready": 0,
        "last_exit_code": 1,
        "waiting_reason": "Running",
        "oom_killed_count": 0,
        "image_pull_errors": 0,
        "failed_scheduling_events": 0,
        "readiness_probe_failures": 6,
    },
    "recommendations": {
        "expected_failure_type": "crash_loop",
        "restart_count": 8,
        "cpu_usage_pct": 76.0,
        "memory_usage_pct": 72.0,
        "pod_ready": 0,
        "last_exit_code": 1,
        "waiting_reason": "CrashLoopBackOff",
        "oom_killed_count": 0,
        "image_pull_errors": 0,
        "failed_scheduling_events": 0,
        "readiness_probe_failures": 4,
    },
}

DRIFT_SCENARIOS = {
    "recommendations": {
        "expected_failure_type": "probe_failure",
        "restart_count": 8,
        "cpu_usage_pct": 76.0,
        "memory_usage_pct": 72.0,
        "pod_ready": 0,
        "last_exit_code": 1,
        "waiting_reason": "CrashLoopBackOff",
        "oom_killed_count": 0,
        "image_pull_errors": 0,
        "failed_scheduling_events": 0,
        "readiness_probe_failures": 4,
    },
}

DRIFT_AFTER_RETRAIN_SCENARIOS = {
    "recommendations": {
        "expected_failure_type": "probe_failure",
        "restart_count": 8,
        "cpu_usage_pct": 76.0,
        "memory_usage_pct": 72.0,
        "pod_ready": 0,
        "last_exit_code": 1,
        "waiting_reason": "CrashLoopBackOff",
        "oom_killed_count": 0,
        "image_pull_errors": 0,
        "failed_scheduling_events": 0,
        "readiness_probe_failures": 4,
    },
}


def jitter(payload, rng):
    result = dict(payload)
    for key in ("cpu_usage_pct", "memory_usage_pct"):
        result[key] = max(0.0, min(100.0, round(result[key] + rng.uniform(-3, 3), 1)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://failure-model-api.mlops.local/predict")
    parser.add_argument(
        "--host-header",
        default=None,
        help="Optional HTTP Host header when sending traffic directly to a gateway IP.",
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenario",
        choices=["baseline", "drift", "drift_after_retrain"],
        default="baseline",
        help="Traffic and delayed ground-truth scenario.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.scenario == "baseline":
        scenarios = BASELINE_SCENARIOS
    elif args.scenario == "drift":
        scenarios = DRIFT_SCENARIOS
    else:
        scenarios = DRIFT_AFTER_RETRAIN_SCENARIOS
    correct = 0
    total = 0
    for _ in range(args.rounds):
        for app_name, scenario in scenarios.items():
            payload = {"app_name": app_name, **jitter(scenario, rng)}
            headers = {"Content-Type": "application/json"}
            if args.host_header:
                headers["Host"] = args.host_header
            request = urllib.request.Request(
                args.url,
                data=json.dumps(payload).encode(),
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                result = json.load(response)
            print(
                app_name,
                result["predicted_failure_type"],
                result["risk_score"],
                result["risk_level"],
                "business_risk=" + str(result.get("business_risk_score")),
                "criticality=" + str(result.get("criticality_label")),
                "traffic_share=" + str(result.get("traffic_share")),
                "expected=" + str(result["expected_failure_type"]),
                "correct=" + str(result["quality_match"]),
            )
            total += 1
            correct += int(bool(result["quality_match"]))
            time.sleep(args.interval)
    print(f"production_accuracy={correct / total:.4f} correct={correct} total={total}")


if __name__ == "__main__":
    main()
