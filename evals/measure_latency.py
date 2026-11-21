"""p95 latency measurement against a RUNNING server (with a real LLM).

    python evals/measure_latency.py http://127.0.0.1:8000 30

Sends N employee messages (mixed read/write) and reports p50/p90/p95/p99 for
the full request round trip, LLM call included. The case-study budget is 5s.
"""
import statistics
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30

MESSAGES = [
    "What time is my drop today?",
    "What time is my pickup tomorrow?",
    "Any cab around 20:00 for my drop?",
    "What are my company's transport rules?",
]


def main() -> None:
    client = httpx.Client(base_url=BASE, timeout=30)
    tok = client.post("/auth/token", json={"employee_id": "acme-rhea"}).json()["access_token"]
    hdr = {"Authorization": f"Bearer {tok}"}

    latencies = []
    for i in range(N):
        msg = MESSAGES[i % len(MESSAGES)]
        t0 = time.perf_counter()
        resp = client.post("/chat", json={"message": msg}, headers=hdr)
        dt = time.perf_counter() - t0
        assert resp.status_code == 200, resp.text
        latencies.append(dt * 1000)
        print(f"{i + 1:3d}/{N} {dt * 1000:7.0f} ms  {msg}")

    latencies.sort()

    def pct(p: float) -> float:
        k = max(0, min(len(latencies) - 1, int(round(p / 100 * len(latencies))) - 1))
        return latencies[k]

    print("\nn={} mean={:.0f}ms p50={:.0f}ms p90={:.0f}ms p95={:.0f}ms p99={:.0f}ms".format(
        N, statistics.mean(latencies), pct(50), pct(90), pct(95), pct(99)))
    budget_ms = 5000
    over = sum(1 for x in latencies if x > budget_ms)
    print(f"p95 budget 5000ms: {'MET' if pct(95) <= budget_ms else 'MISSED'} ({over} samples over)")


if __name__ == "__main__":
    main()
