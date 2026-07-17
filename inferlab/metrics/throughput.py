"""
Throughput metrics. Week 1 only needs single-stream throughput; the
concurrent-load variant gets exercised properly once batched.py exists in
Week 3, but write the signature now so runner.py doesn't need to change.
"""

from dataclasses import dataclass


@dataclass
class ThroughputReport:
    tokens_per_sec: float           # single-stream: generated_len / e2e_time
    requests_per_sec: float | None  # None for single-stream; used in Week 3


def compute_throughput(num_generated_tokens: int, e2e_seconds: float,
                        num_requests: int = 1) -> ThroughputReport:
    """
    TODO(ruchit): tokens_per_sec = num_generated_tokens / e2e_seconds.
    requests_per_sec = num_requests / e2e_seconds if num_requests > 1 else None.

    EDGE CASE: e2e_seconds == 0 (degenerate/mocked test case) -- guard
    against division by zero rather than letting it NaN silently into a
    benchmark CSV.
    """
    if e2e_seconds == 0:
        raise ValueError(f"e2e_seconds must be positive, got {e2e_seconds}")
    else:
        tokens_per_sec = num_generated_tokens / e2e_seconds
        requests_per_sec = num_requests / e2e_seconds if num_requests > 1 else None
    
        return ThroughputReport(tokens_per_sec=tokens_per_sec, requests_per_sec=requests_per_sec)