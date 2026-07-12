from singularity.core.metrics import Metrics


def test_counters_and_percentiles():
    m = Metrics()
    m.incr("teleports", 3)
    m.incr("teleports")
    for v in [1.0, 2.0, 3.0, 100.0]:
        m.observe("latency_ms", v)
    snap = m.snapshot()
    assert snap["teleports"] == 4.0
    assert snap["latency_ms_p99"] >= snap["latency_ms_p50"]
    assert "singularity_teleports" in m.prometheus()
