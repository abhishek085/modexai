import pytest

from modexai.backends import SimulatedBackend
from modexai.bench import run_benchmark, verdict
from modexai.checks import GRADERS
from modexai.suites import get_suite, list_suites, suite_from_samples

SUITES = list_suites()


def test_suites_exist_and_are_well_formed():
    assert len(SUITES) >= 8
    for s in SUITES:
        ids = [it["id"] for it in s.items]
        assert len(ids) == len(set(ids)), s.id
        assert len(s.items) >= 10, s.id
        for it in s.items:
            assert it["check"]["type"] in GRADERS, (s.id, it["id"])


@pytest.mark.parametrize("suite", SUITES, ids=lambda s: s.id)
def test_perfect_model_scores_100_and_broken_model_scores_0(suite):
    """Proves every gold answer agrees with its grader (and graders reject wrong answers)."""
    backend = SimulatedBackend({"perfect": 1.0, "broken": 0.0})
    perfect, broken = run_benchmark(["perfect", "broken"], suite, backend)
    assert perfect.score == 1.0, [c for c in perfect.cases if not c.passed][:2]
    assert broken.score == 0.0, [c for c in broken.cases if c.passed][:2]


def test_suite_hash_is_stable():
    assert get_suite("ticket-triage").sha == get_suite("ticket-triage@1").sha


def test_custom_samples_infer_labels_and_system():
    s = suite_from_samples([{"prompt": f"t{i}", "expected": lab} for i, lab in enumerate(["a", "b", "a", "b"])])
    assert s.items[0]["check"]["type"] == "label"
    assert "a, b" in s.system


def test_verdict_hires_smallest_passing():
    suite = get_suite("ticket-triage")
    backend = SimulatedBackend({"big": 1.0, "small": 1.0, "tiny": 0.3})
    res = run_benchmark(["big", "small", "tiny"], suite, backend)
    res[0].size_gb, res[1].size_gb, res[2].size_gb = 5.0, 1.0, 0.3
    v = verdict(res, 0.8)
    assert v.winner.model == "small"


def test_verdict_says_when_cloud_earns_its_place():
    suite = get_suite("ticket-triage")
    res = run_benchmark(["weak"], suite, SimulatedBackend({"weak": 0.2}))
    v = verdict(res, 0.8)
    assert v.winner is None and "cloud" in v.reason
