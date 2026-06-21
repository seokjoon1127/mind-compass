"""Unit tests for the pure-math compute core (backend/compute.py).

Run from the project root so `backend` is importable::

    .venv/Scripts/python.exe -m pytest tests/test_compute.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend import compute


# --------------------------------------------------------------------------- #
# normalize_scores
# --------------------------------------------------------------------------- #

def test_normalize_minmax_to_unit_interval():
    raw = {
        "o1": {"f1": 0.2, "f2": 0.9},
        "o2": {"f1": 0.4, "f2": 0.1},
        "o3": {"f1": 0.6, "f2": 0.5},
    }
    directions = {"f1": "higher_better", "f2": "lower_better"}
    out = compute.normalize_scores(raw, directions)

    # f1 spans [0.2, 0.6] -> min->0, max->1, mid linear.
    assert out["o1"]["f1"] == pytest.approx(0.0)
    assert out["o3"]["f1"] == pytest.approx(1.0)
    assert out["o2"]["f1"] == pytest.approx(0.5)

    # f2 spans [0.1, 0.9]; direction is informational only -> NOT inverted.
    assert out["o1"]["f2"] == pytest.approx(1.0)
    assert out["o2"]["f2"] == pytest.approx(0.0)
    assert out["o3"]["f2"] == pytest.approx(0.5)

    # All normalized values land in [0, 1].
    for oid in out:
        for v in out[oid].values():
            assert 0.0 <= v <= 1.0


def test_normalize_flat_factor_maps_to_half():
    raw = {
        "o1": {"flat": 0.7, "spread": 0.0},
        "o2": {"flat": 0.7, "spread": 1.0},
    }
    out = compute.normalize_scores(raw, {"flat": "higher_better", "spread": "higher_better"})
    assert out["o1"]["flat"] == pytest.approx(0.5)
    assert out["o2"]["flat"] == pytest.approx(0.5)
    # The genuinely-spread factor still scales normally.
    assert out["o1"]["spread"] == pytest.approx(0.0)
    assert out["o2"]["spread"] == pytest.approx(1.0)


def test_normalize_preserves_shape():
    raw = {
        "o1": {"f1": 0.1, "f2": 0.2, "f3": 0.3},
        "o2": {"f1": 0.4, "f2": 0.5, "f3": 0.6},
    }
    out = compute.normalize_scores(raw, {"f1": "higher_better", "f2": "higher_better", "f3": "lower_better"})
    assert set(out.keys()) == set(raw.keys())
    for oid in raw:
        assert set(out[oid].keys()) == set(raw[oid].keys())


def test_normalize_empty():
    assert compute.normalize_scores({}, {}) == {}


# --------------------------------------------------------------------------- #
# refine_weights
# --------------------------------------------------------------------------- #

def test_refine_empty_comparisons_returns_prior():
    prior = {"a": 0.5, "b": 0.3, "c": 0.2}
    out = compute.refine_weights(prior, [], lam=0.5)
    assert set(out.keys()) == set(prior.keys())
    for fid in prior:
        assert out[fid] == pytest.approx(prior[fid], abs=1e-6)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)


def test_refine_strong_i_over_j_raises_weight():
    prior = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    # Strong, repeated observation that a > b.
    comparisons = [
        {"i": "a", "j": "b", "target": 1.0, "weight": 5.0},
        {"i": "a", "j": "b", "target": 1.0, "weight": 5.0},
    ]
    out = compute.refine_weights(prior, comparisons, lam=0.5)
    assert out["a"] > out["b"]
    # a should also exceed its prior, b should drop below it.
    assert out["a"] > prior["a"]
    assert out["b"] < prior["b"]


def test_refine_output_is_valid_distribution():
    prior = {"a": 0.4, "b": 0.35, "c": 0.25}
    comparisons = [
        {"i": "a", "j": "c", "target": 0.8, "weight": 1.0},
        {"i": "b", "j": "c", "target": 0.7, "weight": 1.0},
        {"i": "a", "j": "b", "target": 0.5, "weight": 0.5},
    ]
    out = compute.refine_weights(prior, comparisons, lam=0.5)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)
    for v in out.values():
        assert v >= 0.0


def test_refine_target_zero_favors_j():
    prior = {"a": 0.5, "b": 0.5}
    # target 0.0 means j (=b) is clearly more important.
    comparisons = [{"i": "a", "j": "b", "target": 0.0, "weight": 5.0}]
    out = compute.refine_weights(prior, comparisons, lam=0.5)
    assert out["b"] > out["a"]


def test_refine_ignores_unknown_factor_ids():
    prior = {"a": 0.5, "b": 0.5}
    comparisons = [{"i": "a", "j": "ghost", "target": 1.0, "weight": 5.0}]
    out = compute.refine_weights(prior, comparisons, lam=0.5)
    # The bad comparison is dropped -> falls back to prior.
    assert out["a"] == pytest.approx(0.5, abs=1e-6)
    assert out["b"] == pytest.approx(0.5, abs=1e-6)


def test_refine_single_factor():
    out = compute.refine_weights({"only": 1.0}, [], lam=0.5)
    assert out == {"only": pytest.approx(1.0)}


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #

def test_aggregate_matches_manual_wsm():
    weights = {"f1": 0.6, "f2": 0.4}
    norm = {
        "o1": {"f1": 1.0, "f2": 0.0},
        "o2": {"f1": 0.0, "f2": 1.0},
    }
    out = compute.aggregate(weights, norm)
    assert out["o1"] == pytest.approx(0.6)
    assert out["o2"] == pytest.approx(0.4)


def test_aggregate_missing_cell_is_zero():
    weights = {"f1": 0.5, "f2": 0.5}
    norm = {"o1": {"f1": 1.0}}  # f2 missing
    out = compute.aggregate(weights, norm)
    assert out["o1"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# sensitivity
# --------------------------------------------------------------------------- #

def test_sensitivity_flip_threshold_finite_and_correct():
    # 2 options, 2 factors. Crafted so that shifting weight from f1 to f2 flips.
    # o1 dominates on f1, o2 dominates on f2.
    norm = {
        "o1": {"f1": 1.0, "f2": 0.0},
        "o2": {"f1": 0.0, "f2": 1.0},
    }
    weights = {"f1": 0.6, "f2": 0.4}
    sens = compute.sensitivity(weights, norm)

    # With these weights: U(o1)=0.6, U(o2)=0.4 -> o1 wins.
    assert sens["ranking"] == ["o1", "o2"]
    assert sens["utilities"]["o1"] == pytest.approx(0.6)
    assert sens["utilities"]["o2"] == pytest.approx(0.4)
    assert sens["margin"] == pytest.approx(0.2)

    # The winner flips when w_f1 drops to 0.5 (tie) / below 0.5.
    # Current w_f1 = 0.6, so the flip delta is ~0.1.
    flip_f1 = sens["flip_thresholds"]["f1"]
    assert flip_f1 is not None
    assert flip_f1 == pytest.approx(0.1, abs=0.02)

    # Raising w_f2 renormalizes w_f1 = 1 - w_f2 (only two factors), so the winner
    # flips when w_f2 crosses 0.5. Current w_f2 = 0.4 -> flip delta ~0.1.
    flip_f2 = sens["flip_thresholds"]["f2"]
    assert flip_f2 is not None
    assert flip_f2 == pytest.approx(0.1, abs=0.02)

    # top_driver is the factor with the smallest finite flip threshold.
    finite = {k: v for k, v in sens["flip_thresholds"].items() if v is not None}
    assert sens["top_driver"] == min(finite, key=lambda k: finite[k])


def test_sensitivity_asymmetric_top_driver():
    # 2 options, 3 factors. f_swing has a large score gap favoring o2, so a small
    # weight change on it flips the winner; f_quiet barely matters.
    norm = {
        "o1": {"f_lead": 1.0, "f_swing": 0.0, "f_quiet": 0.5},
        "o2": {"f_lead": 0.0, "f_swing": 1.0, "f_quiet": 0.5},
    }
    # o1 currently wins on the back of f_lead.
    weights = {"f_lead": 0.5, "f_swing": 0.4, "f_quiet": 0.1}
    sens = compute.sensitivity(weights, norm)
    assert sens["ranking"][0] == "o1"
    flips = sens["flip_thresholds"]
    # f_quiet is identical across options -> reweighting it can never flip.
    assert flips["f_quiet"] is None
    # f_swing and f_lead each have a finite flip threshold.
    assert flips["f_swing"] is not None
    assert flips["f_lead"] is not None
    finite = {k: v for k, v in flips.items() if v is not None}
    assert sens["top_driver"] == min(finite, key=lambda k: finite[k])


def test_sensitivity_robustness_close_when_small_flip():
    norm = {
        "o1": {"f1": 1.0, "f2": 0.0},
        "o2": {"f1": 0.0, "f2": 1.0},
    }
    # Near-tie weights -> tiny margin and tiny flip threshold -> "close".
    weights = {"f1": 0.51, "f2": 0.49}
    sens = compute.sensitivity(weights, norm)
    assert sens["robustness"] == "close"


def test_sensitivity_stable_when_dominant():
    # o1 strictly dominates o2 on every factor -> no weight change can flip it.
    norm = {
        "o1": {"f1": 0.9, "f2": 0.9},
        "o2": {"f1": 0.1, "f2": 0.1},
    }
    weights = {"f1": 0.5, "f2": 0.5}
    sens = compute.sensitivity(weights, norm)
    assert sens["ranking"][0] == "o1"
    # No reweighting flips a dominated option -> all thresholds None.
    assert all(v is None for v in sens["flip_thresholds"].values())
    # margin is large here.
    assert sens["margin"] == pytest.approx(0.8)
    assert sens["robustness"] == "stable"
    # top_driver falls back to the largest-weight factor (here a tie -> some factor).
    assert sens["top_driver"] in {"f1", "f2"}


def test_sensitivity_single_option():
    norm = {"o1": {"f1": 0.5, "f2": 0.7}}
    weights = {"f1": 0.5, "f2": 0.5}
    sens = compute.sensitivity(weights, norm)
    assert sens["ranking"] == ["o1"]
    assert sens["margin"] == pytest.approx(0.0)
    assert all(v is None for v in sens["flip_thresholds"].values())
    assert sens["robustness"] == "stable"


def test_sensitivity_ranking_tiebreak_deterministic():
    # Identical utilities -> tie-break by option_id ascending.
    norm = {
        "b_opt": {"f1": 0.5},
        "a_opt": {"f1": 0.5},
    }
    weights = {"f1": 1.0}
    sens = compute.sensitivity(weights, norm)
    assert sens["ranking"] == ["a_opt", "b_opt"]


# --------------------------------------------------------------------------- #
# select_next_question
# --------------------------------------------------------------------------- #

def test_select_returns_candidate_from_provided_lists():
    current = {"f1": 0.34, "f2": 0.33, "f3": 0.33}
    norm = {
        "o1": {"f1": 0.8, "f2": 0.2, "f3": 0.5},
        "o2": {"f1": 0.2, "f2": 0.8, "f3": 0.5},
    }
    candidate_pairs = [("f1", "f2"), ("f1", "f3")]
    candidate_indicators = [{"indicator_id": "ind1", "factor_id": "f3"}]
    sel = compute.select_next_question(
        prior=current,
        current=current,
        comparisons=[],
        candidate_pairs=candidate_pairs,
        candidate_indicators=candidate_indicators,
        norm_scores=norm,
        ensemble_size=30,
        seed=0,
    )
    assert sel is not None
    if sel["kind"] == "weight_pairwise":
        assert tuple(sel["pair"]) in {("f1", "f2"), ("f1", "f3")}
    else:
        assert sel["kind"] == "indicator"
        assert sel["indicator_id"] == "ind1"
        assert sel["factor_id"] == "f3"


def test_select_deterministic_for_fixed_seed():
    current = {"f1": 0.4, "f2": 0.35, "f3": 0.25}
    norm = {
        "o1": {"f1": 0.9, "f2": 0.1, "f3": 0.5},
        "o2": {"f1": 0.1, "f2": 0.9, "f3": 0.5},
    }
    pairs = [("f1", "f2"), ("f2", "f3"), ("f1", "f3")]
    inds = [{"indicator_id": "ind1", "factor_id": "f3"}]
    kwargs = dict(
        prior=current,
        current=current,
        comparisons=[],
        candidate_pairs=pairs,
        candidate_indicators=inds,
        norm_scores=norm,
        ensemble_size=30,
        seed=7,
    )
    a = compute.select_next_question(**kwargs)
    b = compute.select_next_question(**kwargs)
    assert a == b


def test_select_none_when_empty_candidates():
    current = {"f1": 0.5, "f2": 0.5}
    sel = compute.select_next_question(
        prior=current,
        current=current,
        comparisons=[],
        candidate_pairs=[],
        candidate_indicators=[],
        norm_scores={"o1": {"f1": 0.5, "f2": 0.5}},
        ensemble_size=30,
        seed=0,
    )
    assert sel is None


def test_select_none_when_ensemble_fully_agrees_tiny_disagreement():
    # One option dominates on all factors -> every ensemble member ranks o1 #1.
    # Two near-identical factors -> pairwise disagreement is split but scaled by
    # weight; we make weights tiny on the candidate pair so the score stays
    # below the 0.05 threshold while the committee fully agrees on the winner.
    current = {"big": 0.98, "tiny_a": 0.01, "tiny_b": 0.01}
    norm = {
        "o1": {"big": 1.0, "tiny_a": 1.0, "tiny_b": 1.0},
        "o2": {"big": 0.0, "tiny_a": 0.0, "tiny_b": 0.0},
    }
    # Only offer the two negligible-weight factors as the trade-off candidate.
    sel = compute.select_next_question(
        prior=current,
        current=current,
        comparisons=[],
        candidate_pairs=[("tiny_a", "tiny_b")],
        candidate_indicators=[],
        norm_scores=norm,
        ensemble_size=30,
        seed=0,
    )
    assert sel is None


def test_select_ignores_candidates_with_unknown_factors():
    current = {"f1": 0.5, "f2": 0.5}
    norm = {"o1": {"f1": 0.6, "f2": 0.4}, "o2": {"f1": 0.4, "f2": 0.6}}
    # The only pair references a factor not in `current` -> must be skipped,
    # leaving the valid indicator as the choice.
    sel = compute.select_next_question(
        prior=current,
        current=current,
        comparisons=[],
        candidate_pairs=[("f1", "ghost")],
        candidate_indicators=[{"indicator_id": "ind1", "factor_id": "f1"}],
        norm_scores=norm,
        ensemble_size=30,
        seed=1,
    )
    assert sel is not None
    assert sel["kind"] == "indicator"
    assert sel["indicator_id"] == "ind1"


# --------------------------------------------------------------------------- #
# detect_conflict
# --------------------------------------------------------------------------- #

def test_detect_conflict_up_and_direction():
    prior = {"a": 0.2, "b": 0.5, "c": 0.3}
    current = {"a": 0.5, "b": 0.45, "c": 0.05}  # a +0.3, b -0.05, c -0.25
    out = compute.detect_conflict(prior, current, threshold=0.15)
    factor_ids = [c["factor_id"] for c in out]
    assert "a" in factor_ids
    assert "c" in factor_ids
    assert "b" not in factor_ids  # below threshold

    by_id = {c["factor_id"]: c for c in out}
    assert by_id["a"]["direction"] == "up"
    assert by_id["c"]["direction"] == "down"
    assert by_id["a"]["prior"] == pytest.approx(0.2)
    assert by_id["a"]["current"] == pytest.approx(0.5)


def test_detect_conflict_sorted_by_abs_diff_desc():
    prior = {"a": 0.1, "b": 0.1}
    current = {"a": 0.45, "b": 0.30}  # a +0.35, b +0.20
    out = compute.detect_conflict(prior, current, threshold=0.15)
    assert [c["factor_id"] for c in out] == ["a", "b"]


def test_detect_conflict_empty_when_within_threshold():
    prior = {"a": 0.5, "b": 0.5}
    current = {"a": 0.55, "b": 0.45}
    assert compute.detect_conflict(prior, current, threshold=0.15) == []


# --------------------------------------------------------------------------- #
# purity / determinism smoke checks
# --------------------------------------------------------------------------- #

def test_compute_does_not_import_backend_submodules():
    import backend.compute as c
    src_attrs = dir(c)
    # The pure core must not pull in store/llm/models/orchestrator.
    for forbidden in ("store", "llm", "models", "orchestrator"):
        assert forbidden not in src_attrs
