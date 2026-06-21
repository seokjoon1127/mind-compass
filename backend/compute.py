"""Pure-math compute core for the Decision Debugger.

Deterministic, side-effect-free numerical routines. This module MUST stay pure:
it depends only on numpy/scipy and the Python standard library, never imports
anything from ``backend`` (no models, no store, no llm, no I/O, no network).

Implements, per decision_debugger_design.md §5.5, §5.7, §5.8, §6.2, §8:

- normalize_scores : per-factor min-max scaling of desirability scores
- refine_weights   : Bradley-Terry logistic refinement with an L2 prior pull
- aggregate        : weighted-sum-model (WSM) utilities
- sensitivity      : one-at-a-time flip-threshold sensitivity analysis
- select_next_question : query-by-committee (QBC) question selection
- detect_conflict  : stated-vs-revealed weight conflicts
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize

# Numerical guards
_EPS = 1e-9
_PROB_LO = 1e-9
_PROB_HI = 1.0 - 1e-9


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable logistic sigmoid.
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _softmax(theta: np.ndarray) -> np.ndarray:
    if theta.size == 0:
        return theta.astype(float)
    z = theta - np.max(theta)
    e = np.exp(z)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full(theta.shape, 1.0 / theta.size, dtype=float)
    return e / s


# --------------------------------------------------------------------------- #
# 1) normalize_scores
# --------------------------------------------------------------------------- #

def normalize_scores(
    raw_scores: dict[str, dict[str, float]],
    directions: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Per-factor min-max scale option scores into [0, 1].

    ``raw_scores``: option_id -> {factor_id -> desirability in [0, 1]} where the
    LLM already encodes 1.0 = best. ``directions`` is INFORMATIONAL ONLY and is
    never used to invert values (the desirability convention already handles it).

    For every factor we min-max scale across options. If a factor's spread
    (max - min) is below 1e-9 we set all of its normalized values to 0.5 to avoid
    amplifying numerical noise. The returned dict preserves the input shape.
    """
    # `directions` is intentionally unused for the math (informational only).
    _ = directions

    option_ids = list(raw_scores.keys())
    if not option_ids:
        return {}

    # Collect the full set of factor ids appearing anywhere.
    factor_ids: list[str] = []
    seen: set[str] = set()
    for oid in option_ids:
        row = raw_scores.get(oid) or {}
        for fid in row:
            if fid not in seen:
                seen.add(fid)
                factor_ids.append(fid)

    out: dict[str, dict[str, float]] = {oid: {} for oid in option_ids}

    for fid in factor_ids:
        vals: list[float] = []
        present: list[str] = []
        for oid in option_ids:
            row = raw_scores.get(oid) or {}
            if fid in row:
                try:
                    vals.append(float(row[fid]))
                    present.append(oid)
                except (TypeError, ValueError):
                    continue
        if not present:
            continue
        arr = np.asarray(vals, dtype=float)
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        spread = vmax - vmin
        if spread < _EPS:
            for oid in present:
                out[oid][fid] = 0.5
        else:
            scaled = (arr - vmin) / spread
            for oid, v in zip(present, scaled):
                out[oid][fid] = float(v)

    return out


# --------------------------------------------------------------------------- #
# 2) refine_weights  (Bradley-Terry + L2 prior pull)
# --------------------------------------------------------------------------- #

def refine_weights(
    prior: dict[str, float],
    comparisons: list[dict[str, Any]],
    lam: float = 0.5,
) -> dict[str, float]:
    """Refine factor weights from soft pairwise comparisons (Bradley-Terry).

    Each factor f has a strength theta_f. For a comparison of (i, j),
    P(i > j) = sigmoid(theta_i - theta_j). We minimize::

        sum_c  weight_c * BCE(target_c, sigmoid(theta_i - theta_j))
        + lam * sum_f (theta_f - theta_prior_f)^2

    with theta_prior_f = log(max(prior_f, 1e-9)), x0 = theta_prior, via L-BFGS-B.
    Returns softmax(theta) as a normalized weight dict (>= 0, sums to 1).

    With no comparisons the prior is returned unchanged (within 1e-6).
    """
    factor_ids = list(prior.keys())
    n = len(factor_ids)
    if n == 0:
        return {}

    idx = {fid: k for k, fid in enumerate(factor_ids)}
    prior_arr = np.array([max(float(prior[fid]), _EPS) for fid in factor_ids], dtype=float)
    theta_prior = np.log(prior_arr)

    # Keep only comparisons whose both endpoints exist in `prior`.
    rows_i: list[int] = []
    rows_j: list[int] = []
    targets: list[float] = []
    sample_w: list[float] = []
    for c in comparisons or []:
        fi = c.get("i")
        fj = c.get("j")
        if fi not in idx or fj not in idx:
            continue
        try:
            t = float(c.get("target", 0.5))
            w = float(c.get("weight", 1.0))
        except (TypeError, ValueError):
            continue
        t = min(max(t, 0.0), 1.0)
        if w <= 0:
            continue
        rows_i.append(idx[fi])
        rows_j.append(idx[fj])
        targets.append(t)
        sample_w.append(w)

    # No usable comparisons -> return prior unchanged (renormalized to sum 1).
    if not rows_i:
        total = float(prior_arr.sum())
        return {fid: float(prior_arr[idx[fid]] / total) for fid in factor_ids}

    I = np.asarray(rows_i, dtype=int)
    J = np.asarray(rows_j, dtype=int)
    T = np.asarray(targets, dtype=float)
    W = np.asarray(sample_w, dtype=float)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        diff = theta[I] - theta[J]
        p = _sigmoid(diff)
        pc = np.clip(p, _PROB_LO, _PROB_HI)
        bce = -(T * np.log(pc) + (1.0 - T) * np.log(1.0 - pc))
        loss = float(np.sum(W * bce))
        reg = lam * float(np.sum((theta - theta_prior) ** 2))

        # Gradient. d/d(diff) of BCE(t, sigmoid(diff)) = (sigmoid(diff) - t).
        # Using p (unclipped except for stability) keeps the gradient consistent.
        g_diff = W * (p - T)
        grad = np.zeros_like(theta)
        np.add.at(grad, I, g_diff)
        np.add.at(grad, J, -g_diff)
        grad += 2.0 * lam * (theta - theta_prior)
        return loss + reg, grad

    res = minimize(
        objective,
        x0=theta_prior.copy(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
    )
    theta_hat = res.x if (res.x is not None and np.all(np.isfinite(res.x))) else theta_prior
    w = _softmax(np.asarray(theta_hat, dtype=float))
    return {fid: float(w[idx[fid]]) for fid in factor_ids}


# --------------------------------------------------------------------------- #
# 3) aggregate (WSM)
# --------------------------------------------------------------------------- #

def aggregate(
    weights: dict[str, float],
    norm_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Weighted-sum-model utility per option.

    U(o) = sum_f weights[f] * norm_scores[o][f]. Missing cells count as 0.
    """
    utilities: dict[str, float] = {}
    for oid, row in (norm_scores or {}).items():
        u = 0.0
        row = row or {}
        for fid, w in (weights or {}).items():
            try:
                s = float(row.get(fid, 0.0))
            except (TypeError, ValueError):
                s = 0.0
            u += float(w) * s
        utilities[oid] = float(u)
    return utilities


# --------------------------------------------------------------------------- #
# 4) sensitivity (OAT flip thresholds)
# --------------------------------------------------------------------------- #

def _ranking_from_utilities(utilities: dict[str, float]) -> list[str]:
    # Sort by utility descending, tie-break by option_id ascending for determinism.
    return sorted(utilities.keys(), key=lambda oid: (-utilities[oid], oid))


def _utilities_with_modified_weight(
    factor_ids: list[str],
    base_w: dict[str, float],
    option_ids: list[str],
    score_mat: np.ndarray,
    f_pos: int,
    new_wf: float,
) -> np.ndarray:
    """Recompute the utility vector when weights[factor f] is set to new_wf and
    the OTHER factors are renormalized proportionally so the total stays 1.

    Returns a numpy array of utilities aligned with ``option_ids``.
    """
    n = len(factor_ids)
    w = np.array([base_w.get(fid, 0.0) for fid in factor_ids], dtype=float)
    new_wf = float(min(max(new_wf, 0.0), 1.0))
    others_sum = float(w.sum() - w[f_pos])
    new_w = np.empty(n, dtype=float)
    new_w[f_pos] = new_wf
    remaining = 1.0 - new_wf
    if others_sum > _EPS:
        scale = remaining / others_sum
        for k in range(n):
            if k != f_pos:
                new_w[k] = w[k] * scale
    else:
        # No mass on other factors: spread the remainder evenly among them.
        if n > 1:
            share = remaining / (n - 1)
            for k in range(n):
                if k != f_pos:
                    new_w[k] = share
    return score_mat @ new_w


def _flip_threshold_for_factor(
    factor_ids: list[str],
    base_w: dict[str, float],
    option_ids: list[str],
    score_mat: np.ndarray,
    f_pos: int,
    current_top: str,
    steps: int = 200,
) -> Optional[float]:
    """Minimal |w_f_new - w_f_current| that changes the #1-ranked option.

    Sweeps w_f over a fine grid across [0, 1], detects the nearest grid point to
    the current value where argmax changes, then refines via bisection between
    that point and its neighbor on the no-flip side. Returns None if the top
    option never changes across [0, 1].
    """
    if len(option_ids) < 2:
        return None

    top_idx = option_ids.index(current_top)
    w_cur = float(base_w.get(factor_ids[f_pos], 0.0))
    w_cur = min(max(w_cur, 0.0), 1.0)

    def top_at(wf: float) -> int:
        u = _utilities_with_modified_weight(
            factor_ids, base_w, option_ids, score_mat, f_pos, wf
        )
        # argmax with deterministic tie-break by option_id ascending.
        best = int(np.argmax(u))
        umax = u[best]
        tied = [k for k in range(len(option_ids)) if u[k] >= umax - 1e-12]
        if len(tied) > 1:
            tied.sort(key=lambda k: option_ids[k])
            best = tied[0]
        return best

    grid = np.linspace(0.0, 1.0, steps + 1)
    # Order grid points by distance from the current weight so we find the
    # closest flipping weight first.
    order = sorted(range(grid.size), key=lambda k: abs(grid[k] - w_cur))

    best_delta: Optional[float] = None
    flip_wf: Optional[float] = None
    no_flip_wf = w_cur
    for k in order:
        wf = float(grid[k])
        if top_at(wf) != top_idx:
            flip_wf = wf
            best_delta = abs(wf - w_cur)
            break
        else:
            no_flip_wf = wf  # last seen non-flipping grid point (closest scan)

    if flip_wf is None:
        return None

    # Refine between a known no-flip weight and the flip weight via bisection.
    # Anchor the no-flip side at the current weight (guaranteed no-flip).
    lo, hi = w_cur, flip_wf  # lo: no flip, hi: flip
    if top_at(lo) != top_idx:
        # Current weight itself already flips relative to current_top: zero delta.
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if top_at(mid) != top_idx:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < 1e-7:
            break
    return float(abs(hi - w_cur))


def sensitivity(
    weights: dict[str, float],
    norm_scores: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """One-at-a-time sensitivity analysis with flip thresholds.

    Returns a dict with keys: utilities, ranking, flip_thresholds, top_driver,
    margin, robustness. Handles the degenerate single-option case.
    """
    weights = weights or {}
    norm_scores = norm_scores or {}

    utilities = aggregate(weights, norm_scores)
    ranking = _ranking_from_utilities(utilities)
    factor_ids = list(weights.keys())

    # Degenerate cases: 0 or 1 option.
    if len(ranking) <= 1:
        flips = {fid: None for fid in factor_ids}
        top_driver = None
        if factor_ids:
            top_driver = max(factor_ids, key=lambda f: weights.get(f, 0.0))
        return {
            "utilities": utilities,
            "ranking": ranking,
            "flip_thresholds": flips,
            "top_driver": top_driver,
            "margin": 0.0,
            "robustness": "stable",
        }

    option_ids = ranking[:]  # stable order; index() used internally
    current_top = ranking[0]
    margin = float(utilities[ranking[0]] - utilities[ranking[1]])

    # Build the option x factor score matrix aligned to option_ids / factor_ids.
    score_mat = np.zeros((len(option_ids), len(factor_ids)), dtype=float)
    for oi, oid in enumerate(option_ids):
        row = norm_scores.get(oid) or {}
        for fi, fid in enumerate(factor_ids):
            try:
                score_mat[oi, fi] = float(row.get(fid, 0.0))
            except (TypeError, ValueError):
                score_mat[oi, fi] = 0.0

    flips: dict[str, Optional[float]] = {}
    for fi, fid in enumerate(factor_ids):
        flips[fid] = _flip_threshold_for_factor(
            factor_ids, weights, option_ids, score_mat, fi, current_top
        )

    finite = {fid: v for fid, v in flips.items() if v is not None}
    if finite:
        top_driver = min(finite, key=lambda f: finite[f])
        smallest_flip = min(finite.values())
    else:
        top_driver = (
            max(factor_ids, key=lambda f: weights.get(f, 0.0)) if factor_ids else None
        )
        smallest_flip = None

    close = (margin < 0.05) or (smallest_flip is not None and smallest_flip < 0.1)
    robustness = "close" if close else "stable"

    return {
        "utilities": utilities,
        "ranking": ranking,
        "flip_thresholds": flips,
        "top_driver": top_driver,
        "margin": margin,
        "robustness": robustness,
    }


# --------------------------------------------------------------------------- #
# 5) select_next_question (QBC)
# --------------------------------------------------------------------------- #

def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def select_next_question(
    prior: dict[str, float],
    current: dict[str, float],
    comparisons: list[dict[str, Any]],
    candidate_pairs: list[tuple[str, str]],
    candidate_indicators: list[dict[str, Any]],
    norm_scores: dict[str, dict[str, float]],
    ensemble_size: int = 30,
    seed: int = 0,
) -> Optional[dict[str, Any]]:
    """Query-by-committee heuristic for choosing the next question.

    Builds an ensemble of ``ensemble_size`` weight vectors near ``current``
    (theta = log(clip(current)) + N(0, 0.5), w = softmax(theta)) using a
    deterministic rng seeded with ``seed``. Scores each candidate by ensemble
    disagreement and returns the max-disagreement candidate.

    Returns ``None`` if both candidate lists are empty, or if the whole ensemble
    agrees on the same top option AND the maximum disagreement is below 0.05.
    """
    # `prior` / `comparisons` are not needed for the QBC disagreement heuristic
    # but are part of the orchestrator's call contract.
    _ = prior, comparisons

    has_pairs = bool(candidate_pairs)
    has_inds = bool(candidate_indicators)
    if not has_pairs and not has_inds:
        return None

    factor_ids = list((current or {}).keys())
    if not factor_ids:
        return None
    idx = {fid: k for k, fid in enumerate(factor_ids)}

    rng = np.random.default_rng(seed)
    base = np.array(
        [max(float(current.get(fid, 0.0)), _EPS) for fid in factor_ids], dtype=float
    )
    log_base = np.log(base)

    m = max(int(ensemble_size), 1)
    n = len(factor_ids)
    noise = rng.normal(0.0, 0.5, size=(m, n))
    thetas = log_base[None, :] + noise
    # Row-wise softmax -> ensemble of weight vectors (m x n).
    z = thetas - thetas.max(axis=1, keepdims=True)
    e = np.exp(z)
    ensemble = e / e.sum(axis=1, keepdims=True)

    mean_w = ensemble.mean(axis=0)

    # Ensemble agreement on the top option (for the None short-circuit).
    same_top = True
    if norm_scores:
        option_ids = list(norm_scores.keys())
        if option_ids:
            score_mat = np.zeros((len(option_ids), n), dtype=float)
            for oi, oid in enumerate(option_ids):
                row = norm_scores.get(oid) or {}
                for fi, fid in enumerate(factor_ids):
                    try:
                        score_mat[oi, fi] = float(row.get(fid, 0.0))
                    except (TypeError, ValueError):
                        score_mat[oi, fi] = 0.0
            # utilities for every ensemble member: (m x options)
            member_utils = ensemble @ score_mat.T
            tops = np.argmax(member_utils, axis=1)
            same_top = bool(np.all(tops == tops[0]))
        else:
            same_top = True
    else:
        same_top = True

    best_score = -1.0
    best: Optional[dict[str, Any]] = None

    # Pairwise candidates: binary entropy of the fraction where w[fa] > w[fb],
    # scaled by (mean w[fa] + mean w[fb]) to break ties toward heavy trade-offs.
    for pair in candidate_pairs or []:
        try:
            fa, fb = pair[0], pair[1]
        except (TypeError, IndexError, ValueError):
            continue
        if fa not in idx or fb not in idx:
            continue
        a = ensemble[:, idx[fa]]
        b = ensemble[:, idx[fb]]
        frac = float(np.mean(a > b))
        disagreement = _binary_entropy(frac)
        # Scale by the combined weight mass so trade-offs between heavy factors
        # win ties and negligible-weight pairs score near zero.
        scale = float(mean_w[idx[fa]] + mean_w[idx[fb]])
        score = disagreement * scale
        if score > best_score:
            best_score = score
            best = {"kind": "weight_pairwise", "pair": (fa, fb)}

    # Indicator candidates: normalized std of w[f] across ensemble, scaled by
    # mean w[f].
    for ind in candidate_indicators or []:
        fid = ind.get("factor_id")
        ind_id = ind.get("indicator_id")
        if fid not in idx:
            continue
        col = ensemble[:, idx[fid]]
        std = float(np.std(col))
        mw = float(mean_w[idx[fid]])
        # Coefficient-of-variation style disagreement, bounded into ~[0,1].
        norm_var = std / (mw + _EPS)
        disagreement = float(min(norm_var, 1.0))
        score = disagreement * mw
        if score > best_score:
            best_score = score
            best = {"kind": "indicator", "indicator_id": ind_id, "factor_id": fid}

    if best is None:
        return None

    # Stop only when the committee fully agrees on the winner AND nothing left is
    # ambiguous enough to matter. best_score >= the raw disagreement of the chosen
    # candidate (it is the disagreement scaled up by 1 + weight mass), so the
    # threshold is crossed only when the disagreement is genuinely tiny.
    if same_top and best_score < 0.05:
        return None

    return best


# --------------------------------------------------------------------------- #
# 6) detect_conflict
# --------------------------------------------------------------------------- #

def detect_conflict(
    prior: dict[str, float],
    current: dict[str, float],
    threshold: float = 0.15,
) -> list[dict[str, Any]]:
    """Flag factors where the refined weight diverges from the prior.

    For each factor, if |current[f] - prior[f]| > threshold, emit
    {factor_id, prior, current, direction}. Sorted by |difference| descending.
    """
    prior = prior or {}
    current = current or {}
    conflicts: list[dict[str, Any]] = []
    for fid in prior:
        if fid not in current:
            continue
        try:
            p = float(prior[fid])
            c = float(current[fid])
        except (TypeError, ValueError):
            continue
        diff = c - p
        if abs(diff) > threshold:
            conflicts.append({
                "factor_id": fid,
                "prior": p,
                "current": c,
                "direction": "up" if diff > 0 else "down",
            })
    conflicts.sort(key=lambda d: abs(d["current"] - d["prior"]), reverse=True)
    return conflicts
