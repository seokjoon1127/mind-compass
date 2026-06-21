"""Session state machine: SETUP -> ANALYZE -> QUESTIONING -> RESULT.

Wires the LLM layer (speaks in human-readable names) to the typed models
(keyed by id) and the pure Compute core.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from . import compute, llm, store
from .models import (
    Answer,
    DecisionSession,
    Factor,
    Indicator,
    Option,
    Question,
    Result,
    Score,
    State,
    WeightState,
    new_id,
)

# --- tuning constants ---
NUM_QUESTIONS = 5     # fixed number of MAIN questions asked — no early stopping
INDICATOR_WEIGHT = 0.4
LAMBDA = 0.5          # L2 pull of refined weights toward the LLM prior
MAX_DECOMP_DEPTH = 4  # §6.3: how deep recursive sub-questions may go before giving up


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #

def create_session(context: str) -> DecisionSession:
    session = DecisionSession(id=new_id("sess"), context=context, status=State.SETUP)
    store.save(session)
    analyze(session)
    return session


def analyze(session: DecisionSession) -> None:
    """Run all autonomous LLM decisions and prepare the question queue."""
    session.status = State.ANALYZE

    # 1) options + summary
    extracted = llm.extract_options(session.context)
    session.context_summary = extracted["context_summary"] or session.context
    options: list[Option] = []
    for o in extracted["options"]:
        options.append(Option(id=new_id("opt"), name=str(o.get("name", "")).strip(),
                              description=str(o.get("description", "")).strip()))
    options = [o for o in options if o.name]
    if len(options) < 2:
        raise ValueError("Could not identify at least two distinct options from the input.")
    session.options = options

    # 2) factors + initial importance
    raw_factors = llm.decide_factors(
        [{"name": o.name, "description": o.description} for o in options],
        session.context_summary,
    )
    factors: list[Factor] = []
    seen_names: set[str] = set()
    for f in raw_factors:
        name = str(f.get("name", "")).strip()
        if not name or name.casefold() in seen_names:
            continue
        seen_names.add(name.casefold())
        ftype = f.get("type", "concrete")
        ftype = ftype if ftype in ("concrete", "abstract") else "concrete"
        direction = f.get("direction", "higher_better")
        direction = direction if direction in ("higher_better", "lower_better") else "higher_better"
        try:
            importance = float(f.get("importance", 1.0))
        except (TypeError, ValueError):
            importance = 1.0
        factors.append(Factor(
            id=new_id("fac"), name=name,
            description=str(f.get("description", "")).strip(),
            relevance=str(f.get("relevance", "")).strip(),
            type=ftype, direction=direction, importance=max(importance, 1e-6),
        ))
    if len(factors) < 2:
        raise ValueError("Could not identify at least two decision factors.")
    session.factors = factors

    name_to_factor = {f.name.casefold(): f for f in factors}

    # (3 — removed) Indicator / yes-no questions are no longer used: every question is a
    # two-card comparison. Hard-to-decide pairs are handled by recursive decomposition (§6.3).

    # 4) pairwise verbalizations (batched)
    raw_pairs = llm.verbalize_pairs(
        [{"name": f.name} for f in factors], session.context_summary,
    )
    for p in raw_pairs:
        fa = name_to_factor.get(str(p.get("factor_a", "")).strip().casefold())
        fb = name_to_factor.get(str(p.get("factor_b", "")).strip().casefold())
        if not fa or not fb or fa.id == fb.id:
            continue
        session.pairwise_verbalizations[_pair_key(fa.id, fb.id)] = {
            "question": str(p.get("question", "")).strip(),
            "example_a": str(p.get("example_a", "")).strip(),
            "example_b": str(p.get("example_b", "")).strip(),
            "first": fa.id,  # which factor the verbalization treated as "a"
        }

    # 5) option scores
    raw_scores = llm.score_options(
        [{"name": o.name, "description": o.description} for o in options],
        [{"name": f.name, "description": f.description, "type": f.type, "direction": f.direction}
         for f in factors],
        session.context_summary,
    )
    opt_by_name = {o.name.casefold(): o for o in options}
    matrix: dict[str, dict[str, Score]] = {o.id: {} for o in options}
    for oname, fac_scores in (raw_scores or {}).items():
        opt = opt_by_name.get(str(oname).strip().casefold())
        if not opt or not isinstance(fac_scores, dict):
            continue
        for fname, sc in fac_scores.items():
            fac = name_to_factor.get(str(fname).strip().casefold())
            if not fac:
                continue
            matrix[opt.id][fac.id] = _coerce_score(sc)
    # fill any missing cells defensively
    for o in options:
        for f in factors:
            matrix[o.id].setdefault(f.id, Score(value=0.5, confidence=0.3, rationale=""))
    session.score_matrix = matrix

    # normalize (per-factor min-max; direction is informational only — scores are desirability)
    raw_values = {oid: {fid: matrix[oid][fid].value for fid in matrix[oid]} for oid in matrix}
    directions = {f.id: f.direction for f in factors}
    session.norm_scores = compute.normalize_scores(raw_values, directions)

    # weights: prior from importance, current = prior initially
    total_imp = sum(f.importance for f in factors) or 1.0
    prior = {f.id: f.importance / total_imp for f in factors}
    session.weights = WeightState(prior=prior, current=dict(prior), conflict=[])

    # build question queue: every factor pair (all questions are two-card comparisons)
    questions: list[Question] = []
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            questions.append(Question(
                id=new_id("q"), kind="weight_pairwise",
                payload={"factor_a_id": factors[i].id, "factor_b_id": factors[j].id},
            ))
    session.questions = questions

    # seed rank history
    session.rank_history = [compute.aggregate(session.weights.current, session.norm_scores)]
    session.status = State.QUESTIONING
    store.save(session)


def next_question(session: DecisionSession) -> Optional[Question]:
    """Pick the next question to ask, or None when we should finalize."""
    if session.status == State.RESULT:
        return None

    # If we are mid-decomposition (resolving a "잘 모르겠어요"), keep serving the
    # sub-question chain until it rolls up to an answer for the original question.
    if session.active_decomposition:
        root_id = session.active_decomposition.get("root_id")
        sub = next((q for q in session.questions
                    if q.kind == "sub_question"
                    and q.payload.get("root_id") == root_id
                    and q.status == "pending"), None)
        return sub if sub is not None else _next_subquestion(session)

    answered = _main_answered(session)            # sub-questions do NOT count
    pending = [q for q in session.questions
               if q.status == "pending" and q.kind == "weight_pairwise"]

    # Fixed-length interview: ask exactly NUM_QUESTIONS main questions (or until we run
    # out). No early stopping — the user asked to always answer the full set.
    if not pending or answered >= NUM_QUESTIONS:
        finalize(session)
        return None

    comparisons = _build_comparisons(session)
    candidate_pairs = [
        (q.payload["factor_a_id"], q.payload["factor_b_id"]) for q in pending
    ]

    selection = compute.select_next_question(
        prior=session.weights.prior,
        current=session.weights.current,
        comparisons=comparisons,
        candidate_pairs=candidate_pairs,
        candidate_indicators=[],
        norm_scores=session.norm_scores,
        ensemble_size=30,
        seed=answered,  # deterministic per turn
    )

    # QBC only chooses WHICH question is most informative; it never ends the interview
    # early now. If it has no preference, just take the next pending question.
    chosen = _match_question(pending, selection) if selection else None
    if chosen is None:
        chosen = pending[0]
    store.save(session)
    return chosen


def submit_answer(session: DecisionSession, question_id: str, value: Any) -> Question | None:
    """Record an answer (or sub-answer), refine weights, return the next question."""
    q = session.question(question_id)
    if q is None:
        raise KeyError("Unknown question id")

    v = str(value).strip().lower()

    # A sub-question answer drives the recursive decomposition (§6.3).
    if q.kind == "sub_question":
        return _handle_sub_answer(session, q, value)

    if q.status == "answered":
        return next_question(session)  # idempotent-ish

    # "잘 모르겠어요" on a main trade-off starts recursive decomposition instead of skipping.
    if q.kind == "weight_pairwise" and v == "unknown":
        return _start_decomposition(session, q)

    # Normal main answer (pairwise pick / "비슷", or indicator).
    session.answers.append(Answer(question_id=question_id, value=value, timestamp=time.time()))
    q.status = "answered"
    _recompute(session)
    store.save(session)
    return next_question(session)


def _recompute(session: DecisionSession) -> None:
    """Refine weights from all answers, detect conflict, append a utility snapshot."""
    comparisons = _build_comparisons(session)
    session.weights.current = compute.refine_weights(
        prior=session.weights.prior, comparisons=comparisons, lam=LAMBDA,
    )
    session.weights.conflict = compute.detect_conflict(
        session.weights.prior, session.weights.current,
    )
    session.rank_history.append(
        compute.aggregate(session.weights.current, session.norm_scores)
    )


# --------------------------------------------------------------------------- #
# Recursive decomposition (§6.3): hard trade-off -> easier sub-questions -> roll-up
# --------------------------------------------------------------------------- #

def _start_decomposition(session: DecisionSession, root_q: Question) -> Question:
    session.active_decomposition = {
        "root_id": root_q.id,
        "factor_a_id": root_q.payload["factor_a_id"],
        "factor_b_id": root_q.payload["factor_b_id"],
        "depth": 0,
    }
    return _next_subquestion(session)


def _next_subquestion(session: DecisionSession) -> Question:
    dec = session.active_decomposition
    dec["depth"] = int(dec.get("depth", 0)) + 1
    fa = session.factor(dec["factor_a_id"])
    fb = session.factor(dec["factor_b_id"])
    asked = [
        qq.payload.get("question_text", "")
        for qq in session.questions
        if qq.kind == "sub_question" and qq.payload.get("root_id") == dec["root_id"]
    ]
    data = llm.decompose_question(
        {"name": fa.name, "description": fa.description},
        {"name": fb.name, "description": fb.description},
        session.context_summary, asked, dec["depth"],
    )
    options = data.get("options") or []
    if not options:  # defensive fallback so the user always has something to pick
        options = [
            {"label": fa.name, "example": "", "favors": "a", "strength": 0.7},
            {"label": fb.name, "example": "", "favors": "b", "strength": 0.7},
        ]
    subq = Question(
        id=new_id("q"), kind="sub_question", parent_id=dec["root_id"],
        payload={
            "root_id": dec["root_id"],
            "factor_a_id": fa.id, "factor_b_id": fb.id,
            "depth": dec["depth"],
            "question_text": data.get("question", f"{fa.name} vs {fb.name}"),
            "options": options,
        },
    )
    session.questions.append(subq)
    store.save(session)
    return subq


def _handle_sub_answer(session: DecisionSession, subq: Question, value: Any) -> Optional[Question]:
    dec = session.active_decomposition
    if dec is None:  # stale sub-question (already resolved) — just advance
        subq.status = "answered"
        return next_question(session)

    v = str(value).strip().lower()
    session.answers.append(Answer(question_id=subq.id, value=value, timestamp=time.time()))
    subq.status = "answered"

    if v == "unknown":
        # still can't decide -> go one level deeper, unless we've hit the limit.
        if dec["depth"] >= MAX_DECOMP_DEPTH:
            return _resolve_decomposition(session, target=0.5, weight=0.3, gave_up=True)
        return _next_subquestion(session)
    if v == "similar":
        return _resolve_decomposition(session, target=0.5, weight=0.6)

    opt = _find_option(subq, value)
    if opt is None:
        if dec["depth"] >= MAX_DECOMP_DEPTH:
            return _resolve_decomposition(session, target=0.5, weight=0.3, gave_up=True)
        return _next_subquestion(session)

    favors = str(opt.get("favors", "neither")).strip().lower()
    try:
        strength = _clamp01(float(opt.get("strength", 0.6)))
    except (TypeError, ValueError):
        strength = 0.6
    if favors == "a":
        target = 0.5 + 0.5 * strength
    elif favors == "b":
        target = 0.5 - 0.5 * strength
    else:
        target = 0.5
    return _resolve_decomposition(session, target=target, weight=1.0)


def _resolve_decomposition(
    session: DecisionSession, target: float, weight: float, gave_up: bool = False,
) -> Optional[Question]:
    """Roll the sub-answers up into a single (A vs B) comparison for the root question."""
    dec = session.active_decomposition
    root = session.question(dec["root_id"]) if dec else None
    if root is not None:
        session.answers.append(Answer(
            question_id=root.id,
            value={"rolled_up": True, "target": float(target), "weight": float(weight),
                   "gave_up": bool(gave_up)},
            timestamp=time.time(),
        ))
        root.status = "answered"
    session.active_decomposition = None
    _recompute(session)
    store.save(session)
    return next_question(session)


def _find_option(subq: Question, value: Any) -> Optional[dict]:
    val = str(value).strip()
    opts = subq.payload.get("options", []) or []
    for opt in opts:
        if str(opt.get("label", "")).strip() == val:
            return opt
    low = val.lower()
    for opt in opts:
        if str(opt.get("label", "")).strip().lower() == low:
            return opt
    return None


def finalize(session: DecisionSession) -> Result:
    if session.result is not None and session.status == State.RESULT:
        return session.result

    weights = session.weights.current or session.weights.prior
    sens = compute.sensitivity(weights, session.norm_scores)
    conflict = compute.detect_conflict(session.weights.prior, weights)

    id_to_oname = {o.id: o.name for o in session.options}
    id_to_fname = {f.id: f.name for f in session.factors}

    ranking_ids: list[str] = sens.get("ranking", [])
    utilities = sens.get("utilities", {})

    # assemble report input in human-readable names. decision_summary is included so the
    # LLM matches the user's language (the report call has no other access to the original text).
    report_input = {
        "decision_summary": session.context_summary,
        "options_ranked": [
            {"option": id_to_oname.get(oid, oid), "utility": round(utilities.get(oid, 0.0), 4)}
            for oid in ranking_ids
        ],
        "weights": sorted(
            ({"factor": id_to_fname.get(fid, fid), "weight": round(w, 4)}
             for fid, w in weights.items()),
            key=lambda d: -d["weight"],
        ),
        "top_driver": id_to_fname.get(sens.get("top_driver"), sens.get("top_driver")),
        "flip_thresholds": {
            id_to_fname.get(fid, fid): (None if v is None else round(v, 4))
            for fid, v in sens.get("flip_thresholds", {}).items()
        },
        "robustness": sens.get("robustness", "stable"),
        "margin": round(sens.get("margin", 0.0), 4),
        "score_rationales": _top_rationales(session),
        "conflicts": [
            {
                "factor": id_to_fname.get(c["factor_id"], c["factor_id"]),
                "stated_weight": round(c["prior"], 4),
                "revealed_weight": round(c["current"], 4),
                "direction": c["direction"],
            }
            for c in conflict
        ],
    }
    report = llm.generate_report(report_input)

    session.result = Result(
        utilities=utilities,
        ranking=ranking_ids,
        sensitivity=sens,
        robustness=sens.get("robustness", "stable"),
        report=report["report"],
        next_info=report["next_info"],
        conflict=conflict,
    )
    session.status = State.RESULT
    store.save(session)
    return session.result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _pair_key(fa_id: str, fb_id: str) -> str:
    return "||".join(sorted([fa_id, fb_id]))


def _coerce_score(sc: Any) -> Score:
    if not isinstance(sc, dict):
        try:
            return Score(value=_clamp01(float(sc)))
        except (TypeError, ValueError):
            return Score(value=0.5, confidence=0.3)
    try:
        value = _clamp01(float(sc.get("value", 0.5)))
    except (TypeError, ValueError):
        value = 0.5
    rng = sc.get("range")
    if isinstance(rng, list) and len(rng) == 2:
        try:
            rng = [_clamp01(float(rng[0])), _clamp01(float(rng[1]))]
        except (TypeError, ValueError):
            rng = None
    else:
        rng = None
    conf = sc.get("confidence")
    try:
        conf = _clamp01(float(conf)) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    return Score(value=value, range=rng, confidence=conf,
                 rationale=str(sc.get("rationale", "")).strip())


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _build_comparisons(session: DecisionSession) -> list[dict]:
    """Convert all answers into BT comparisons over factor importance."""
    comparisons: list[dict] = []
    indicator_signals: dict[str, list[float]] = {}

    answered = {a.question_id: a.value for a in session.answers}
    for q in session.questions:
        if q.status != "answered":
            continue
        value = answered.get(q.id)
        if q.kind == "weight_pairwise":
            fa, fb = q.payload["factor_a_id"], q.payload["factor_b_id"]
            # resolved via recursive decomposition (§6.3 roll-up)
            if isinstance(value, dict) and value.get("rolled_up"):
                w = float(value.get("weight", 1.0))
                if w > 0:
                    comparisons.append({
                        "i": fa, "j": fb,
                        "target": _clamp01(float(value.get("target", 0.5))), "weight": w,
                    })
                continue
            v = str(value).strip().lower()
            if v == "a":
                comparisons.append({"i": fa, "j": fb, "target": 1.0, "weight": 1.0})
            elif v == "b":
                comparisons.append({"i": fa, "j": fb, "target": 0.0, "weight": 1.0})
            elif v == "similar":
                comparisons.append({"i": fa, "j": fb, "target": 0.5, "weight": 0.5})
            # "unknown" -> contribute nothing (root is being decomposed)
        elif q.kind == "indicator":
            ind = _find_indicator(session, q.payload["indicator_id"])
            if ind is None:
                continue
            sig = _signal_from_answer(ind, value)
            if sig is not None:
                indicator_signals.setdefault(q.payload["factor_id"], []).append(sig)

    # fold indicator signals into soft comparisons vs every other factor
    factor_ids = [f.id for f in session.factors]
    for fid, sigs in indicator_signals.items():
        s = sum(sigs) / len(sigs)
        for other in factor_ids:
            if other == fid:
                continue
            comparisons.append({"i": fid, "j": other, "target": s, "weight": INDICATOR_WEIGHT})
    return comparisons


def _find_indicator(session: DecisionSession, indicator_id: str) -> Optional[Indicator]:
    for f in session.factors:
        for ind in f.indicators:
            if ind.id == indicator_id:
                return ind
    return None


def _signal_from_answer(ind: Indicator, value: Any) -> Optional[float]:
    m = ind.mapping or {}
    try:
        if ind.answer_type == "binary":
            v = str(value).strip().lower()
            yes = v in ("yes", "y", "true", "1", "네", "예", "응")
            no = v in ("no", "n", "false", "0", "아니", "아니오", "아뇨")
            if yes:
                return _clamp01(float(m.get("yes", 0.7)))
            if no:
                return _clamp01(float(m.get("no", 0.3)))
            return None
        if ind.answer_type == "scale":
            v = float(value)
            lo, hi = float(m.get("min", 1)), float(m.get("max", 5))
            ls, hs = float(m.get("min_signal", 0.2)), float(m.get("max_signal", 0.9))
            return _clamp01(_lerp(v, lo, hi, ls, hs))
        if ind.answer_type == "count":
            v = float(value)
            lo, hi = float(m.get("low", 0)), float(m.get("high", 10))
            ls, hs = float(m.get("low_signal", 0.2)), float(m.get("high_signal", 0.9))
            return _clamp01(_lerp(v, lo, hi, ls, hs))
        if ind.answer_type == "choice":
            v = str(value).strip()
            for ch in m.get("choices", []):
                if str(ch.get("label", "")).strip() == v:
                    return _clamp01(float(ch.get("signal", 0.5)))
            return None
    except (TypeError, ValueError):
        return None
    return None


def _lerp(v: float, lo: float, hi: float, slo: float, shi: float) -> float:
    if hi == lo:
        return (slo + shi) / 2.0
    t = (v - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return slo + t * (shi - slo)


def _main_answered(session: DecisionSession) -> int:
    """Answered MAIN questions (pairwise/indicator). Sub-questions don't count toward
    the question budget — they're part of resolving one main question."""
    return sum(1 for q in session.questions
               if q.kind in ("weight_pairwise", "indicator") and q.status == "answered")


def _match_question(pending: list[Question], selection: dict) -> Optional[Question]:
    if not selection:
        return None
    if selection.get("kind") == "weight_pairwise":
        pair = set(selection.get("pair", []))
        for q in pending:
            if q.kind == "weight_pairwise" and {q.payload["factor_a_id"], q.payload["factor_b_id"]} == pair:
                return q
    elif selection.get("kind") == "indicator":
        ind_id = selection.get("indicator_id")
        for q in pending:
            if q.kind == "indicator" and q.payload["indicator_id"] == ind_id:
                return q
    return None


def _top_rationales(session: DecisionSession, per_factor: int = 1) -> list[dict]:
    """A few representative score rationales for the report (highest-weight factors)."""
    weights = session.weights.current or session.weights.prior
    top_factor_ids = sorted(weights, key=lambda fid: -weights.get(fid, 0))[:3]
    id_to_oname = {o.id: o.name for o in session.options}
    id_to_fname = {f.id: f.name for f in session.factors}
    out: list[dict] = []
    for fid in top_factor_ids:
        for oid in session.score_matrix:
            sc = session.score_matrix[oid].get(fid)
            if sc and sc.rationale:
                out.append({
                    "factor": id_to_fname.get(fid, fid),
                    "option": id_to_oname.get(oid, oid),
                    "value": round(sc.value, 3),
                    "rationale": sc.rationale,
                })
    return out


# --------------------------------------------------------------------------- #
# Public view serializers (for the API / frontend)
# --------------------------------------------------------------------------- #

def question_view(session: DecisionSession, q: Question) -> dict:
    answered = _main_answered(session)
    progress = {"answered": answered, "max": NUM_QUESTIONS}

    if q.kind == "sub_question":
        return {
            "id": q.id,
            "kind": "sub_question",
            "question": q.payload.get("question_text", ""),
            "cards": [
                {"name": str(o.get("label", "")), "example": str(o.get("example", "")),
                 "value": str(o.get("label", ""))}
                for o in (q.payload.get("options", []) or [])[:2]
            ],
            "depth": q.payload.get("depth", 1),
            "decomposing": True,
            "progress": progress,
        }

    if q.kind == "weight_pairwise":
        fa = session.factor(q.payload["factor_a_id"])
        fb = session.factor(q.payload["factor_b_id"])
        verb = session.pairwise_verbalizations.get(_pair_key(fa.id, fb.id), {})
        # Align example/question orientation with (fa, fb)
        first = verb.get("first")
        flipped = first is not None and first == fb.id
        ex_a, ex_b = verb.get("example_a", ""), verb.get("example_b", "")
        if flipped:
            ex_a, ex_b = ex_b, ex_a
        # The stored prose names factors in the LLM's original a/b order. If that is
        # reversed vs the displayed cards, fall back to a neutral template so the
        # question can't read backwards from the A/B buttons.
        question = verb.get("question")
        if flipped or not question:
            question = f"{fa.name} vs {fb.name} — 무엇이 더 중요한가요?"
        return {
            "id": q.id,
            "kind": "weight_pairwise",
            "question": question,
            "factor_a": {"id": fa.id, "name": fa.name, "description": fa.description},
            "factor_b": {"id": fb.id, "name": fb.name, "description": fb.description},
            "example_a": ex_a,
            "example_b": ex_b,
            "progress": progress,
        }
    # indicator
    ind = _find_indicator(session, q.payload["indicator_id"])
    fac = session.factor(q.payload["factor_id"])
    view: dict[str, Any] = {
        "id": q.id,
        "kind": "indicator",
        "question": ind.question if ind else "",
        "answer_type": ind.answer_type if ind else "binary",
        "factor": {"id": fac.id, "name": fac.name} if fac else None,
        "progress": progress,
    }
    if ind and ind.answer_type == "choice":
        view["choices"] = [
            {"value": str(c.get("label", "")), "label": str(c.get("label", ""))}
            for c in (ind.mapping.get("choices", []) or [])
        ]
    elif ind and ind.answer_type == "scale":
        view["scale"] = {"min": int(ind.mapping.get("min", 1)), "max": int(ind.mapping.get("max", 5))}
    return view


def result_view(session: DecisionSession) -> dict:
    if session.result is None:
        finalize(session)
    res = session.result
    id_to_oname = {o.id: o.name for o in session.options}
    id_to_fname = {f.id: f.name for f in session.factors}
    weights = session.weights.current or session.weights.prior

    ranking = [
        {"option_id": oid, "name": id_to_oname.get(oid, oid),
         "utility": round(res.utilities.get(oid, 0.0), 4)}
        for oid in res.ranking
    ]
    flips = res.sensitivity.get("flip_thresholds", {})
    drivers = sorted(
        (
            {
                "factor": id_to_fname.get(fid, fid),
                "weight": round(weights.get(fid, 0.0), 4),
                "flip_threshold": (None if flips.get(fid) is None else round(flips[fid], 4)),
            }
            for fid in weights
        ),
        key=lambda d: -d["weight"],
    )
    return {
        "status": session.status.value,
        "ranking": ranking,
        "winner": ranking[0] if ranking else None,
        "robustness": res.robustness,
        "margin": round(res.sensitivity.get("margin", 0.0), 4),
        "top_driver": id_to_fname.get(res.sensitivity.get("top_driver"),
                                      res.sensitivity.get("top_driver")),
        "report": res.report,
        "next_info": res.next_info,
        "drivers": drivers,
        "conflict": [
            {
                "factor": id_to_fname.get(c["factor_id"], c["factor_id"]),
                "stated": round(c["prior"], 3),
                "revealed": round(c["current"], 3),
                "direction": c["direction"],
            }
            for c in res.conflict
        ],
        "scores": [
            {
                "option": id_to_oname.get(oid, oid),
                "factor": id_to_fname.get(fid, fid),
                "value": round(session.score_matrix[oid][fid].value, 3),
                "rationale": session.score_matrix[oid][fid].rationale,
            }
            for oid in session.score_matrix for fid in session.score_matrix[oid]
        ],
    }


def session_view(session: DecisionSession) -> dict:
    return {
        "id": session.id,
        "status": session.status.value,
        "context_summary": session.context_summary,
        "options": [{"id": o.id, "name": o.name, "description": o.description} for o in session.options],
        "factors": [
            {"id": f.id, "name": f.name, "description": f.description,
             "type": f.type, "relevance": f.relevance}
            for f in session.factors
        ],
    }
