"""LLM prompt builders.

Each builder returns (system, user) message strings and demands a strict JSON
object reply. Per §11 of the design doc, the bulk of the sophistication is invested
in `decide_factors` and `score_options` — the two calls the user never verifies.

All natural-language fields (questions, rationales, report) must be written in the
SAME language as the user's input. JSON keys stay in English.
"""
from __future__ import annotations

import json
from typing import Any

_LANG_RULE = (
    "CRITICAL LANGUAGE RULE: Write EVERY human-readable value in the SAME language the user "
    "wrote their problem in. This includes option names, factor names, descriptions, relevance, "
    "questions, examples, choice labels, rationales, summaries, and report text — names included, "
    "not just sentences. Only JSON keys and enum values (type, direction, answer_type) stay in "
    "English exactly as specified."
)

_JSON_RULE = "Reply with ONLY a single valid JSON object. No markdown fences, no prose before or after."

_EASY = (
    "TONE: phrase everything in plain, everyday words a 12-year-old would understand. No jargon, "
    "no stiff or technical terms. Keep names short and concrete; keep questions and examples warm, "
    "friendly, and natural — like a thoughtful friend asking."
)


# --------------------------------------------------------------------------- #
# 1. extract_options
# --------------------------------------------------------------------------- #
def extract_options(context: str) -> tuple[str, str]:
    system = (
        "You are a decision analyst. The user describes a dilemma in free text. "
        "Your job is to normalize it into 2-4 mutually-exclusive, well-defined options "
        "and a concise neutral summary of their situation. Do NOT ask the user anything — "
        "resolve ambiguity yourself with the most reasonable interpretation.\n"
        f"{_LANG_RULE}\n{_JSON_RULE}"
    )
    user = (
        "User's dilemma:\n"
        f'"""\n{context}\n"""\n\n'
        "Return JSON of the exact shape:\n"
        "{\n"
        '  "context_summary": "<2-4 sentence neutral summary of the decision and constraints>",\n'
        '  "options": [ {"name": "<short distinct label>", "description": "<one sentence>"}, ... ]\n'
        "}\n"
        "Rules: 2-4 options; options must be mutually exclusive and concrete; "
        "if the user only implies options, infer the most sensible explicit set."
    )
    return system, user


# --------------------------------------------------------------------------- #
# 2. decide_factors  (HIGH-STAKES — user never verifies this)
# --------------------------------------------------------------------------- #
def decide_factors(options: list[dict], context_summary: str) -> tuple[str, str]:
    system = (
        "You are a rigorous decision analyst choosing the criteria (factors) that should "
        "drive THIS decision, and their initial importance. The user will NOT review or edit "
        "your factor set, so it must be correct, complete, and non-redundant.\n\n"
        "Mandatory self-checks before answering:\n"
        "1. META-FACTOR CHECKLIST — explicitly consider whether each of these belongs, and "
        "include any that are relevant: opportunity cost, reversibility, time horizon, risk/"
        "downside, financial cost, effort/feasibility, alignment with long-term goals. People "
        "routinely forget these.\n"
        "2. COMPLETENESS — 'What important criterion am I missing?'\n"
        "3. NON-REDUNDANCY — 'Are any two factors measuring the same thing?' Merge if so.\n"
        "4. RANK-FIRST IMPORTANCE — decide the RELATIVE ordering of importance first, then assign "
        "importance numbers consistent with that ranking (the numbers are only a prior; the user's "
        "later choices refine them).\n\n"
        "Factor types:\n"
        "- 'concrete': can be scored directly from the options (e.g. salary, distance).\n"
        "- 'abstract': a soft preference that must be measured indirectly via easy indicator "
        "questions later (e.g. 'how much you value growth', 'stress tolerance').\n"
        "Factor names must be SHORT, concrete, and in plain everyday words — avoid abstract or "
        "technical labels (prefer wording like 'money' over 'financial compensation').\n"
        f"{_LANG_RULE}\n{_EASY}\n{_JSON_RULE}"
    )
    user = (
        f"Decision summary:\n{context_summary}\n\n"
        f"Options:\n{json.dumps(options, ensure_ascii=False, indent=2)}\n\n"
        "Choose 5-7 factors. Return JSON:\n"
        "{\n"
        '  "factors": [\n'
        "    {\n"
        '      "name": "<short distinct criterion name>",\n'
        '      "description": "<what it measures, one sentence>",\n'
        '      "relevance": "<why it matters for THIS decision>",\n'
        '      "type": "concrete" | "abstract",\n'
        '      "direction": "higher_better" | "lower_better",\n'
        '      "importance": <number 0..1, relative initial weight>\n'
        "    }, ...\n"
        "  ]\n"
        "}\n"
        "Make factor names unique. Importance values should reflect your ranking "
        "(they need not sum to 1; they will be normalized)."
    )
    return system, user


# --------------------------------------------------------------------------- #
# 3. decompose_indicators
# --------------------------------------------------------------------------- #
def decompose_indicators(abstract_factors: list[dict], context_summary: str) -> tuple[str, str]:
    system = (
        "You operationalize ABSTRACT decision factors into easy, concrete indicator questions "
        "that indirectly reveal how much the user cares about each factor. A good indicator is "
        "answerable in seconds and behavioral/observable, not a direct 'how much do you value X?' "
        "question.\n\n"
        "Each indicator needs a STRUCTURED mapping converting the answer into a 'signal' in [0,1], "
        "where 1 = the answer reveals the user cares a LOT about this factor, 0 = cares little.\n"
        "Mapping format by answer_type:\n"
        '- binary:  {"yes": <signal 0..1>, "no": <signal 0..1>}\n'
        '- scale:   {"min": 1, "max": 5, "min_signal": <0..1>, "max_signal": <0..1>}  (linear)\n'
        '- count:   {"low": <n>, "high": <n>, "low_signal": <0..1>, "high_signal": <0..1>}  (linear, clamped)\n'
        '- choice:  {"choices": [{"label": "<text>", "signal": <0..1>}, ...]}\n'
        f"{_LANG_RULE}\n{_EASY}\n{_JSON_RULE}"
    )
    user = (
        f"Decision summary:\n{context_summary}\n\n"
        f"Abstract factors:\n{json.dumps(abstract_factors, ensure_ascii=False, indent=2)}\n\n"
        "For EACH abstract factor produce 1-2 indicators. Return JSON:\n"
        "{\n"
        '  "indicators": [\n'
        "    {\n"
        '      "factor_name": "<exact name of the abstract factor>",\n'
        '      "question": "<easy concrete question>",\n'
        '      "answer_type": "binary" | "scale" | "count" | "choice",\n'
        '      "mapping": { ... per the format above ... }\n'
        "    }, ...\n"
        "  ]\n"
        "}\n"
        "For choice questions, give 2-4 choices. For scale, prefer a 1-5 scale."
    )
    return system, user


# --------------------------------------------------------------------------- #
# 4. verbalize_pairs  (batched — all factor pairs at once)
# --------------------------------------------------------------------------- #
def verbalize_pairs(factors: list[dict], context_summary: str) -> tuple[str, str]:
    system = (
        "You turn factor trade-offs into a single, friendly, concrete question the user can answer "
        "by picking a side. For each pair of factors, write a short question asking which matters "
        "more for this decision. For each side, give a CONCRETE 1-2 sentence example of what "
        "prioritising it actually looks like in real life — a real scene or choice the person would "
        "make, never an abstract definition.\n"
        f"{_LANG_RULE}\n{_EASY}\n{_JSON_RULE}"
    )
    names = [f["name"] for f in factors]
    user = (
        f"Decision summary:\n{context_summary}\n\n"
        f"Factors: {json.dumps(names, ensure_ascii=False)}\n\n"
        "For EVERY unordered pair of distinct factors, produce an entry. Return JSON:\n"
        "{\n"
        '  "pairs": [\n'
        "    {\n"
        '      "factor_a": "<name>",\n'
        '      "factor_b": "<name>",\n'
        '      "question": "<which matters more to you, in plain friendly words>",\n'
        '      "example_a": "<concrete 1-2 sentence real-life scene of leaning to factor_a, not a definition>",\n'
        '      "example_b": "<concrete 1-2 sentence real-life scene of leaning to factor_b, not a definition>"\n'
        "    }, ...\n"
        "  ]\n"
        "}"
    )
    return system, user


# --------------------------------------------------------------------------- #
# 5. score_options  (HIGH-STAKES — user never verifies this)
# --------------------------------------------------------------------------- #
def score_options(options: list[dict], factors: list[dict], context_summary: str) -> tuple[str, str]:
    system = (
        "You score each option against each factor. The user will NOT review these scores, so "
        "they must be calibrated and well-reasoned.\n\n"
        "Rules:\n"
        "1. RELATIVE ANCHORING — within a single factor, score the options BY COMPARISON to each "
        "other so they are consistent (the best option on that factor anchors high, the worst anchors "
        "low). Use the full 0..1 range when options genuinely differ.\n"
        "2. DIRECTION-NAIVE — always score so that 1.0 = 'best on this factor' regardless of whether "
        "the factor is higher_better or lower_better; just rate desirability.\n"
        "3. MANDATORY RATIONALE — every score needs a one-line reason. No bare numbers.\n"
        "4. UNCERTAINTY — when you are unsure or the value depends on the future, give a wider "
        "'range' [lo, hi] and a lower 'confidence'. Be conservative on speculative future factors.\n"
        f"{_LANG_RULE}\n{_JSON_RULE}"
    )
    user = (
        f"Decision summary:\n{context_summary}\n\n"
        f"Options:\n{json.dumps(options, ensure_ascii=False, indent=2)}\n\n"
        f"Factors:\n{json.dumps(factors, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON keyed by option name then factor name:\n"
        "{\n"
        '  "scores": {\n'
        '    "<option name>": {\n'
        '      "<factor name>": {\n'
        '        "value": <0..1 desirability>,\n'
        '        "range": [<lo 0..1>, <hi 0..1>],\n'
        '        "confidence": <0..1>,\n'
        '        "rationale": "<one line>"\n'
        "      }, ...\n"
        "    }, ...\n"
        "  }\n"
        "}\n"
        "Include EVERY option × factor cell."
    )
    return system, user


# --------------------------------------------------------------------------- #
# 5b. decompose_question  (§6.3 recursive decomposition of a hard trade-off)
# --------------------------------------------------------------------------- #
def decompose_question(
    factor_a: dict, factor_b: dict, context_summary: str,
    asked_so_far: list[str], depth: int,
) -> tuple[str, str]:
    system = (
        "The user could not decide which of two decision factors matters more. Break that "
        "trade-off into ONE easier, more concrete question that INDIRECTLY reveals which side "
        "they lean toward — prefer a concrete scenario or a past-behaviour proxy over abstract "
        "wording (e.g. instead of 'do you value growth or stability', ask 'in the last year, did "
        "you spend personal time learning something new for your career?'). "
        "The deeper the level, the simpler and more concrete the question must be. "
        "Produce EXACTLY TWO answer options, shown as cards the user taps — one leaning to factor "
        "A, one to factor B — each with a short label, a concrete real-life example, and a strength.\n"
        f"{_LANG_RULE}\n{_EASY}\n{_JSON_RULE}"
    )
    user = (
        f"Decision summary:\n{context_summary}\n\n"
        f"Factor A: {factor_a.get('name','')} — {factor_a.get('description','')}\n"
        f"Factor B: {factor_b.get('name','')} — {factor_b.get('description','')}\n"
        f"Decomposition depth (1 = first split, higher = must be even simpler): {depth}\n"
        f"Already-asked easier questions (make this one DIFFERENT and simpler):\n"
        f"{json.dumps(asked_so_far, ensure_ascii=False)}\n\n"
        "Return JSON with EXACTLY TWO options (first leans to Factor A, second to Factor B):\n"
        "{\n"
        '  "question": "<one easy, concrete question>",\n'
        '  "options": [\n'
        '    {"label": "<short 2-5 word title>", "example": "<concrete 1-2 sentence scene>", "favors": "a", "strength": <0..1>},\n'
        '    {"label": "<short 2-5 word title>", "example": "<concrete 1-2 sentence scene>", "favors": "b", "strength": <0..1>}\n'
        "  ]\n"
        "}\n"
        "Each option is a card: a short label plus a concrete real-life example. "
        "favors \"a\" = picking it means Factor A matters more; \"b\" = Factor B. "
        "strength 0.5 (mild) .. 1.0 (strong)."
    )
    return system, user


# --------------------------------------------------------------------------- #
# 6. generate_report
# --------------------------------------------------------------------------- #
def generate_report(result_data: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You write a short, friendly decision-debugging report. The goal is NOT to declare a winner "
        "with a score, but to EXPLAIN: why the leading option leads, what assumption would flip the "
        "result, which input is most uncertain, and whether what the user said matches the numbers. "
        "Be honest about closeness. Reference options and factors by their names.\n"
        "LANGUAGE: write the ENTIRE report in the SAME language as the 'decision_summary' field in the "
        "data below (that is the user's language). Do NOT write in English unless that summary is in "
        "English.\n"
        "FORMAT: exactly four numbered points — 1) 2) 3) 4) — each only 1-2 sentences. Put EACH point "
        "on its own line, separated by a blank line (\\n\\n). Never run them together in one paragraph.\n"
        f"{_EASY}\n{_JSON_RULE}"
    )
    user = (
        "Decision analysis data (utilities, ranking, sensitivity, weights, score rationales, "
        "stated-vs-revealed conflicts):\n"
        f"{json.dumps(result_data, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON (note the \\n\\n blank lines between points):\n"
        "{\n"
        '  "report": "1) <leading option and its biggest reason>.\\n\\n'
        '2) <the smallest change that would flip the result>.\\n\\n'
        '3) <the most uncertain assumption>.\\n\\n'
        '4) <whether stated vs revealed preferences conflict, or that they don\'t>.",\n'
        '  "next_info": ["<one concrete thing worth finding out next>", ...]\n'
        "}\n"
        "Write in the user's language (match decision_summary). Each numbered point MUST be separated "
        "by a blank line."
    )
    return system, user
