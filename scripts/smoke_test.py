"""End-to-end smoke test: drives one real session through the orchestrator.

Makes REAL OpenAI calls (uses .env). Auto-answers every question so we can verify
the full SETUP -> ANALYZE -> QUESTIONING -> RESULT pipeline produces a coherent
result. Never prints the API key.

Run (from project root, using the venv python):  python -m scripts.smoke_test
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import orchestrator  # noqa: E402

SAMPLE = (
    "지금 다니는 안정적인 대기업에 계속 다닐지, 초기 스타트업으로 이직할지 고민이야. "
    "스타트업은 연봉은 비슷한데 지분을 주고 성장 기회가 크지만 불안정해. "
    "대기업은 워라밸이 좋고 안정적이지만 성장이 더뎌."
)


def auto_value(qv: dict):
    kind = qv.get("kind")
    if kind == "weight_pairwise":
        return "a"
    atype = qv.get("answer_type")
    if atype == "binary":
        return "yes"
    if atype == "choice":
        choices = qv.get("choices") or []
        return choices[0]["value"] if choices else "yes"
    if atype == "scale":
        return int(qv.get("scale", {}).get("max", 5))
    if atype == "count":
        return 3
    return "yes"


def main() -> int:
    print("Creating session (this makes several OpenAI calls)...")
    session = orchestrator.create_session(SAMPLE)
    print(f"  session={session.id}  status={session.status.value}")
    print(f"  options: {[o.name for o in session.options]}")
    print(f"  factors: {[(f.name, f.type) for f in session.factors]}")

    q = orchestrator.next_question(session)
    asked = 0
    while q is not None:
        qv = orchestrator.question_view(session, q)
        val = auto_value(qv)
        print(f"  Q{asked+1} [{qv['kind']}] {qv['question'][:60]!r} -> {val}")
        q = orchestrator.submit_answer(session, q.id, val)
        asked += 1
        if asked > 12:
            print("  (safety break)")
            break

    result = orchestrator.result_view(session)
    print("\n=== RESULT ===")
    print(f"  winner: {result['winner']}")
    print(f"  robustness: {result['robustness']}  margin: {result['margin']}  top_driver: {result['top_driver']}")
    print("  ranking:")
    for r in result["ranking"]:
        print(f"    - {r['name']}: {r['utility']}")
    print("  drivers:")
    for d in result["drivers"]:
        print(f"    - {d['factor']}: w={d['weight']} flip={d['flip_threshold']}")
    print("\n  report:\n" + "\n".join("    " + line for line in result["report"].splitlines()))
    print("\n  next_info:")
    for n in result["next_info"]:
        print(f"    - {n}")
    print("\nSMOKE TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
