"""FastAPI app: REST API for the Decision Debugger + static frontend hosting.

The OpenAI key never crosses this boundary — the browser only ever talks to these
endpoints; all model calls happen server-side.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import orchestrator, store
from .config import settings
from .models import AnswerRequest, CreateSessionRequest

logger = logging.getLogger("decision_debugger")

app = FastAPI(title="Mind Compass", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local-only MVP
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    # Report only whether a key is configured — never the key itself.
    return {"ok": True, "model": settings.openai_model, "key_configured": settings.has_key}


@app.post("/api/sessions")
def create_session(req: CreateSessionRequest) -> JSONResponse:
    context = (req.context or "").strip()
    if len(context) < 5:
        raise HTTPException(status_code=400, detail="고민 내용을 조금 더 자세히 적어주세요.")
    try:
        session = orchestrator.create_session(context)
    except ValueError as exc:
        if str(exc) == "unidentifiable_input":
            raise HTTPException(status_code=400, detail="다시 한번 자세하게 입력해주세요.") from exc
        raise HTTPException(status_code=400, detail="입력을 분석할 수 없습니다. 고민 내용을 구체적으로 작성해주세요.") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze failed")
        # Never echo internal exception text to the client (keep full detail in logs only).
        raise HTTPException(status_code=500, detail="분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.") from exc
    q = orchestrator.next_question(session)
    return JSONResponse({
        "session": orchestrator.session_view(session),
        "question": orchestrator.question_view(session, q) if q else None,
        "done": q is None,
    })


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return orchestrator.session_view(session)


@app.get("/api/sessions/{session_id}/next-question")
def get_next_question(session_id: str) -> dict:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    q = orchestrator.next_question(session)
    return {
        "question": orchestrator.question_view(session, q) if q else None,
        "done": q is None,
    }


@app.post("/api/sessions/{session_id}/answers")
def post_answer(session_id: str, req: AnswerRequest) -> dict:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    try:
        q = orchestrator.submit_answer(session, req.question_id, req.value)
    except KeyError:
        raise HTTPException(status_code=400, detail="알 수 없는 질문입니다.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("submit_answer failed")
        raise HTTPException(status_code=500, detail="처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.") from exc
    return {
        "accepted": True,
        "done": q is None,
        "question": orchestrator.question_view(session, q) if q else None,
    }


@app.get("/api/sessions/{session_id}/result")
def get_result(session_id: str) -> dict:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    try:
        return orchestrator.result_view(session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("result failed")
        raise HTTPException(status_code=500, detail="리포트 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.") from exc


@app.get("/api/share/{session_id}")
def get_shared_result(session_id: str) -> dict:
    """Read-only public endpoint for shared result links."""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="공유 링크가 만료되었거나 존재하지 않습니다. 서버가 재시작되면 링크가 초기화됩니다.",
        )
    try:
        return orchestrator.result_view(session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("shared result failed")
        raise HTTPException(status_code=500, detail="결과를 불러올 수 없습니다.") from exc


# --- static frontend (mounted last so /api/* wins) ---
if settings.frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")
