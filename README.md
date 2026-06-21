# Decision Debugger

LLM이 판단 기준(factor)·중요도·선택지 점수를 **자율로 결정**하고, 사용자는 **답하기 쉬운 질문에 선택만** 하면, 그 선택으로 숨은 선호를 추정해 **어떤 가정이 결론을 바꾸는지**까지 보여주는 의사결정 디버깅 도구.

설계 전문: [`decision_debugger_design.md`](decision_debugger_design.md) · UI 디자인 가이드: [`DESIGN.md`](DESIGN.md) (Apple 스타일)

---

## 빠른 실행

```powershell
# Windows PowerShell
.\run.ps1
```
```bat
:: cmd
run.bat
```

브라우저에서 **http://localhost:8000** 접속.

수동 실행:
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000
```

> **API 키**는 `.env` 파일에만 있습니다. `.env`는 `.gitignore`로 제외되며, 키는 서버 메모리에서만 사용되고 브라우저·로그·응답 어디에도 노출되지 않습니다. 모델은 `.env`의 `OPENAI_MODEL`로 변경할 수 있습니다 (기본 `gpt-4o`).

---

## 아키텍처

```
브라우저 (frontend/) ──HTTP──> FastAPI (backend/main.py)
                                  │
                     ┌────────────┴────────────┐
              LLM Layer (자율 결정)        Compute Layer (결정론)
              backend/llm.py              backend/compute.py
              backend/prompts.py          · 가중치 정교화 (Bradley-Terry)
              · 선택지/factor/점수         · 집계 (WSM)
              · indicator/질문/리포트       · 민감도 (OAT)
                                  │        · 질문 선택 (QBC)
                       Orchestrator (상태머신)
                       backend/orchestrator.py
                       SETUP→ANALYZE→QUESTIONING→RESULT
```

| 파일 | 역할 |
|---|---|
| `backend/main.py` | FastAPI 라우트 + 정적 프론트엔드 호스팅 |
| `backend/orchestrator.py` | 세션 상태머신, LLM↔Compute 연결 |
| `backend/llm.py` / `prompts.py` | OpenAI 호출 (서버 전용), 프롬프트 |
| `backend/compute.py` | 순수 수학 코어 (테스트 가능) |
| `backend/models.py` | pydantic 데이터 모델 (공유 계약) |
| `frontend/` | Apple 스타일 SPA (HTML/CSS/JS, 빌드 불필요) |
| `tests/` | 계산 코어 단위 테스트 |

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/sessions` | `{context}` → 세션 생성 + 분석 + 첫 질문 |
| GET | `/api/sessions/{id}/next-question` | 다음 질문 (없으면 `done:true`) |
| POST | `/api/sessions/{id}/answers` | `{question_id, value}` → 가중치 정교화 + 다음 질문 |
| GET | `/api/sessions/{id}/result` | 최종 리포트 |

## 테스트

```powershell
.\.venv\Scripts\python -m pytest -q
```

## 배포 (Render)

이 앱은 **상시 켜진 서버**가 필요하므로(세션이 인메모리) Render 같은 호스팅이 맞습니다. 코드 변경 없이 `render.yaml` 블루프린트로 배포됩니다.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/seokjoon1127/decision-debugger)

수동 단계:
1. [render.com](https://render.com) 로그인 → **New → Blueprint** → 이 GitHub 저장소 연결
2. Render가 `render.yaml`을 읽음 → **`OPENAI_API_KEY`** 값을 붙여넣으라고 물어봄 (저장소엔 없음, 대시보드에만 저장)
3. **Apply** → 빌드/배포 → 발급된 URL로 접속

> 무료 티어는 15분 유휴 시 잠들었다가 다음 접속에 ~30–60초 콜드스타트가 있습니다(데모엔 충분). 키는 평문 노출 이력이 있으면 새로 발급해 넣으세요.

## 비고

- 세션은 인메모리 저장 (서버 재시작 시 사라짐).
- 모든 질문은 "카드 2장 비교" 형식으로 통일, 고정 5문항(조기 종료 없음), "잘 모르겠어요" 시 더 쉬운 질문으로 재귀 분해. 자세한 내용은 설계서 [`decision_debugger_design_v3.md`](decision_debugger_design_v3.md) 참고.
