# Mind Compass — 시스템 설계서 (v3 · 구현 반영본)

> v2 설계서를 실제 구현 기준으로 갱신한 문서. 주요 변경: **모든 질문을 "카드 2장 비교" 한 형식으로 통일**(네/아니오·척도 등 indicator 질문 제거), **"잘 모르겠어요" 시 더 쉬운 질문으로 재귀 분해 후 roll-up**, **조기 종료 없이 고정 N문항**, 그리고 **계산 레이어 상세화**(§8).

## 0. 한 줄 정의
사용자의 의사결정을, LLM이 판단 기준(factor)·중요도·선택지 점수를 **자율적으로 결정**하고, 사용자는 **"A vs B" 카드 두 장 중 하나만 고르면**(어려우면 더 쉬운 카드 질문으로 쪼개서) 그 선택으로 숨은 선호를 추정한 뒤, **어떤 가정이 결론을 바꾸는지**까지 보여주는 의사결정 디버깅 도구.

---

## 1. 설계 원칙

1. **LLM ↔ 계산 분리.** LLM은 물렁한 일(기준·중요도·점수 결정, 질문 생성, 리포트), 결정론적 수학은 엄밀한 일(가중치 정교화·집계·민감도·질문선택). 수학 코어는 **순수 함수** — 같은 입력 → 같은 출력, 테스트 가능(`backend/compute.py`, 단위테스트 27개).
2. **질문은 인사이트 생성용.** 추상적 trade-off는 직접 묻지 않고, 못 고르겠으면 **더 쉽고 구체적인 카드 질문으로 분해**해 간접적으로 끌어낸다.
3. **사용자는 "고르기"만 한다.** factor·중요도(초기 가중치)·선택지 점수는 **전부 LLM이 자율로 결정**한다. 사용자의 유일한 입력은 **카드 선택**이고, 그 선택이 가중치를 정교화한다. ⚠️ 무보정이므로 LLM 판단(특히 factor 세트·점수)이 매우 정교해야 한다 → §10.
4. **결과는 점수가 아니라 설명.** 최종 산출물은 "B 84점"이 아니라 "왜 B인가 + 뭐가 바뀌면 뒤집히나 + 뭘 더 알아야 하나".

---

## 2. 아키텍처 (실제 스택)

```
┌──────────────────────────────────────────────┐
│  브라우저 (frontend/ — Apple 스타일 바닐라 JS) │   ← API 키 절대 노출 안 됨
└───────────────────┬──────────────────────────┘
                    │  HTTP (/api/*)
┌───────────────────▼──────────────────────────┐
│        FastAPI (backend/main.py)              │
│   ┌──────────────┐      ┌─────────────────┐   │
│   │  LLM Layer   │      │  Compute Layer  │   │
│   │ llm.py       │      │  compute.py     │   │
│   │ prompts.py   │      │ (numpy/scipy,   │   │
│   │ (OpenAI,     │      │  순수함수)       │   │
│   │  서버 전용)   │      │ ─ 가중치 정교화  │   │
│   │ ─ 선택지     │      │   (BT 로지스틱)  │   │
│   │ ─ factor+중요도│     │ ─ 집계 (WSM)    │   │
│   │ ─ 비교질문/예시│     │ ─ 민감도 (OAT)  │   │
│   │ ─ 하위질문 분해│     │ ─ 질문선택 (QBC)│   │
│   │ ─ 점수/리포트 │      │ ─ 충돌 감지      │   │
│   └──────────────┘      └─────────────────┘   │
│        Orchestrator (orchestrator.py)         │
│        세션 상태머신 + roll-up                  │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│        Store (store.py · 인메모리 세션)         │
└──────────────────────────────────────────────┘
```

- **백엔드:** Python 3.13 + FastAPI + numpy + scipy + openai SDK. LLM = OpenAI(기본 `gpt-4o`, `.env`의 `OPENAI_MODEL`로 교체). 모든 LLM 호출은 `temperature 0` + JSON 모드 + 스키마 검증.
- **프론트엔드:** 빌드 불필요한 바닐라 HTML/CSS/JS. `DESIGN.md`(Apple) 토큰 적용. FastAPI가 정적 서빙.
- **저장:** 인메모리(프로세스 메모리). 재시작 시 휘발.
- **보안:** API 키는 `.env`(gitignore)에만. 코드/프론트/로그/응답 어디에도 안 들어감. 모든 모델 호출은 서버에서만. 500 응답은 일반 메시지(예외 원문 비노출). `/api/health`는 `key_configured` 불리언만.

**데이터 흐름:** LLM이 factor·중요도·점수·질문을 자율 결정 → 사용자는 카드 선택만 → 못 고르면 더 쉬운 카드로 분해 후 roll-up → 계산기가 선택으로 가중치 정교화·집계·민감도 계산 → LLM이 한국어 리포트.

---

## 3. 데이터 모델 (`backend/models.py`)

### DecisionSession
| 필드 | 타입 | 설명 |
|---|---|---|
| id | str | 세션 ID |
| status | State | `SETUP`\|`ANALYZE`\|`QUESTIONING`\|`RESULT` |
| context / context_summary | str | 사용자 원문 / LLM 요약 (리포트 언어 기준) |
| options | [Option] | 선택지 2~4개 (LLM 정규화) |
| factors | [Factor] | 판단 기준 5~7개 (LLM 자율 결정) |
| weights | WeightState | prior(LLM 초기) + current(선택 반영) + conflict |
| score_matrix | {option_id × factor_id → Score} | 점수표 (LLM 자율) |
| norm_scores | {option_id × factor_id → [0,1]} | 정규화 점수 |
| pairwise_verbalizations | {factor쌍 → {question, example_a, example_b}} | 미리 생성한 비교 질문/예시 |
| questions | [Question] | 질문 큐 |
| answers | [Answer] | 사용자 선택 로그 |
| active_decomposition | dict? | 분해 진행 상태 {root_id, factor_a_id, factor_b_id, depth} |
| result | Result? | 최종 결과 |

### Option `{ id, name, description }`

### Factor (LLM이 이름·중요도·타입까지 자율 결정)
`{ id, name, description, relevance, type("concrete"|"abstract"), direction, importance }`
> `type`/`direction`은 점수화 참고용. (v2의 indicator 분해는 제거됨 — §5/§6 참고.)

### Score (option × factor — LLM 자율, 사용자 수정 없음)
`{ value[0,1] 선호도, source="llm", rationale, range?, confidence? }`

### WeightState
`{ prior:{factor→w}, current:{factor→w}, conflict:[{factor_id, prior, current, direction}] }`

### Question
| 필드 | 설명 |
|---|---|
| id | |
| kind | `"weight_pairwise"` (기준 A vs B) \| `"sub_question"` (분해된 더 쉬운 카드 질문) |
| payload | pairwise: `{factor_a_id, factor_b_id}` / sub_question: `{root_id, factor_a_id, factor_b_id, depth, question_text, options:[{label, example, favors, strength}]}` |
| parent_id | sub_question일 때 원래 질문 id |
| status | `pending` \| `answered` |

### Answer (사용자의 유일한 입력)
`{ question_id, value, timestamp }`
- pairwise: value ∈ `"a"|"b"|"unknown"`
- sub_question: value = 고른 카드 label 또는 `"unknown"`
- **roll-up으로 결정된 원 질문**: value = `{rolled_up:true, target, weight, gave_up}` (계산기가 비교로 변환)

### Result
`{ utilities, ranking, sensitivity, robustness("stable"|"close"), report(NL), next_info[], conflict[] }`

---

## 4. 세션 상태머신

```
SETUP → ANALYZE → QUESTIONING → RESULT
                      ↑__│ (카드 선택 → 가중치 정교화 루프)
                      ↑__│ (못 고르면 sub_question 재귀 분해 → roll-up)
```

| 상태 | 무슨 일 | 주체 |
|---|---|---|
| SETUP | 고민 입력 접수 | 사용자 (입력만) |
| ANALYZE | 선택지 정규화 + factor·중요도 결정 + 비교질문/예시 생성 + 선택지 점수 결정 + 정규화 | **LLM 자율 + Compute** (컨펌 없음) |
| QUESTIONING | **고정 N개**의 카드 질문 루프, 사용자 선택, 가중치 정교화 (못 고르면 분해→roll-up) | 사용자 선택 + Compute |
| RESULT | 집계 + 민감도 + 충돌 + 리포트 | Compute + LLM |

> **사용자 상호작용은 SETUP(입력)과 QUESTIONING(카드 선택) 두 군데뿐.** 나머지는 시스템이 자율 처리.

---

## 5. 모듈 설계 (입력 / 출력 / 방법)

### 5.1 입력 & 선택지 정규화 — [LLM] `extract_options`
- 입력: 자유서술 고민 → 출력: `{ options[], context_summary }`. 모호해도 묻지 않고 가장 합리적으로 정규화.

### 5.2 Factor + 중요도 결정 — [LLM] `decide_factors`
- 입력: options, context → 기준 5~7개 + 각 초기 중요도(prior). 메타 factor(기회비용·가역성·시간지평·리스크) 자기점검. **이름은 짧고 쉬운 일상어**로(예: "재정적 보상"→"비용"). 사용자 개입 없음.

### 5.3 비교 질문 + 예시 생성 — [LLM] `verbalize_pairs` (배치)
- 입력: factors, context → 모든 기준 쌍에 대해 `{question, example_a, example_b}`를 한 번에 생성해 캐시.
- 예시는 추상 정의가 아니라 **그쪽을 우선했을 때의 구체적 1~2문장 장면**.
- (v2의 `decompose_indicators`(네/아니오·척도 간접질문)는 **제거**. 추상 기준은 비교 질문 + 분해(§6.3)로 측정.)

### 5.4 질문 엔진 — [Compute + LLM] → **§6에서 상세**

### 5.5 선택 → 가중치 정교화 — [Compute] `refine_weights` → **§8.2**

### 5.6 선택지 점수화 — [LLM] `score_options`
- 각 선택지를 factor별로 채점(같은 기준 내 상대 비교로 일관성). 1.0=그 기준에서 최고(desirability). 근거(rationale) 필수, 불확실하면 range/confidence. → 정규화는 §8.1.
- ⚠️ 사용자 무검증 → 결과 품질 좌우 (§10).

### 5.7 집계 — [Compute] `aggregate` → **§8.3**
### 5.8 민감도 — [Compute] `sensitivity` → **§8.4**
### 5.9 리포트 — [LLM] `generate_report`
- 입력: 효용·랭킹·민감도·가중치·근거·충돌 + **decision_summary(언어 기준)**.
- 출력: 한국어(=사용자 언어) NL 리포트. **4개 항목을 빈 줄로 구분**: ① 1위와 이유(top driver) ② 뭐가 바뀌면 뒤집힘 ③ 가장 불확실한 가정 ④ stated vs revealed 충돌 + `next_info[]`.

---

## 6. 질문 엔진 상세 (가장 중요)

### 6.1 질문은 단 한 형식 — "카드 2장 비교"
모든 질문(메인이든 분해된 하위든)이 동일한 UI: **질문 한 줄 + 카드 2장(이름 + 예시) + `잘 모르겠어요`**.
- **weight_pairwise:** 카드 A = 기준A(이름)+예시, 카드 B = 기준B+예시. 고르면 그 기준이 더 중요.
- **sub_question:** 카드 A/B = 더 쉬운 구체 상황 2개(각각 A쪽/B쪽으로 기움).
- "둘 다 비슷" 버튼은 제거(선택지는 A / B / 잘 모르겠어요).

### 6.2 질문 선택 (다음에 뭘 물을까) — QBC, 순서 결정용
- ANALYZE에서 만들어둔 비교 질문 후보(모든 기준 쌍) 중, **Query-by-Committee 휴리스틱**으로 *"지금 가장 의견이 갈리는 = 결과를 뒤집을 수 있는 가장 애매한 trade-off"* 를 고른다 → §8.5.
- **종료는 조기 종료 없이 고정 N개**(현재 `NUM_QUESTIONS=5`). QBC는 종료가 아니라 *순서*에만 쓴다. (하위 질문은 N에 포함 안 됨.)

### 6.3 재귀 분해 (사용자가 못 고를 때) — 핵심
- 트리거: pairwise에서 **`잘 모르겠어요`** 선택.
- 동작: 그 trade-off(A vs B)를 **더 쉽고 구체적인 카드 2장 질문**으로 쪼갠다(`decompose_question`, 즉석 LLM 생성). 각 카드는 A/B 중 어디로 기우는지(`favors`)와 세기(`strength`)를 내장.
- **재귀:** 하위 질문에서 또 `잘 모르겠어요` → 한 단계 더 쉬운 카드 질문(깊이 +1). **최대 깊이 4**까지. 그 안에 못 정하면 "동등(약한 신호)"으로 마무리.
- **roll-up:** 카드를 고르면 → §8.7로 원래 A vs B 비교값을 계산해 **원 질문을 "결정"** 처리 → 가중치에 반영 → **다음 메인 질문으로**.
- 하위 질문은 **메인 N개 예산에 포함되지 않는다** (한 메인 질문을 푸는 과정).

```
"안정성 vs 성장 — 뭐가 중요?"   (못 고름 → 잘 모르겠어요)
   └─ "작년 주말에 더 자주 한 일은?"  [아르바이트(안정쪽) | 취미·자기계발(성장쪽)]   (구체 카드)
        └─ (또 모르겠으면) 더 쉬운 카드 …  →  카드 선택 시 roll-up → 안정 vs 성장 비교값 산출
```

### 6.4 선택 → 비교 신호
- pairwise: A→(i=A,j=B,target=1,weight=1), B→(target=0,weight=1).
- sub_question roll-up: favors/strength → target=0.5±0.5·strength, weight=1.0 (못 정하고 종료 시 target 0.5, weight 0.3).
- 이 비교들이 §8.2 BT 회귀로 들어가 가중치를 정교화.

### 6.5 종료 조건
- **메인 질문 N개(=5) 다 묻기.** (피로 방지 상한 = 동시에 고정 길이.) 조기 종료 로직 없음 → 진행도 `질문 k/5`가 항상 정직.

---

## 7. LLM 호출 명세 (구조화 I/O, temperature 0, JSON 모드)

| 함수 | 시점 | 입력 | 출력(JSON) | 비고 |
|---|---|---|---|---|
| `extract_options` | ANALYZE | context | `{options[], context_summary}` | 컨펌 없이 자율 정규화 |
| `decide_factors` | ANALYZE | options, summary | `factors[]{name,desc,type,direction,importance,relevance}` | **고난도(§10)**, 쉬운 이름 |
| `verbalize_pairs` | ANALYZE | factors, summary | `pairs[]{factor_a,factor_b,question,example_a,example_b}` | 모든 쌍 배치, 구체 예시 |
| `score_options` | ANALYZE | options, factors | `{(option,factor)→{value,range,confidence,rationale}}` | **고난도(§10)** |
| `decompose_question` | QUESTIONING(필요시) | factor_a, factor_b, summary, 이미물은것, depth | `{question, options:[{label,example,favors,strength}]×2}` | "모르겠어요" 시 즉석 생성 |
| `generate_report` | RESULT | result_data(+decision_summary) | `{report, next_info[]}` | 사용자 언어, 4항목 줄바꿈 |

공통 규칙: **① 사용자 언어 그대로(이름 포함, JSON 키만 영어) ② 중학생도 알 쉬운 일상어·친근하게 ③ 유효한 JSON 하나만.** 출력은 스키마 검증(실패 시 1회 재시도) 후 Store/Compute로.

> 세션당 LLM 호출: ANALYZE 4회(extract/decide/verbalize/score) + "모르겠어요" 누를 때마다 1회(decompose) + RESULT 1회(report).

---

## 8. 계산 레이어 — 어떻게 계산하는가 (핵심) ★

모두 `backend/compute.py`의 **순수 함수**(numpy/scipy만, 결정론적, 단위테스트됨). 기호: σ는 시그모이드, prior=LLM 초기 중요도(정규화), w=최종 가중치.

```
σ(x) = 1 / (1 + e^(−x))
```

### 8.1 점수 정규화 — `normalize_scores(raw_scores, directions)`
- 입력: `raw_scores[option][factor] ∈ [0,1]` (LLM이 매긴 **선호도**, 1=그 기준에서 최고), `directions`(참고용).
- **factor별 min-max**로 [0,1] 재정규화 → 스케일 다른 기준이 가중합을 지배하는 것 방지.
- ⚠️ **direction으로 뒤집지 않음** (LLM이 이미 desirability로 매김 — 뒤집으면 이중 적용). 한 factor에서 옵션 간 차이가 거의 없으면(spread < 1e-9) 전부 **0.5**로(노이즈 증폭 방지). 형태 보존.

### 8.2 가중치 정교화 — `refine_weights(prior, comparisons, lam=0.5)` (Bradley-Terry)
각 factor f에 **강도 θ_f**를 둔다. 두 기준 비교는 BT 모델:
```
P(기준 i 가 j 보다 중요) = σ(θ_i − θ_f_j)
```
사용자의 카드 선택/roll-up이 비교 리스트 `{i, j, target∈[0,1], weight}`로 들어온다(target=1: i 확실, 0: j, 0.5: 동등; weight: 신뢰도). LLM 초기 중요도를 warm-start/정규화 기준으로:
```
θ_prior_f = log(max(prior_f, 1e-9))
손실 L(θ) = Σ_obs  weightₒ · BCE( targetₒ , σ(θ_iₒ − θ_jₒ) )   +   λ · Σ_f (θ_f − θ_prior_f)²
          BCE(t, p) = −[ t·log p + (1−t)·log(1−p) ]   (p는 [1e-9, 1−1e-9]로 클립)
```
- `scipy.optimize.minimize`(L-BFGS-B, 해석적 그래디언트 ∝ (p−t))로 θ 최소화, 시작점 θ_prior.
- 결과 **w = softmax(θ)** (합=1, 음수 없음). **비교가 없으면 prior 그대로**(1e-6 이내).
- λ(=0.5)는 사용자 신호와 LLM prior 사이의 균형 — 신호가 약하면 prior 쪽으로 당겨짐. **soft target**이라 카드 강도(strength)나 동등도 자연스럽게 표현.

### 8.3 집계 — `aggregate(weights, norm_scores)` (WSM 가중합)
```
U(option) = Σ_f  w_f · s(option, f)
```
옵션별 효용. 빠진 셀은 0 취급.

### 8.4 민감도 — `sensitivity(weights, norm_scores)` (OAT / 토네이도)
- `utilities`, `ranking`(효용 내림차순, 동점은 id로 안정 정렬).
- **`flip_thresholds[f]`** = factor f의 가중치 w_f를 [0,1]로 쓸어보며(나머지 factor는 비례 재정규화해 합=1 유지) **1위 옵션이 바뀌는 최소 |Δw_f|** (그리드 ~200스텝 + 이분 정밀화). [0,1] 안에서 안 바뀌면 `None`.
- **`top_driver`** = flip_threshold가 가장 작은 기준(=가장 쉽게 결과를 뒤집는 동인). 아무것도 안 뒤집히면 가중치 최대 기준.
- **`margin`** = U(1위) − U(2위).
- **`robustness`** = `margin < 0.05` 또는 (유한한 최소 flip < 0.1) 이면 **"close(근소함)"**, 아니면 **"stable(탄탄함)"**. (옵션 1개면 ranking 1, margin 0, flip None, stable.)

### 8.5 다음 질문 선택 — `select_next_question(...)` (Query-by-Committee)
- `current` 근처로 **가중치 앙상블 30개** 생성: `θ = log(clip(current)) + N(0, 0.5)`, `w = softmax(θ)`, RNG는 시드 고정(`np.random.default_rng(seed)`, **결정론적**).
- 후보 쌍 (fa, fb)의 **의견 갈림(disagreement)** = 앙상블에서 `w[fa] > w[fb]`인 비율의 **이진 엔트로피** × 두 기준의 가중치 질량(heavy 기준 trade-off 우대). 최댓값 쌍 선택.
- 위원회가 1위에 합의 + 최대 disagreement < 0.05면 `None`(더 물어도 안 바뀜)을 반환할 수 있으나, **현재 오케스트레이터는 고정 N문항이라 종료엔 안 쓰고 순서 결정에만 사용**한다.

### 8.6 충돌 감지 — `detect_conflict(prior, current, threshold=0.15)`
- `|current_f − prior_f| > 0.15` 인 factor를 **stated(LLM 초기) vs revealed(선택 반영) 충돌**로 표시: `{factor_id, prior, current, direction("up"|"down")}`, |차이| 내림차순.

### 8.7 분해 roll-up (오케스트레이터)
- 하위 질문 카드 = `{favors∈"a"|"b"|"neither", strength∈[0,1]}`.
- 비교값: `favors=="a" → target = 0.5 + 0.5·strength`, `"b" → 0.5 − 0.5·strength`, `neither → 0.5`. weight = 1.0(카드 선택). 깊이 4까지 못 정하면 target 0.5, weight 0.3.
- 이 비교가 원래 (A,B) 쌍을 "결정"하고 §8.2로 들어가 가중치를 정교화.

**튜닝 상수:** `NUM_QUESTIONS=5`(고정 질문 수), `LAMBDA=0.5`(prior 당김), `MAX_DECOMP_DEPTH=4`(분해 한계).

---

## 9. API 인터페이스

```
POST  /api/sessions                       {context}            → {session, question, done}   (ANALYZE 자동 실행 + 첫 질문)
GET   /api/sessions/{id}/next-question                          → {question, done}
POST  /api/sessions/{id}/answers          {question_id, value} → {accepted, done, question}
GET   /api/sessions/{id}/result                                → Result 뷰
GET   /api/health                                              → {ok, model, key_configured}
```
> 컨펌/편집 엔드포인트 없음 — 사용자는 선택(answers)만 보낸다. `question` 객체에는 카드 2장(`factor_a/example_a`, `factor_b/example_b` 또는 sub_question의 `cards[]`)과 진행도가 담긴다.

---

## 10. 정교함이 필요한 곳 (품질 병목) ★

사용자가 무검증이므로 LLM 자율 판단의 품질이 전부다. 위험도:

| LLM 출력 | 사용자 검증 | 위험도 |
|---|---|---|
| 가중치(중요도) | 있음(카드 선택으로 정교화) | 낮음 |
| **factor 세트** | 없음 | **높음** |
| **선택지 점수** | 없음 | **높음** |
| 비교/하위 질문 문구 | 간접(이상하면 티남) | 중간 |

→ 프롬프트 정교함의 대부분을 **`decide_factors`·`score_options`** 에 투입.

**`decide_factors` 기법:** 메타 factor 체크리스트(기회비용·가역성·시간지평·리스크) 강제, 중복/누락 자기검증, 중요도는 *순위* 위주(어차피 선택으로 정교화), **짧고 쉬운 이름**.
**`score_options` 기법:** 근거(rationale) 강제, 같은 기준 내 *상대 비교*로 일관성, 불확실하면 range/confidence(민감도가 활용), 미래 불확실 factor는 보수적.
**질문 문구 기법:** 추상 정의 금지·구체 장면, 쉬운 일상어, 사용자 언어 일치.

---

## 11. 실행 / 보안 / 기술 스택

**실행 (Windows):**
```powershell
.\run.ps1     # 또는 run.bat
# → http://localhost:8000
```
수동: `python -m venv .venv` → `.\.venv\Scripts\python -m pip install -r requirements.txt` → `... -m uvicorn backend.main:app --port 8000`. 테스트: `... -m pytest`.

**보안:** API 키는 `.env`(gitignore)에만. 코드/프론트/로그/HTTP 응답 어디에도 안 들어감. 모든 OpenAI 호출은 서버 전용. `/api/health`는 `key_configured` 불리언만. 500 응답은 일반 메시지(예외 원문 비노출).

**스택:** Python 3.13 · FastAPI · numpy/scipy · openai · pydantic · python-dotenv / 바닐라 HTML·CSS·JS(Apple `DESIGN.md`). 인메모리 세션.

---

## 부록: 한눈에 보는 전체 흐름

```
[사용자] 고민 입력
  → [LLM] 선택지 정규화 + factor·중요도 결정 + 비교질문/예시 생성 + 선택지 점수
  → [Compute] 점수 정규화, prior 가중치 세팅
  → [사용자] 카드 2장 중 하나 선택  (못 고르면 → 더 쉬운 카드로 재귀 분해 → roll-up)
  → [Compute] 선택으로 가중치 정교화 (BT 로지스틱 + L2 prior, +충돌 감지)
  → 고정 5문항 반복 (QBC가 순서 결정)
  → [Compute] WSM 집계 + OAT 민감도
  → [LLM] 한국어 디버깅 리포트 (왜 + 뭐가 뒤집나 + 뭘 더 알아야 + 충돌)
```

**역할 분담:** LLM = factor·중요도·점수·질문·리포트 자율 결정 / Compute = 가중치 정교화·집계·민감도·질문선택·roll-up / **사용자 = 카드 선택만.**
