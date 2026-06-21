# Decision Debugger — 시스템 설계서 (v2)

## 0. 한 줄 정의
사용자의 의사결정을, LLM이 판단 기준(factor)·중요도·선택지 점수를 **자율적으로 결정**하고, **사용자는 답하기 쉬운 질문에 선택만** 하면 그 선택으로 숨은 선호를 추정한 뒤, **어떤 가정이 결론을 바꾸는지**까지 보여주는 의사결정 디버깅 도구.

---

## 1. 설계 원칙 (4개)

1. **LLM ↔ 계산 분리.** LLM은 물렁한 일(기준·중요도·점수 결정, 질문 생성, 리포트), 결정론적 수학은 엄밀한 일(가중치 정교화·집계·민감도). 수학 코어는 **순수 함수** — 같은 입력 → 같은 출력, 테스트 가능.
2. **질문 중심.** 질문은 데이터 수집이 아니라 인사이트 생성. 추상 factor는 직접 묻지 않고 **indicator(쉬운 간접 질문)** 로 측정한다.
3. **사용자는 "선택"만 한다.** factor, 각 factor의 중요도(초기 가중치), 선택지 점수는 **전부 LLM이 자율로 결정**한다. 사용자 컨펌·편집·슬라이더 없음. 사용자의 유일한 입력은 **질문에 대한 선택**이고, 그 선택이 가중치를 정교화한다. ⚠️ **무보정이므로 LLM 판단(특히 factor 세트·점수)은 매우 정교해야 한다 → §11 참고.**
4. **결과는 점수가 아니라 설명.** 최종 산출물은 "B 84점"이 아니라 "왜 B인가 + 뭐가 바뀌면 뒤집히나 + 뭘 더 알아야 하나".

---

## 2. 아키텍처

```
┌─────────────────────────────────────────────┐
│                UI / API                      │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│        Orchestrator (세션 상태머신)           │
│   ┌──────────────┐      ┌─────────────────┐  │
│   │  LLM Layer   │      │  Compute Layer  │  │
│   │ (자율 결정)   │      │  (결정론)        │  │
│   │ ─ 선택지 정규화│      │ ─ 가중치 정교화 │  │
│   │ ─ factor+중요도│     │   (BT 로지스틱)  │  │
│   │ ─ indicator  │      │ ─ 집계 (WSM)    │  │
│   │ ─ 선택지 점수 │      │ ─ 민감도 (OAT)  │  │
│   │ ─ 리포트     │      │                 │  │
│   └──────────────┘      └─────────────────┘  │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│           Store (세션 상태)                   │
└─────────────────────────────────────────────┘
```

**데이터 흐름:** LLM이 factor·중요도·점수를 **자율 결정** → 사용자는 질문에 **선택만** → 계산기가 그 선택으로 **가중치를 정교화**하고 집계·민감도를 계산. LLM이 정한 점수/factor에 사용자 확인 단계는 없으며, 가중치만 사용자 선택으로 다듬어진다.

---

## 3. 데이터 모델

### DecisionSession
| 필드 | 타입 | 설명 |
|---|---|---|
| id | str | 세션 ID |
| status | State | 상태머신 현재 상태 |
| context | str | 사용자 상황 설명 |
| options | [Option] | 선택지 2~4개 (LLM 정규화) |
| factors | [Factor] | 판단 기준 (LLM 자율 결정) |
| weights | WeightState | LLM 초기값 + 선택 반영 |
| score_matrix | {option_id × factor_id → Score} | 점수표 (LLM 자율 결정) |
| answers | [Answer] | 사용자 선택 로그 |
| result | Result? | 최종 결과 |

### Option
`{ id, name, description }`

### Factor  (LLM이 이름·중요도·타입까지 자율 결정)
| 필드 | 설명 |
|---|---|
| id, name, description | 기본 |
| type | `"concrete"` (직접 점수화) \| `"abstract"` (indicator 질문 필요) |
| direction | `"higher_better"` \| `"lower_better"` |
| importance | LLM이 정한 초기 중요도 (→ WeightState.prior) |
| relevance | 왜 중요한지 (LLM 설명, 리포트용) |
| indicators | [Indicator] — abstract일 때만 |

### Indicator (abstract factor에 대한 사용자 선호를 끌어내는 쉬운 질문)
| 필드 | 설명 |
|---|---|
| id, factor_id | |
| question | 답하기 쉬운 구체 질문 (NL) |
| answer_type | `"binary"` \| `"scale"` \| `"count"` \| `"choice"` |
| mapping | 선택 → 가중치 신호 변환 규칙 |

### Question
| 필드 | 설명 |
|---|---|
| id | |
| kind | `"weight_pairwise"` \| `"indicator"` \| `"sub_question"` |
| payload | weight_pairwise: (factor_a, factor_b)+NL질문+예시2개 / indicator: indicator_id |
| parent_id | 상위 질문 (재귀 분해 시) |
| status | `pending` \| `answered` \| `skipped` |
| info_value | 예상 정보가치 (선택 우선순위) |

### Answer  (사용자의 유일한 입력)
`{ question_id, value, timestamp }`

### WeightState
| 필드 | 설명 |
|---|---|
| prior | `{factor_id → w}` — **LLM이 자율 결정** (슬라이더·컨펌 없음) |
| current | `{factor_id → w}` — 사용자 선택 반영 후 점추정 |
| conflict | stated(LLM 초기) vs revealed(선택) 충돌 신호 |

### Score (option × factor) — **LLM 자율 결정, 사용자 수정 없음**
| 필드 | 설명 |
|---|---|
| value | 정규화된 점수 [0,1] |
| source | `"llm"` (항상) |
| rationale | 근거 (NL, 리포트용) |
| range / confidence | LLM이 표시한 불확실성 (민감도 입력으로 사용) |

### Result
`{ utilities, ranking, sensitivity, robustness("stable"|"close"), report(NL), next_info[] }`

---

## 4. 세션 상태머신

```
SETUP → ANALYZE → QUESTIONING → RESULT
                      ↑_____│ (선택 반영 루프)
```

| 상태 | 무슨 일 | 주체 |
|---|---|---|
| SETUP | 고민 입력 접수 | 사용자 (입력만) |
| ANALYZE | 선택지 정규화 + factor·중요도 결정 + indicator 생성 + **선택지 점수까지 모두 결정** | **LLM 자율** (컨펌 없음) |
| QUESTIONING | pairwise/indicator 질문 루프, 사용자 **선택**, 가중치 정교화 | 사용자 선택 + Compute |
| RESULT | 집계 + 민감도 + 리포트 | Compute + LLM |

> **사용자 상호작용은 SETUP(입력)과 QUESTIONING(선택) 두 군데뿐.** 나머지는 전부 시스템이 자율 처리.

---

## 5. 모듈 설계 (입력 / 출력 / 방법)

### 5.1 입력 & 선택지 정규화 — [LLM 자율]
- **입력:** 사용자 자유서술 고민
- **처리:** 선택지 2~4개 추출/명확화 + 상황 맥락 요약. 모호해도 **사용자에게 묻지 않고** LLM이 가장 합리적으로 정규화. (선택지가 상호배타·잘 정의되도록)
- **출력:** `{ options[], context_summary }`

### 5.2 Factor + 중요도 결정 — [LLM 자율]
- **입력:** options, context
- **처리:** 판단 기준 5~7개와 **각 기준의 초기 중요도(가중치)** 를 LLM이 결정. 각 factor `{name, description, relevance, type, direction, importance}`. 자주 빠뜨리는 메타 factor(기회비용·가역성·시간지평) 자기점검 필수.
- **출력:** factors[] + WeightState.prior
- **사용자 개입 없음.** (편집·컨펌·슬라이더 제거)

### 5.3 Factor → Indicator 분해 (operationalization) — [LLM]
- **대상:** `type == abstract`인 factor만
- **처리:** 그 factor에 대한 사용자 선호를 *간접적으로* 끌어낼 관찰 가능 질문 2~4개 생성 + answer_type + mapping.
  - 예: factor "집중도" → `["방금 문단 다시 읽음?(binary)", "30분간 폰 횟수(count)", "한 문제 평소보다 오래?(scale)"]`
- **출력:** `factor.indicators[]`
- **재귀:** 질문이 또 어려우면 sub_question으로 한 단계 더 분해.

### 5.4 질문 엔진 — [LLM + Compute]  → **§6에서 상세**

### 5.5 선택 → 가중치 정교화 — [Compute]
- **입력:** 사용자의 weight_pairwise / indicator 선택들
- **알고리즘:** **Bradley-Terry 로지스틱 회귀**. 각 선택 "A>B" = 라벨, feature = φ(A)−φ(B), P = σ(θ·Δφ). **LLM이 정한 초기 중요도(prior)** 로 warm-start + **L2 정규화로 prior 쪽 당김** — 즉 LLM 초기값을 사용자 선택이 덮어쓴다.
- **출력:** WeightState.current + **충돌 신호**(LLM 초기 중요도와 사용자 선택이 어긋나면 stated-vs-revealed 충돌)

### 5.6 선택지 점수화 φ — [LLM 자율]
- **처리:** 각 선택지를 factor별로 LLM이 점수화. 근거(rationale)와 불확실성(range/confidence) 포함. **사용자 수정·컨펌 없음.**
- **정규화:** factor별 [0,1] min-max (스케일 다른 factor가 가중합 지배 방지 — 필수)
- **출력:** score_matrix
- ⚠️ 사용자 검증이 없으므로 **이 점수의 품질이 결과를 좌우** → §11.

### 5.7 집계 — [Compute]
- `U(option) = Σ_f w_f · s(option, f)` (WSM = θ·φ)
- 옵션: WPM(가중곱, 스케일 강건), TOPSIS(선택지 多)

### 5.8 민감도 분석 — [Compute]
- **OAT/토네이도:** 각 w_f를 ±흔들어(재정규화) ranking이 뒤집히는 **최소 변화량(flip threshold)** 계산 → driver 순위
- **출력:** factor별 flip_threshold, robustness(`stable`/`close`), top driver
- **가성비 옵션:** 점추정에 작은 노이즈 N회 → flip율

### 5.9 리포트 생성 — [LLM]
- **입력:** utilities, sensitivity, weights, score rationales, 충돌신호
- **출력 (NL 디버깅 리포트):** 최선 선택지 / 왜(top driver) / 뭐 바뀌면 뒤집힘(flip 조건) / 가장 불확실한 가정 / 추가 확인 정보 / (있으면) stated-vs-revealed 충돌

---

## 6. 질문 엔진 상세 (가장 중요)

### 6.1 질문 종류
1. **weight_pairwise** — factor 간 중요도 trade-off ("구현가능성 vs 참신함?")
2. **indicator** — abstract factor에 대한 선호를 끌어내는 구체 질문
3. **sub_question** — 위 둘이 어려울 때 재귀 분해된 더 쉬운 질문

### 6.2 질문 선택 (다음에 뭘 물을까)
결정론(EIG 미사용) → **Query-by-Committee(QBC) 휴리스틱**:
- LLM 초기 가중치 + 지금까지 선택과 모순 없는 **weight 벡터 앙상블 20~50개** 유지
- 후보 질문 중 **앙상블이 가장 갈리는(disagreement 최대)** 걸 우선 선택 = "지금 결과를 뒤집을 수 있는 가장 애매한 trade-off"
- 동시에 아직 측정 안 된 abstract factor의 indicator 질문도 큐에

> *(확장: QBC를 베이지안 EIG/BALD로 교체하면 정식 active learning)*

### 6.3 재귀 분해 (사용자가 선택 못 할 때)
- 트리거: 사용자가 "모르겠다" 표시 or 일관성 없이 선택
- 규칙: 추상 → 구체 indicator, 또는 큰 trade-off → 더 좁은 trade-off
- 구조: `Question.parent_id` 트리. **리프(선택 가능한 단위)까지 내려가 선택 받고 → 상위로 roll-up**해 원 질문 값 추정

```
"커리어 성장 얼마나 중요?"  (선택 어려움)
   └─ "연봉 같으면 성장 빠른+빡센 vs 편한 일?"   (구체 선택)
   └─ "작년에 새 기술 배우려 주말 쓴 적?"        (행동 proxy)
        → 선택들 roll-up → 커리어 가중치 신호
```

### 6.4 선택 → factor 신호 매핑 (indicator roll-up)
- 각 indicator 선택을 mapping 규칙으로 점수화 → 같은 factor의 indicator들 평균/가중합 → 그 factor 가중치 신호
- 이 신호가 §5.5의 BT 회귀로 들어가 가중치를 정교화

### 6.5 종료 조건
- 앙상블 합의(disagreement < 임계) **or** 선두 선택지 K턴 연속 동일 **or** 질문 상한 N
- **피로 방지:** 총 질문 5~8개 상한 (MVP)

---

## 7. LLM 호출 명세 (구조화 I/O, temperature 0)

| 함수 | 입력 | 출력(JSON) | 비고 |
|---|---|---|---|
| `extract_options` | context | `{options[], context_summary}` | 컨펌 없이 자율 정규화 |
| `decide_factors` | options, context | `factors[]{name,desc,type,direction,importance,relevance}` | **factor+중요도 자율 결정 (고난도)** |
| `decompose_indicators` | factor, context | `indicators[]{question,answer_type,mapping}` | |
| `verbalize_pairwise` | factor_a, factor_b | `{question, example_a, example_b}` | |
| `score_options` | options, factors | `{(option,factor) → {value,range,rationale}}` | **점수 자율 결정 (고난도)** |
| `generate_report` | result_data | `{report, next_info[]}` | |

모든 출력은 스키마 검증 후 Store/Compute로. **굵게 표시된 두 호출이 사용자 무검증 → 최고 난도(§11).**

---

## 8. 수학 레이어 (핵심 식)

**가중치 정교화 (Bradley-Terry + L2 prior):**
```
minimize_θ   Σᵢ −log σ(yᵢ · θ·Δφᵢ)  +  λ‖θ − θ_prior‖²
s.t.  θ ≥ 0,  Σ w = 1   (softmax 파라미터화 or projection)
```
θ_prior = LLM이 정한 초기 중요도.

**집계:** `U(o) = θ · φ(o) = Σ_f w_f · s(o,f)`

**민감도(flip threshold):** factor f의 w_f를 키우거나 줄여 ranking이 바뀌는 최소 Δ. 단순 sweep 또는 Triantaphyllou 임계가중치 공식.

---

## 9. API 인터페이스 (설계)

```
POST   /sessions              {context}            → session   (ANALYZE 자동 실행)
GET    /sessions/{id}/next-question                → Question
POST   /sessions/{id}/answers {question_id, value} → {accepted, follow_up?}
GET    /sessions/{id}/result                       → Result
```
> 컨펌/편집 엔드포인트 없음 — 사용자는 선택(answers)만 보낸다.

---

## 10. MVP vs 확장

### MVP (주말~1주 빌드)
- 단일 사용자, factor 5~7, 질문 5~8
- indicator 분해 **1단계**
- **factor·중요도·점수 전부 LLM 자율 결정 (컨펌·편집 없음)**
- 가중치 정교화: BT 로지스틱 회귀 (sklearn)
- 집계: WSM / 민감도: OAT
- LLM 디버깅 리포트

### MVP 제외
- 베이지안 posterior / 몬테카를로
- 팀 conflict 모드
- AHP, EIG/BALD 질문 선택
- 다단계 재귀 분해
- 세션 히스토리 저장

### 확장 로드맵
1. **팀 모드** — 팀원별 가중치 추정 + 충돌이 "아이디어 차이 vs 기준 차이"인지 분석
2. **베이지안 업그레이드** — 파티클 필터 + MC → 승률 + 신뢰구간
3. **EIG/BALD 질문 선택** — QBC를 정식 정보이득으로 교체
4. **다단계 재귀 분해** + WPM/TOPSIS

---

## 11. 정교함이 필요한 곳 (품질 병목)  ★ 핵심

사용자가 무검증이므로 LLM 자율 판단의 품질이 전부다. 단, 모든 LLM 출력이 똑같이 위험한 건 아니다:

| LLM 출력 | 사용자 검증 | 위험도 |
|---|---|---|
| 가중치 (중요도) | **있음** (선택으로 정교화/덮어쓰기) | 낮음 |
| **factor 세트** | 없음 | **높음** |
| **선택지 점수** | 없음 | **높음** |

→ **프롬프트 정교함의 대부분을 `decide_factors`와 `score_options` 두 곳에 투입.** LLM이 여기서 틀리면 잡아줄 사람이 없다.

**`decide_factors` 정교화 기법**
- 도메인별 메타 factor 체크리스트(기회비용·가역성·시간지평·리스크) 강제 점검
- 비슷한 결정 예시 few-shot 제공
- 자기검증 패스: "빠진 중요 기준 없나? 중복되는 기준 없나?" 재확인
- 중요도는 절대 숫자보다 *순위* 위주로 뽑고 prior로만 사용(어차피 선택으로 정교화됨)

**`score_options` 정교화 기법**
- 각 점수에 **근거(rationale) 강제** — 근거 없는 점수 금지
- 불확실하면 단일값 대신 **범위/confidence** 출력 (민감도가 이걸 활용)
- 동일 기준 내 선택지들을 *상대 비교*로 채점(앵커 제공)해 일관성 확보
- 미래 불확실 factor는 보수적으로 + confidence 낮게 표시

---

## 부록: 한눈에 보는 전체 흐름

```
[사용자] 고민 입력
  → [LLM 자율] 선택지 정규화 + factor·중요도 결정 + indicator 생성 + 선택지 점수 결정
  → [사용자] 질문에 선택만  (어려우면 재귀 분해된 쉬운 질문에 선택)
  → [Compute] 선택으로 가중치 정교화 (BT 로지스틱, +충돌 감지)
  → [Compute] WSM 집계 + OAT 민감도
  → [LLM] 디버깅 리포트 (왜 + 뭐가 뒤집나 + 뭘 더 알아야)
```

**역할 분담:** LLM = factor·중요도·점수·질문·리포트 전부 자율 결정 / Compute = 가중치 정교화·집계·민감도 / **사용자 = 질문에 선택만.**
