# LLM Behavioral Study on Constraint-Aware GridWorld (v2)

- 실험 목적: "환경은 Rule, 판단은 LLM"으로 역할을 고정한 상태에서, 동일 Qwen 모델이 상황마다 어떻게 판단하고 코드(패치)를 만들어내는지 관찰.
- "매번 같은 결과"가 아니라 "다른 상황에서 각자 이유 있는 반응"을 보는 것이 목표.
- 3개의 독립 변동원을 함께 도입:
  1. **시나리오 perturbation**: run별 seed로 obstacle/keepout 위치 ±1 셀 shift (`RP seed = 7000+run_idx`).
  2. **A\* tiebreak 난수화**: 동률 f-score에 MOVE_COST의 1% 이하 난수 tiebreak (`seed = 9000+run_idx`).
  3. **실제 LLM 호출**: `llm_patcher.py`가 env-var 스텁이 아니라 Qwen2.5-Coder-7B-Instruct를 `temperature=0.7, top_p=0.95`로 호출, `seed = 1000+run_idx`. run마다 다른 JSON을 생성.
- 실험 규모: 3 시나리오 × 2 patcher(deterministic / llm_qwen) × 10 run = **60 run**.
- 공개 아티팩트: 이 폴더의 집계 보고서와 JSON. 원시 모델 출력과 내부 실행 산출물은 공개본에서 제외했습니다.

---

## 1. 총량 지표

| scenario | patcher | runs | improved_rate | keepout_delta (mean±std) | unique_patches | LLM fallback |
|---|---|---|---|---|---|---|
| battery_tight | deterministic | 10 | 0.50 | +0.80±0.87 | 2/10 | 0 |
| battery_tight | llm_qwen | 10 | 0.50 | +0.60±1.20 | **9/10** | 1 |
| keepout_01 | deterministic | 10 | 0.10 | +0.10±0.30 | 2/10 | 0 |
| keepout_01 | llm_qwen | 10 | 0.10 | +0.10±0.30 | 3/10 | 2 |
| obstacle_dense | deterministic | 10 | 0.50 | +1.40±1.74 | 3/10 | 0 |
| obstacle_dense | llm_qwen | 10 | 0.30 | +1.10±1.81 | **6/10** | 1 |

- `improved_rate`는 거의 비슷하지만, **unique_patches(고유 패치 서명 수)가 결정론적 패치의 2~3/10에 비해 LLM은 9/10, 6/10까지 증가**. 같은 종류 문제를 같은 방식이 아니라 매번 다른 조합/값으로 접근한다는 직접 증거.
- `LLM fallback = 4/30`: Qwen이 JSON 파싱 불가한 답을 생성한 run. 이는 "코드 기반 LLM의 취약점"이 실제로 드러나는 지점. validator가 있어서 안전하게 deterministic으로 폴백.

## 2. 동일 상수에 대한 값 일관성

같은 상수를 여러 번 선택했을 때의 분산(stdev) — 0에 가까울수록 LLM이 일관된 값을 선택한 것.

**deterministic (대조군)**: 모든 run에서 KEEP_OUT_PENALTY=300, LOW_BATTERY_PENALTY=80, RISK_WEIGHT=1.8로 고정. stdev=0.

**llm_qwen — battery_tight**
| 상수 | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| LOW_BATTERY_PENALTY | 6 | 446.67 | 219.60 | 80 | 800 |
| RISK_WEIGHT | 6 | 2.17 | 0.37 | 2.0 | 3.0 |
| KEEP_OUT_PENALTY | 5 | 101.00 | 103.94 | 5 | 300 |
| REPLAN_THRESHOLD | 3 | 2.67 | 0.94 | 2 | 4 |

**llm_qwen — obstacle_dense**
| 상수 | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| KEEP_OUT_PENALTY | 6 | 44.17 | 42.57 | 5 | 100 |
| RISK_WEIGHT | 4 | 2.00 | 0.00 | 2.0 | 2.0 |
| OBSTACLE_NEAR_PENALTY | 2 | 10.00 | 5.00 | 5 | 15 |

해석:
- **RISK_WEIGHT=2.0은 사실상 고정값(stdev≈0)** → LLM이 일관된 안전 정책을 학습했음.
- **KEEP_OUT_PENALTY는 5~300 사이로 매우 다양** → 같은 문제에 다른 강도의 해결책을 선택. "안전영역 진입을 억제한다"는 방향성은 일관, 크기는 판단마다 다름.
- **LOW_BATTERY_PENALTY도 80~800 range** → 배터리 시나리오에서 LLM이 실험적으로 다양한 penalty 세기를 시도.

## 3. 상황 판단(situation awareness) 증거

`scenario_keepout_01`은 perturbation 후에도 baseline이 성공(keepout_violations=0)하는 run이 대부분. LLM이 "상황이 이미 괜찮다"를 판단할 수 있다면 "변경 없음"을 반환해야 함.

결과 — llm_qwen, keepout_01 10 run 중:
- run 01/03/04/06/08/09: baseline 통과 → `changes=[]` (no-op), `expected_effect="…baseline performance is acceptable"`
- run 00: baseline 통과지만 패치 제안 (KEEP_OUT_PENALTY=5, LOW_BATTERY_PENALTY=50). 사유는 있지만 상황에는 과잉.
- run 07: baseline에서 1 violation 발생 → 정확히 보수적 패치(KEEP_OUT_PENALTY=5, RISK_WEIGHT=2.0) 제안 → `1 → 0`으로 성공.
- run 02/05: Qwen이 JSON 파싱 불가 응답을 내어 fallback.

→ **같은 모델이 "통과 상황"에선 대부분 no-op, "실패 상황"에선 개입**. 상황-판단 능력이 확인됨. 다만 run 00/01의 과잉 개입/무변경 혼재에서 보듯 경계면에서 흔들림.

## 4. 같은 문제에 다른 처방 — 질적 예시

`scenario_obstacle_dense`는 perturbation에 따라 baseline keepout violation 수가 run마다 0~5로 다양. LLM이 같은 "keepout 위반 발생" 상황에 내리는 처방:

| run | baseline kv | LLM 처방 | patched kv |
|---|---|---|---|
| 01 | 4 | KEEP_OUT_PENALTY=100 + RISK_WEIGHT=2.0 | 0 |
| 03 | 2 | KEEP_OUT_PENALTY=50 + RISK_WEIGHT=2.0 | 0 |
| 04 | 5 | KEEP_OUT_PENALTY=5 + RISK_WEIGHT=2.0 | 0 |
| 05 | 0 | KEEP_OUT_PENALTY=100 + OBSTACLE_NEAR_PENALTY=15 + LOW_BATTERY_PENALTY=150 | 0 |
| 07 | 1 | KEEP_OUT_PENALTY=5 + RISK_WEIGHT=2.0 | 1 (개선 실패) |
| 00 | 0 | KEEP_OUT_PENALTY=5 + OBSTACLE_NEAR_PENALTY=5 + LOW_BATTERY_PENALTY=10 | 0 |

관찰:
- **공통 직관**: "KEEP_OUT_PENALTY ↑ + RISK_WEIGHT ↑"라는 방향은 4/6 run에서 공유. 이것이 "이유의 일관성".
- **값 선택은 상황에 의존하지 않고 흔들림**: run 04(kv=5로 가장 심각)는 오히려 KEEP_OUT_PENALTY=5로 가장 약함. LLM이 failure severity를 penalty 크기에 체계적으로 매핑하지 못하는 경향.
- **과잉-반응**: run 05는 baseline이 이미 통과(kv=0)한 상황인데도 3개 상수를 모두 건드림. 해석: failure_report의 `suggested_focus="No change needed."`에도 불구하고 "혹시 모르니" 개선을 시도.
- **드문 regression**: `battery_tight run05`는 kv 0→2로 악화 (LOW_BATTERY_PENALTY=500 + REPLAN_THRESHOLD=2 조합이 경로를 위험구역으로 몰아넣음). LLM 개입이 오히려 해가 된 run도 존재.

## 5. A\* tiebreak + perturbation + LLM 변동성의 상호작용

각 run의 grid figure(`before_after_paths.png`)에서 확인:
- perturbation으로 같은 시나리오의 obstacle/keepout 위치가 매번 다름 → baseline 통과/실패 분포가 스스로 자연스러워짐.
- A\* tiebreak 난수화로 동률 비용 경로에서 매번 다른 cell 시퀀스 선택 → 시각적으로도 run별 경로 다양성 확인 가능.
- 결과적으로 **"동일 시나리오 ID라도 실제 실험 조건이 매 run 다르다"**. 고정 rule로 고정 결과가 나오는 상태를 제거함.

## 6. 연구 결론 요약

1. 환경-규칙/의사결정-LLM 분리 구조에서 Qwen2.5-Coder-7B는 **방향성(safety↑) 수준의 일관성**은 유지하지만, **구체 수치 선택은 run마다 상당히 다름** (KEEP_OUT_PENALTY 5~300, LOW_BATTERY_PENALTY 80~800).
2. `no-op`과 `patch` 사이의 **상황 판단은 대체로 올바름** (baseline 통과 시 no-op 6/10). 경계 케이스에선 흔들림.
3. **JSON 스키마 생성 신뢰성**: 30회 중 4회 파싱 실패. 실제 UGV 시스템에 배치할 경우 validator + fallback이 필수.
4. 결정론적 patcher 대비 improvement rate는 비슷(0.30~0.50)하지만, **정책 다양성(unique_patches 6~9/10)은 압도적으로 높음**. 이는 "LLM이 같은 모델이라도 상황별로 다른 코드를 짠다"는 사실의 직접 증거.
5. 간헐적 regression (1/30) 사례도 확인됨. LLM-in-the-loop 시스템에서 **사후 검증 루프(improved 여부 체크 → rollback)의 필요성**을 뒷받침.

## 7. 재현 명령

```bash
# 전체 sweep (3 scenarios × 2 patchers × 10 runs, 모든 변동원 on)
cd feedback-driven-code-revision
PYTHONPATH=src python scripts/run_sweep.py \
  --runs 10 --perturb --astar-tiebreak \
  --patchers deterministic llm \
  --llm-backend qwen \
  --llm-model-path /path/to/Qwen2.5-Coder-7B-Instruct \
  --llm-temperature 0.7 \
  --artifacts-root artifacts/toy_grid_patch_v2

# 집계 및 일관성 리포트 생성
PYTHONPATH=src python scripts/make_report.py --root artifacts/toy_grid_patch_v2
```

원시 LLM 응답과 개별 실행 trace는 라이선스 및 공개 범위를 보수적으로 적용해 이 저장소에 포함하지 않았습니다.
