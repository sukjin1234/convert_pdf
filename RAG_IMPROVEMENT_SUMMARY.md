# Dify 사내 RAG 개선 요약

## 적용 범위

이번 변경은 입학전형 전용 분기나 문서명 하드코딩을 파이프라인 코드에 추가하지 않는다. 실제 문서의 질문과 정답은 교체 가능한 `evaluation/rag_acceptance_cases.json`에만 저장하고, 평가기는 제목·문단·표·리스트·수식이라는 구조 유형과 일반적인 정답 별칭/출처 스키마만 이해한다.

## 기준선

- 로컬 단위/통합 테스트: 182 passed, 1 skipped.
- 운영 Knowledge API: 문서 2개, 세그먼트 1,407개, 기존 메타데이터 감사 이슈 0건. 첫 문서는 610개 세그먼트/22페이지/34개 섹션, 두 번째 문서는 797개 세그먼트/60페이지/81개 섹션이다.
- 운영 Chatflow 단일 추적: 전체 22.313초. `Structured Lookup` 15.744초, 최종 생성 2.337초, 독립질문 재작성 1.335초, Knowledge 검색 0.490초.
- 운영 응답 계약은 로컬 최신 계약보다 오래된 상태다. 구조화 조회 응답은 37,346자였고 compact match/prefilter 진단 필드가 없었다.
- 저장된 8문항 운영 평가 기준선은 답변 통과 3/8(37.5%), 평균 42.405초, 최대 53.315초였다. 당시 문서 ID 기본값이 현재 Knowledge 문서 ID와 달라 문서 필터 실패도 함께 섞여 있었다.

## Before / After

| 구분 | Before | After |
| --- | --- | --- |
| 파싱 검증 | page/section 누락과 제어문자 감사 | 원본 PDF의 구조 앵커와 Knowledge 세그먼트 경계를 비교하고 95% 게이트 적용 |
| 질문셋 | 합성 Markdown 중심 33문항, 운영용 8문항 | 실제 PDF 각 18문항, 총 36문항과 문서/페이지/구간 정답 |
| 정확도 | 단일 pass/fail 또는 문자열 포함 검사 | 답변 내용 정확도와 인용 출처 정확도를 독립 집계 |
| 문서 연결 | 오래된 UUID 기본값 | Knowledge 문서명을 조회해 실행 시점 UUID 자동 연결 |
| 속도 | 전체 시간 위주 | Chatflow 노드/단계별 평균·p95와 로컬 조회 세부 단계 밀리초 기록 |
| 재사용성 | 입학 문서 기본값이 스크립트에 존재 | 문서 경로·이름·질문·정답을 매니페스트 파라미터로 분리 |

## 반영 파일

- `app/pipeline_metrics.py`: 단조 시계 기반 단계 계측기.
- `app/rag.py`: `lookup_matches` 진단에 `stage_latency_ms` 추가.
- `scripts/evaluate_rag_system.py`: Knowledge API 구조 감사, Chatflow 품질/출처/속도 통합 평가기.
- `evaluation/rag_acceptance_cases.json`: 두 실제 PDF를 균등하게 다루는 36문항 골든셋.
- `tests/test_rag_system_evaluation.py`: 도메인 독립 평가 계약과 계측 회귀 테스트.

## 실행 방법

```powershell
python -X utf8 scripts\evaluate_rag_system.py `
  --manifest evaluation\rag_acceptance_cases.json `
  --dataset-id <Knowledge dataset UUID> `
  --concurrency 4 `
  --fail-under-boundary 0.95 `
  --fail-under-answer 0.95 `
  --fail-under-citation 0.95 `
  --max-p95-latency 10
```

다른 도메인은 매니페스트의 `documents`와 `cases`만 교체한다. 각 케이스는 질문, 허용 정답 별칭 그룹, 금지값(선택), 출처 페이지/섹션을 선언한다.

## 배포 후 적용 사항

1. 담당자가 운영 서버에서 `git pull`한다.
2. FastAPI 서비스를 재시작하여 최신 `Query Plan`/`Structured Lookup`/`Verify Answer` 응답 계약을 배포한다.
3. Dify Chatflow의 HTTP 노드가 새 서비스 주소와 기존 `.env` 키를 그대로 사용하도록 확인한다.
4. 기존 Knowledge PDF는 파서 변경이 실제 인덱스에 반영되도록 재처리한다. 이번 작업에서는 운영 인덱스를 변경하지 않았다.
5. 위 통합 평가를 1회 실행한다. 10초 게이트를 넘으면 보고서의 `slowest_nodes`에서 p95가 가장 큰 노드부터 조정한다.

현재 API 키로는 Dify App/Knowledge 호출은 가능하지만 Console workflow 편집 권한은 확인되지 않았다(`workflow_edit_available: false`). 따라서 원격 노드 그래프는 `DIFY_RAG_NODE_SETUP.md`의 변경 계약을 Dify Studio에서 반영해야 한다.

## 2026-08-23 배포 후 보정 라운드

원격 적용 후 36문항을 다시 측정한 결과는 답변 22.22%, 출처 22.22%, 둘 다 통과 11.11%, 전체 p95 58.056초였다. Structured Lookup p95는 20.743초에서 7.314초로 개선됐지만 다음 데이터 흐름 결함이 남아 있었다.

1. 독립형 질문 재작성 노드가 한국어 원문을 영어로 바꿔 표의 행 식별어를 잃었다.
2. 구조화 검색이 열은 찾았지만 행을 구분하지 못해 관련 없는 일정까지 `required_values`로 만들었다.
3. 정확한 구조화 날짜도 Final/Rewrite LLM을 다시 통과하면서 요일이 변조됐다.
4. 평가기는 세그먼트 줄바꿈을 먼저 제거해 page/section 메타데이터를 읽지 못했고, 숫자 `20`을 `2027` 내부에서 찾는 부분 문자열 오탐 가능성이 있었다.

이번 보정은 문서명이나 입학전형 키워드를 추가하지 않고 다음 범용 계약을 반영한다.

- `/query/plan`, `/lookup`에 `retrieval_query`를 추가한다. `query`는 원문, `retrieval_query`는 문맥 보완용 독립 질문이다.
- 검색 점수에는 두 질의를 함께 쓰되 필드·행 선택에 필요한 원문 어휘를 보존한다.
- Merge Evidence 응답에 `has_direct_answer`, `deterministic_answer`, `answer_contract_status`를 추가한다.
- direct answer가 있으면 Dify에서 Final/Verify/Rewrite LLM을 우회하고 정확한 값과 출처를 그대로 출력한다.
- 구조 평가기는 raw segment에서 page/section을 먼저 읽은 뒤 표시용 텍스트만 정리한다.
- 숫자-only 정답 별칭은 숫자 경계를 사용해 더 큰 숫자의 일부와 일치하지 않게 한다.

로컬 검증 결과:

- 전체 테스트: 191 passed, 1 skipped
- 범용 RAG 품질셋: 33/33, 100%
- 수정 평가기로 재측정한 현재 Knowledge 구조 정확도: 93.06%

구조 정확도는 기존 인덱스 기준선이므로 이번 API/Chatflow 보정만 배포해도 자동으로 바뀌지 않는다. 파서 구조 변경을 Knowledge에 반영하는 라운드에서는 두 PDF를 다시 처리·색인한 뒤 측정해야 한다.
