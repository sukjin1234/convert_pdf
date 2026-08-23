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

Console access JWT와 CSRF 토큰이 `.env`에 제공되면 `scripts/dify_workflow_patch.py`로 draft 백업, 그래프 패치, 검증, 게시까지 수행할 수 있다. 토큰과 쿠키는 코드나 보고서에 저장하지 않는다.

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

## 2026-08-23 Console API 노드 반영 및 게시본 검증

Console API로 현재 draft를 백업한 뒤 44개 노드/30개 엣지 그래프를 검증하고 게시했다. 게시된 그래프 해시는 `0b485ed45c5932f8f6c3318c1822b865a2ef1d9d56fb576a8ec68a09b7621cf6`이다.

반영한 범용 데이터 흐름:

1. Start에 선택 입력 `document_id`를 추가하고 Query Plan, Structured Lookup까지 전달한다.
2. 사용자 원문 `query`와 문맥 보완용 `retrieval_query`를 분리한다.
3. Knowledge 검색과 Structured Lookup을 병렬 실행한다.
4. Structured Lookup의 완전한 단일값 답변은 `IF Deterministic Direct Answer`에서 Final/Verify/Rewrite LLM을 우회한다.
5. 직접답변이 불가능한 질문만 기존 생성·검증·재작성 경로로 보낸다.

게시본 단건 검증에서 배포 계약은 통과했고, 문서 범위를 지정한 `수시1차 원서접수 기간`은 정답과 단일 출처가 일치했다. 해당 실행의 전체 스트림은 14.137초였고 주요 노드는 질문 재작성 5.634초, Structured Lookup 0.613초, 최종 생성 3.557초였다. 문서 범위를 지정하지 않은 동일 질의는 서로 다른 문서의 동명 일정이 섞여 오답이므로, 호출자가 선택한 문서의 RAG 아티팩트 ID를 `inputs.document_id`로 전달해야 한다.

문서별 아티팩트 ID를 명시해 실행한 36문항 게시본 결과:

- 요청 성공률: 100% (36/36)
- 답변 정확도: 19.44% (7/36)
- 출처 정확도: 30.56% (11/36)
- 답변과 출처 모두 통과: 13.89% (5/36)
- 평균/p50/p95/최대: 18.064/17.530/29.284/30.715초
- 노드 p95: 분류 4.990초, 생성 3.731초, 검색 1.330초, 검증 0.197초

직접답변 분기는 일정형 문항에서 4.469~8.053초 실행을 만들었지만, 현재 운영 FastAPI는 `기간`, `값`, `몇`, `amount`, `period` 같은 값 유형 단어를 표 행 식별자로 강제해 올바른 행을 버리는 문제가 남아 있다. 로컬 수정은 이 단어들을 정확 일치하는 범용 scalar 속성으로 분류하고, 이벤트·항목명 같은 실제 행 식별어만 필터에 사용한다. 입학전형 키워드나 문서명은 사용하지 않는다.

평가 도구도 Knowledge 문서 UUID와 RAG 아티팩트 UUID를 구분한다. 세그먼트의 `[document_id: ...]`를 자동 해석하며, 운영 메타데이터가 오래된 경우 다음처럼 매니페스트 문서키별 ID를 명시할 수 있다.

```powershell
python -X utf8 scripts\evaluate_rag_system.py `
  --manifest evaluation\rag_acceptance_cases.json `
  --skip-structure `
  --document-id <manifest-key>=<artifact-id> `
  --concurrency 4
```

현재 Knowledge API 키가 조회하는 두 데이터셋은 비어 있고, Chatflow가 연결한 `ipsi` 데이터셋은 Console 세션에서만 확인된다. 또한 Knowledge 세그먼트의 `[document_id]` 값과 현재 구조화 저장소 ID가 일치하지 않는다. 따라서 다음 배포에서는 FastAPI 재시작 후 아티팩트/Knowledge 동기화를 다시 수행하고, UUID 바인딩을 확인한 뒤 36문항을 재측정해야 한다.

이번 로컬 검증은 `200 passed, 1 skipped`를 기준으로 한다. 운영에 아직 반영되지 않은 코드는 범용 scalar 행 선택 수정, 평가기의 아티팩트 ID 바인딩, Console 세션 진단 지원이다.

## 2026-08-23 구조 복구·빠른 경로 최종 로컬 검증

원격 재기동 후 단건 실행을 다시 추적한 결과 전체 16.551초 중 `Rewrite Standalone Query`가 5.723초, Structured Lookup이 4.171초, Final LLM이 3.894초였다. 로컬 검색 p95는 1초 미만이므로 주 병목은 검색 엔진이 아니라 모든 독립 질문에 강제된 재작성 LLM이었다. 또한 원격 Structured Lookup 응답에 최신 scalar 수정 결과가 없어서 실행 중인 코드와 저장소 버전을 구분할 수 없었다.

이번 변경은 다음을 반영한다.

- Chatflow에서 `Rewrite Standalone Query` LLM 노드를 제거하고 `Session Follow-up Guard`가 짧은 후속 질문에만 이전 질문을 결정적으로 결합한다. 일반 질문은 원문 그대로 Query Plan으로 전달한다.
- `/health`와 Structured Lookup 진단에 `pipeline_revision=rag-20260823-fastpath-v2`를 넣어 실제 배포 버전을 추적한다.
- OpenDataLoader의 제목·표·읽기 순서를 기본으로 유지하면서, 같은 페이지에서 누락된 네이티브 PDF 텍스트 블록만 `Text-layer recovery` 구간에 병합한다. 파일명이나 입학 용어는 사용하지 않는다.
- PDF 목록 렌더러가 붙인 `1. Ⅴ.` 형태의 중복 번호를 구조적으로 해석해 장/절 계층을 복구한다.
- 표의 일반 열 이름, 섹션 제목, 페이지 내 인접 캡션과 값의 관계를 분리한다. 이웃 문맥은 검색에는 쓰지만 행에 명시되지 않은 숫자의 직접답변 생성에는 쓰지 않는다.
- 파일명(`*.pdf`)을 표의 필수 행 엔터티로 오인하지 않으며, 복합 질문은 직접답변 근거 행과 같은 표/섹션의 형제 근거를 앞쪽 문맥에 함께 배치한다.
- 평가기의 수식 기호 별칭을 문자 그대로 검증해 `÷`, `/` 같은 기호가 정규화 과정에서 사라지지 않게 한다.

실제 두 PDF와 36문항 골든셋을 새 파서로 로컬 평가한 결과:

| 지표 | 결과 | 목표 |
| --- | ---: | ---: |
| 구조 경계 정확도 | 97.73% | 95% |
| 원문 구조 앵커 회수율 | 95.98% | 참고 |
| 페이지 커버리지 | 99.18% | 참고 |
| 답변 근거 내용 회수 | 100.00% (36/36) | 95% |
| 출처 구간 Recall@8 | 100.00% (36/36) | 95% |
| 답변 근거+출처 동시 통과 | 100.00% (36/36) | 95% |
| 로컬 분류+검색 p95 | 1.170초 | 참고 |
| 전체 회귀 테스트 | 204 passed, 1 skipped | 전체 통과 |

차트 축과 값이 별도 그래픽 객체인 구간도 누락된 네이티브 텍스트 블록을 페이지 단위로 복구해 근거 회수 게이트를 통과했다. 숫자형 제목을 페이지 번호로 제거하지 않도록 요소 타입을 보존하며, 캐시 스키마를 `pdf-cache-v6-typed-page-noise`로 올려 이전 변환 결과가 재사용되지 않게 했다.

위 정확도는 최종 LLM 생성 전의 구조·근거·출처 검색 게이트다. 게시된 Chatflow의 최종 답변 내용/인용 정확도와 전체 p95 10초는 빠른 경로 그래프 게시 후 별도로 최종 판정한다.

재사용 가능한 로컬 검증 명령:

```powershell
python scripts\evaluate_local_pdf_rag.py `
  --manifest evaluation\rag_acceptance_cases.json `
  --out .runtime\local_pdf_rag_evaluation.json
```

현재 Console API 세션 JWT는 만료되어 빠른 경로 그래프의 게시와 게시 후 10초 게이트 측정은 대기 상태다. `.env`의 Console 세션 값을 새로 저장한 뒤 다음 명령으로 백업·검증·게시하며, 게시 후 36문항 Chatflow 평가에서 답변/출처 95%와 p95 10초를 최종 판정한다.

```powershell
python scripts\dify_workflow_patch.py --apply --publish
```
