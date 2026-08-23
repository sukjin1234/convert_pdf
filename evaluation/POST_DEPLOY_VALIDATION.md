# Dify RAG 배포 후 검증 결과

검증일: 2026-08-23 (Asia/Seoul)  
검증 커밋: `5f096db`  
대상: Dify App API 및 Knowledge API에 배포된 두 PDF, 36개 수용 질문

## 판정 요약

| 항목 | 목표 | 배포 후 결과 | 판정 |
| --- | ---: | ---: | --- |
| 문서 구조 경계 정확도 | 95% 이상 | 93.06% | 실패 |
| 답변 정확도 | 95% 이상 | 22.22% (8/36) | 실패 |
| 출처 정확도 | 95% 이상 | 22.22% (8/36) | 실패 |
| 답변과 출처 모두 정확 | 95% 이상 | 11.11% (4/36) | 실패 |
| 분류~최종 응답 p95 | 10초 이하 | 58.056초 | 실패 |

API 호출 자체는 36/36 성공했다. 정확한 답변과 출처를 모두 충족한 케이스는 다음 네 개다.

- `g27_mechanical_rural_quota`
- `g27_first_application_period`
- `g26_spatial_project_ministry`
- `g26_dorm_room_counts`

원시 결과는 `.runtime/rag_acceptance_post_deploy.json`, 사람이 읽는 자동 생성 보고서는 `evaluation/rag_acceptance_post_deploy.md`에 있다.

## 구조 정확도

| 문서 | 경계 정확도 | 앵커 회수 | 페이지 커버리지 | 유효 세그먼트 |
| --- | ---: | ---: | ---: | ---: |
| 2027 모집요강 | 99.83% | 283/284 | 22/22 | 610/610 |
| `document.pdf` | 89.31% | 411/512 | 59/61 | 797/797 |
| 가중 전체 | 93.06% | 694/796 | - | 1,407/1,407 |

첫 문서는 목표를 넘었다. 두 번째 문서의 실패는 특정 입학 용어가 아니라 카드형 숫자, 차트의 축-값 연결, 다단 레이아웃, 시각 제목과 본문 사이의 연관 보존 문제에 집중된다. 특히 차트 값이 연도와 분리되는 실제 구조 손실과, PDF 시각 텍스트와 Knowledge 세그먼트 표현이 달라 평가 앵커가 일치하지 않는 경우가 함께 있다. 따라서 특정 문서 키워드를 추가해 점수만 높이는 방식은 적용하지 않는다.

주의: 현재 `scripts/evaluate_rag_system.py`의 `to_segment_view`는 페이지·섹션 메타데이터를 추출하기 전에 줄바꿈을 제거한다. 위 수치는 원본 세그먼트에서 메타데이터를 먼저 읽도록 런타임에서 보정해 측정한 값이다. 저장된 평가 코드의 0% 구조 점수는 실제 Knowledge 상태가 아니라 평가기 순서 오류다.

## 응답 속도

| 지표 | 배포 전 | 배포 후 | 변화 |
| --- | ---: | ---: | ---: |
| 전체 평균 | 76.668초 | 45.226초 | 41.0% 단축 |
| 전체 p95 | 104.519초 | 58.056초 | 44.5% 단축 |
| 전체 최대 | 118.940초 | 58.356초 | 50.9% 단축 |
| Structured Lookup 평균 | 11.473초 | 4.698초 | 59.1% 단축 |
| Structured Lookup p95 | 20.743초 | 7.314초 | 64.7% 단축 |
| 구조화 응답 평균 크기 | 27,606자 | 21,754자 | 21.2% 감소 |

배포 코드의 조회 최적화는 실제 효과가 있었다. 다만 노드 실행시간 합계는 평균 10.675초인 반면 요청 전체에서 설명되지 않는 대기시간이 평균 34.551초, p95 45.117초다. 동시 요청 시 Dify 실행 대기열 또는 스트리밍 전달 지연이 전체 p95의 가장 큰 비중이다.

노드 자체의 주요 지연은 다음과 같다.

- Structured Lookup: 평균 4.698초, p95 7.314초
- Rewrite Answer LLM: 평균 2.735초, p95 4.508초
- Final Answer: 평균 2.542초, p95 4.076초
- Rewrite Standalone Query: 평균 0.987초, p95 1.521초
- Knowledge Retrieval: 평균 0.513초, p95 0.602초

## 품질 실패의 확인된 원인

1. App API의 `inputs.document_id`가 Chatflow Start 노드에 등록되어 있지 않아 완전히 버려진다. 스트리밍 `workflow_started.data.inputs`에 `document_id`가 없고 Query Plan도 `null`을 반환했다. 이 상태에서는 두 문서 및 버전이 섞일 수 있다.
2. Rewrite Standalone Query가 한국어 원문의 핵심 행 식별어를 영어로 바꾼다. 실제 요청에서 `원서접수`가 `application submissions`로 바뀌었고 `/lookup`에는 한국어 원문이 전달되지 않았다. 구조화 표 검색은 열 `수시1차`는 찾았지만 행 `원서접수`를 구분하지 못해 서류·면접·발표 일정까지 완전값으로 만들었다.
3. 구조화 완전값이 과다해지면 verifier는 간결하고 올바른 답변도 `missing_direct_answer_value`로 실패 처리하고 Rewrite Answer LLM을 실행한다.
4. Final/Rewrite LLM이 근거의 요일 표기 `(월)`, `(수)`를 자연어로 바꾸면서 각각 수요일, 금요일처럼 잘못 생성한 실제 사례가 있다. 구조화 값이 확보된 질문에도 생성 모델을 다시 통과시키는 것이 정확도와 속도를 동시에 악화시킨다.
5. 평균 구조화 응답이 21,754자이고 p95는 59,442자다. 최종 답에 필요하지 않은 인접 행과 supporting context가 여전히 크다.

## 운영 Chatflow에서 필요한 변경

현재 App API는 읽기/실행만 가능하고 workflow 편집 권한이 없어 자동 반영하지 못했다. 아래 변경은 Dify Studio에서 한 번에 반영한 뒤 새 버전을 게시해야 한다.

1. Start 노드에 숨김 Text 입력 `document_id`를 추가한다. 특정 문서 모드에서는 호출자가 Knowledge document id를 전달하고, Query Plan·Build Lookup Body·Structured Lookup까지 같은 값을 유지한다.
2. `/lookup`의 주 질의는 항상 원본 `sys.query`를 사용한다. Rewrite Standalone 결과는 의미 검색 확장용 `sub_queries`로만 사용하고 원문의 언어와 어휘를 대체하지 않는다.
3. `answer_contract.status == direct_answer`이면 Final Answer LLM, Verify, Rewrite Answer LLM을 우회한다. `direct_answer`를 그대로 사용하고 `source_summary`만 결정적으로 붙이는 Code/Template 경로로 보낸다.
4. 생성 경로가 필요한 서술형 질문에만 Final Answer → Verify → 조건부 Rewrite를 유지한다. verifier의 `complete_values`는 구조화 검색이 선택한 대상 행/필드만 포함해야 한다.
5. Structured Lookup 응답은 선택된 `answer_items`, 최소 출처, 짧은 근거만 다음 노드로 전달하고 전체 matches/supporting context는 디버그 모드에서만 포함한다.
6. 동시 실행 제한과 작업자 수를 점검한다. 현재 동시 4개 평가에서 노드 밖 대기 p95가 45.117초이므로 애플리케이션 노드 최적화만으로 전체 p95 10초를 달성할 수 없다.

## 재검증 조건

다음 배포 판정은 아래 항목을 모두 확인해야 한다.

- 36개 질문의 답변 정확도와 출처 정확도가 각각 95% 이상
- 분류 시작부터 최종 응답까지 p95 10초 이하
- 특정 문서 호출에서 `document_id_in_workflow_inputs=true`
- Query Plan과 `/lookup` 요청에 같은 `document_id` 존재
- 구조화 질문에서 `/lookup.query`가 원본 사용자 질문과 동일
- direct answer 경로에서 Final/Verify/Rewrite LLM 미실행
- `document.pdf`의 카드·차트·다단 레이아웃을 포함한 구조 경계 정확도 95% 이상

