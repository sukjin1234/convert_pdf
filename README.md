# Dify OpenDataLoader RAG API

OpenDataLoader로 PDF를 Markdown으로 변환하고, Dify RAG에 바로 넣을 수 있도록 표 행 확장, chunk metadata, structured records, lookup, answer verification을 제공하는 FastAPI 서버입니다.

핵심 엔드포인트:

```text
POST /convert      기존 호환용 PDF -> Markdown
POST /convert/rag  PDF -> Dify 저장용 Markdown + chunks + records
POST /rag/ingest-markdown Markdown -> RAG artifact 저장
POST /query/plan   질문 유형/키워드/하위질문/검색어 확장
POST /lookup       구조화 레코드 exact lookup + evidence packing
POST /answer/verify 답변 숫자/날짜/식별자 근거 검증
POST /chatflow/debug Dify Chatflow 노드 흐름 로컬 재현
POST /eval/log    Dify 실행 trace JSONL 저장
GET  /eval/logs   저장된 평가 로그 목록/상세 조회
GET  /rag/documents 저장된 RAG artifact 목록
GET  /health
```

## 실행

```powershell
pip install -r requirements.txt
opendataloader-pdf-hybrid --port 5002 --ocr-lang "ko,en" --enrich-picture-description
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

스캔 PDF가 많으면 OCR을 강제합니다.

```powershell
opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang "ko,en" --enrich-picture-description
```

## Dify 연결

전체 노드 구성은 [DIFY_RAG_NODE_SETUP.md](DIFY_RAG_NODE_SETUP.md)를 봅니다.

Knowledge Pipeline에서는 `/convert/rag`를 호출하고, 응답의 `dify_markdown`만 Knowledge Base에 저장합니다.

`/convert/rag`는 multipart form-data입니다. 이 노드에서는 `Content-Type` 헤더를 직접 넣지 않습니다.

```text
Method: POST
URL: http://<fastapi-host>:8000/convert/rag
Body Type: form-data
pdf          File  {{File.file}}
document_id  Text  {{File.file.related_id}}
file_name    Text  {{File.file.name}}
```

Chatflow의 JSON/Raw HTTP 노드에서는 다음 헤더만 넣습니다.

```text
Content-Type: application/json
```

`/lookup` 응답의 `context`는 Dify LLM에 바로 넣는 evidence 본문입니다. 목록/표 질문에서는 `context` 맨 위에 `[Direct Answer - complete structured result]`와 `complete_values`가 들어가므로, LLM은 이 값을 빠뜨리지 않고 답해야 합니다. 표 행의 `♣`, `†`, `◈` 같은 표시와 `표시/indicates/means` 범례도 자동으로 연결해 행 필드에 반영합니다. 범례가 표 바로 아래가 아니라 다음 페이지나 다른 섹션에 있어도 문서 안에서 해당 기호의 의미가 하나로 명확하면 연결됩니다. 추가로 `direct_answer`, `answer_items`, `answer_field`, `filter_terms`, `evidence_items`, `diagnostics`, `answer_style`, `sub_queries`가 들어오므로 디버깅과 프롬프트 분기에 사용할 수 있습니다.

## 수동 테스트

```powershell
curl.exe http://127.0.0.1:8000/health

curl.exe -F "pdf=@2026학년도 입학전형 신입생 모집요강.pdf" -F "document_id=sample_2026" -F "file_name=2026_admission.pdf" http://127.0.0.1:8000/convert/rag
```

조회 테스트:

```powershell
curl.exe -H "Content-Type: application/json" -d "{\"query\":\"컴퓨터정보공학과 모집인원 알려줘\"}" http://127.0.0.1:8000/query/plan
```

Dify Chatflow 디버깅:

```powershell
curl.exe -H "Content-Type: application/json" -d "{\"query\":\"전공심화 개설 학과 알려줘\",\"document_id\":\"sample_2026\",\"knowledge_result\":[],\"answer\":\"제공된 문서에는 충분한 근거가 없습니다.\"}" http://127.0.0.1:8000/chatflow/debug
```

`/chatflow/debug`는 `Query Plan -> Structured Lookup -> Merge Evidence -> Verify` 흐름을 한 번에 재현한다. `Knowledge Retrieval`이 비어 있어도 `merge_evidence.evidence_context`에 `Structured Lookup - authoritative exact evidence`가 있고 `draft_verification.valid=true`이면 API 쪽 근거 생성은 정상이다. 이때 Dify 최종 답변이 실패하면 `Merge Evidence` 코드, `evidence_priority` 출력 변수, Final Answer LLM 프롬프트 연결을 먼저 확인한다.

평가 로그 저장:

```powershell
curl.exe -H "Content-Type: application/json" -d "{\"query\":\"전공심화 개설 학과 알려줘\",\"document_id\":\"sample_2026\",\"final_answer\":\"...\",\"verify_result\":{\"valid\":true}}" http://127.0.0.1:8000/eval/log

curl.exe http://127.0.0.1:8000/eval/logs?limit=20
```

`/eval/log`는 Dify HTTP 노드에서 질문, 각 노드 입력/출력, 최종 답변, 검증 결과를 그대로 받아 JSONL과 run_id별 JSON 파일로 저장한다.

저장된 Dify 실행 로그 분석:

```powershell
python scripts/analyze_eval_logs.py --log-dir dify-rag-eval-logs --limit 20

python scripts/analyze_eval_logs.py --log-dir dify-rag-eval-logs --json
```

이 스크립트는 `document_id` 누락, `direct_answer` 부재, `Verify Answer.valid=false`, `valid=true`인데 구조화 직접답변이 없는 케이스를 요약한다. 새 PDF를 재적재한 뒤 같은 질문을 다시 실행하고 이 스크립트의 `missing_document_id`와 `missing_direct_answer` 수가 줄어드는지 확인한다.

## 환경 변수

```text
ODL_HYBRID_URL                  기본 http://localhost:5002
ODL_HYBRID_MODE                 기본 auto, 허용값 auto/full
ODL_CONVERSION_TIMEOUT_SECONDS  기본 360
ODL_HYBRID_TIMEOUT_MS           기본 300000
ODL_MAX_PDF_BYTES               기본 83886080
ODL_TABLE_METHOD                기본 cluster
ODL_USE_STRUCT_TREE             기본 false
RAG_STORE_DIR                   기본 OS temp/dify-rag-store
EVAL_LOG_DIR                    기본 OS temp/dify-rag-eval-logs
```

`RAG_STORE_DIR`에는 `/convert/rag` 결과 artifact가 JSON으로 저장됩니다. API 서버가 재시작되어도 `/lookup`이 이전 records를 다시 로드할 수 있습니다.
`EVAL_LOG_DIR`에는 `/eval/log` 실행 기록이 날짜별 JSONL과 run_id별 JSON 파일로 저장됩니다.

## 테스트

기본 단위 테스트:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

RAG 품질 평가:

```powershell
python scripts/eval_rag_quality.py --cases evaluation/rag_quality_cases.json --fail-under 0.95
```

실행 중인 FastAPI 서버를 통한 HTTP 품질 평가:

```powershell
python scripts/eval_rag_http.py --base-url http://127.0.0.1:8000 --cases evaluation/rag_quality_cases.json --fail-under 0.95
```

평가셋은 입학, 기술 운영, 정책/보안, 제품/요금 도메인을 포함합니다. 핵심 지표는 `evidence_recall`, `exact_value_match`, `query_type_accuracy`, `unsupported_guard`입니다.

샘플 PDF 통합 테스트:

```powershell
$env:RUN_SAMPLE_PDF_TEST="1"
python -m unittest tests.test_sample_pdf_contract -v
```

통합 테스트는 `opendataloader-pdf` CLI와 `opendataloader-pdf-hybrid` 백엔드가 실행 가능한 환경에서만 돌립니다.
