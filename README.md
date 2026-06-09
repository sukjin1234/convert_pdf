# Dify OpenDataLoader RAG API

OpenDataLoader로 PDF를 Markdown으로 변환하고, Dify RAG에 바로 넣을 수 있도록 표 행 확장, chunk metadata, structured records, lookup, answer verification을 제공하는 FastAPI 서버입니다.

핵심 엔드포인트:

```text
POST /convert      기존 호환용 PDF -> Markdown
POST /convert/rag  PDF -> Dify 저장용 Markdown + chunks + records
POST /query/plan   질문 유형/키워드/하위질문/검색어 확장
POST /lookup       구조화 레코드 exact lookup + evidence packing
POST /answer/verify 답변 숫자/날짜/식별자 근거 검증
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

`/lookup` 응답의 `context`는 Dify LLM에 바로 넣는 evidence 본문입니다. 추가로 `evidence_items`, `diagnostics`, `answer_style`, `sub_queries`가 들어오므로 디버깅과 프롬프트 분기에 사용할 수 있습니다.

## 수동 테스트

```powershell
curl.exe http://127.0.0.1:8000/health

curl.exe -F "pdf=@2026학년도 입학전형 신입생 모집요강.pdf" -F "document_id=sample_2026" -F "file_name=2026_admission.pdf" http://127.0.0.1:8000/convert/rag
```

조회 테스트:

```powershell
curl.exe -H "Content-Type: application/json" -d "{\"query\":\"컴퓨터정보공학과 모집인원 알려줘\"}" http://127.0.0.1:8000/query/plan
```

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
```

`RAG_STORE_DIR`에는 `/convert/rag` 결과 artifact가 JSON으로 저장됩니다. API 서버가 재시작되어도 `/lookup`이 이전 records를 다시 로드할 수 있습니다.

## 테스트

기본 단위 테스트:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

샘플 PDF 통합 테스트:

```powershell
$env:RUN_SAMPLE_PDF_TEST="1"
python -m unittest tests.test_sample_pdf_contract -v
```

통합 테스트는 `opendataloader-pdf` CLI와 `opendataloader-pdf-hybrid` 백엔드가 실행 가능한 환경에서만 돌립니다.
