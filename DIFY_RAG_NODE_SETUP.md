# Dify RAG 노드 구성 가이드

이 구성은 Dify를 UI, Knowledge Pipeline, Knowledge Retrieval에 쓰고, 정확도를 좌우하는 PDF 정규화, 표 행 확장, 구조화 조회, 답변 검증은 외부 FastAPI가 맡는 방식이다

목표 흐름:

```text
PDF
-> FastAPI /convert/rag
-> Dify Knowledge Base에 dify_markdown 저장
-> Chatflow에서 /query/plan + /lookup + Knowledge Retrieval
-> LLM 답변
-> /answer/verify
```

## 1. FastAPI 실행

```powershell
pip install -r requirements.txt
opendataloader-pdf-hybrid --port 5002 --ocr-lang "ko,en" --enrich-picture-description & 
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Dify가 Docker 안에서 실행 중이면 URL은 보통 다음 중 하나다.

```text
Windows Docker Desktop: http://host.docker.internal:8000
Linux Docker bridge:    http://172.22.0.1:8000
같은 docker-compose 네트워크: http://<service-name>:8000
```

Dify 컨테이너 안에서 먼저 확인한다.

```bash
curl http://host.docker.internal:8000/health
curl http://172.22.0.1:8000/health
```

## 2. Knowledge Pipeline

구조:

```text
Data Source / File
-> PDF to RAG
-> Parse RAG Response
-> Parent-Child Chunker
-> Knowledge Base
```

### 2.1 Start / Data Source

입력:

```text
file: File
```

Dify 파일 변수에서 보통 선택 가능한 값:

```text
File.file.name
File.file.related_id
File.file.url
File.file.mime_type
```

`document_id`는 직접 추출하지 말고 `File.file.related_id`를 우선 쓴다. 없으면 API가 자동 생성한다.

### 2.2 HTTP Request: PDF to RAG

노드 이름:

```text
PDF to RAG
```

설정:

```text
Method: POST
URL: http://<fastapi-host>:8000/convert/rag
Authorization: None
Body Type: form-data
```

Headers:

```text
비워둔다.
```

중요: multipart/form-data에서는 `Content-Type`을 직접 넣지 않는다. Dify가 boundary를 포함해서 자동으로 넣어야 파일 업로드가 정상 동작한다.

Form-data:

```text
pdf          File  {{File.file}}
document_id  Text  {{File.file.related_id}}
file_name    Text  {{File.file.name}}
```

`pdf` 대신 `file`이라는 필드명을 써도 API가 받는다.

### 2.3 Code: Parse RAG Response

HTTP 노드 출력이 `body: string`으로 보이면 이 Code 노드를 반드시 둔다.

입력 변수:

```text
body = {{PDF to RAG.body}}
```

Python:

```python
import json

def main(body: str) -> dict:
    data = json.loads(body) if isinstance(body, str) else body
    if not data.get("success"):
        raise ValueError(data.get("error") or "PDF RAG conversion failed")

    stats = data.get("stats") or {}
    dify_markdown = data.get("dify_markdown") or ""
    if not dify_markdown.strip():
        raise ValueError("dify_markdown is empty")

    return {
        "document_id": data.get("document_id", ""),
        "file_name": data.get("file_name", ""),
        "dify_markdown": dify_markdown,
        "chunk_count": int(stats.get("chunk_count") or 0),
        "record_count": int(stats.get("record_count") or 0),
    }
```

출력 변수 타입:

```text
document_id: string
file_name: string
dify_markdown: string
chunk_count: number
record_count: number
```

### 2.4 Parent-Child Chunker

입력 텍스트:

```text
{{Parse RAG Response.dify_markdown}}
```

권장 설정:

```text
Parent chunk size: 1000~1600 tokens
Child chunk size: 250~450 tokens
Overlap: 50~100 tokens
Separator: 기본값 또는 Markdown paragraph 기준
```

### 2.5 Knowledge Base Node

Parent-Child Chunker 결과를 Knowledge Base 노드에 연결한다.

권장 설정:

```text
Chunk Structure: Parent-child
Indexing method: High Quality
Embedding model: qwen3-embedding:8b
Retrieval mode: Hybrid Search
Top K: 20
Rerank: 가능하면 활성화
```

저장 대상은 반드시 `Parse RAG Response.dify_markdown`에서 시작한 Parent-Child Chunker 결과다. `/convert/rag` 응답의 `markdown`, `chunks`, `records`를 Knowledge Base에 그대로 저장하지 않는다.

## 3. Chatflow

구조:

```text
Start
-> Query Plan
-> Parse Query Plan
-> Structured Lookup
-> Knowledge Retrieval
-> Merge Evidence
-> Final Answer LLM
-> Build Verify Body
-> Verify Answer
-> Parse Verify Result
-> Conditional
```

### 3.1 Start

입력:

```text
query: string
document_id: string, optional
```

일반 Chatflow에서는 `query`에 사용자 질문을 넣고, 특정 문서만 조회할 때만 `document_id`를 넣는다.

### 3.2 HTTP Request: Query Plan

노드 이름:

```text
Query Plan
```

설정:

```text
Method: POST
URL: http://<fastapi-host>:8000/query/plan
Authorization: None
Body Type: JSON
```

Headers:

```text
Content-Type: application/json
```

JSON Body:

```json
{
  "query": "{{sys.query}}",
  "document_id": "{{Start.document_id}}"
}
```

### 3.3 Code: Parse Query Plan

입력 변수:

```text
body = {{Query Plan.body}}
```

Python:

```python
import json

def main(body) -> dict:
    data = json.loads(body) if isinstance(body, str) else body
    entities = data.get("entities") or []
    keywords = data.get("keywords") or []
    expanded_queries = data.get("expanded_queries") or []

    return {
        "query": data.get("query", ""),
        "document_id": data.get("document_id") or "",
        "query_type": data.get("query_type", "semantic"),
        "answer_style": data.get("answer_style", "grounded_explanation"),
        "entities": entities,
        "keywords": keywords,
        "sub_queries": data.get("sub_queries") or [],
        "expanded_queries": expanded_queries,
        "entities_text": ", ".join(entities),
        "sub_query_text": "\n".join(data.get("sub_queries") or []),
        "expanded_query_text": "\n".join(expanded_queries),
    }
```

출력 변수 타입:

```text
query: string
document_id: string
query_type: string
answer_style: string
entities: array[string]
keywords: array[string]
sub_queries: array[string]
expanded_queries: array[string]
entities_text: string
sub_query_text: string
expanded_query_text: string
```

### 3.4 HTTP Request: Structured Lookup

노드 이름:

```text
Structured Lookup
```

설정:

```text
Method: POST
URL: http://<fastapi-host>:8000/lookup
Authorization: None
Body Type: JSON
```

Headers:

```text
Content-Type: application/json
```

JSON Body:

```json
{
  "query": "{{sys.query}}",
  "document_id": "{{Parse Query Plan.document_id}}",
  "entities": "{{Parse Query Plan.entities}}",
  "query_type": "{{Parse Query Plan.query_type}}",
  "limit": 8
}
```

만약 Dify JSON 편집기에서 `entities` 배열 변수를 문자열로만 넣는다면, 중간에 Code 노드를 하나 더 두고 raw JSON body를 만든 뒤 HTTP Request의 raw body로 넣는다.

### 3.5 Knowledge Retrieval

검색 질의:

```text
{{Parse Query Plan.expanded_query_text}}
```

대체 질의:

```text
{{sys.query}}
```

권장 설정:

```text
Retrieval mode: Hybrid Search
Top K: 12~20
Score threshold: 처음에는 낮게 또는 비활성
Rerank: 가능하면 활성화
```

### 3.6 Code: Merge Evidence

입력 변수:

```text
lookup_body = {{Structured Lookup.body}}
knowledge_result = {{Knowledge Retrieval.result}}
```

입력 변수명을 이미 `knowledge_context`로 만들어 둔 경우에도 아래 코드는 그대로 동작한다.

Python:

```python
import json

def parse_json(value):
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            return value
    return value

def main(lookup_body=None, knowledge_result=None, knowledge_context=None, knowledge_body=None) -> dict:
    lookup = json.loads(lookup_body) if isinstance(lookup_body, str) else (lookup_body or {})
    structured_context = lookup.get("context") or ""
    diagnostics = lookup.get("diagnostics") or {}

    knowledge_source = knowledge_result
    if knowledge_source is None:
        knowledge_source = knowledge_context
    if knowledge_source is None:
        knowledge_source = knowledge_body

    knowledge_items = parse_json(knowledge_source) or []
    if isinstance(knowledge_items, dict):
        knowledge_items = (
            knowledge_items.get("result")
            or knowledge_items.get("records")
            or knowledge_items.get("data")
            or knowledge_items.get("documents")
            or []
        )

    knowledge_blocks = []
    for index, item in enumerate(knowledge_items[:12], start=1):
        if isinstance(item, dict):
            content = item.get("content") or item.get("text") or item.get("segment", {}).get("content") or ""
            metadata = item.get("metadata") or {}
            source = metadata.get("source") or metadata.get("file_name") or item.get("title") or ""
            knowledge_blocks.append(f"[knowledge {index}] source={source}\n{content}")
        else:
            knowledge_blocks.append(f"[knowledge {index}]\n{item}")

    evidence_context = "\n\n".join(
        part for part in [
            "[Structured Lookup]\n" + structured_context if structured_context else "",
            "[Knowledge Retrieval]\n" + "\n\n".join(knowledge_blocks) if knowledge_blocks else "",
        ]
        if part
    )

    return {
        "structured_context": structured_context,
        "knowledge_context": "\n\n".join(knowledge_blocks),
        "evidence_context": evidence_context,
        "lookup_diagnostics": json.dumps(diagnostics, ensure_ascii=False),
        "lookup_answerability": json.dumps(diagnostics.get("answerability") or {}, ensure_ascii=False),
    }
```

출력 변수 타입:

```text
structured_context: string
knowledge_context: string
evidence_context: string
lookup_diagnostics: string
lookup_answerability: string
```

### 3.7 LLM: Final Answer

System Prompt:

```text
You are a precise RAG answer writer.
Answer only from the provided evidence.
If the evidence is insufficient, say that the provided document does not contain enough evidence.
For numeric, date, policy, requirement, procedure, troubleshooting, identifier, contact, and metric answers, preserve exact values and conditions from evidence.
Prefer evidence lines that include explicit field labels, table headers, key-value pairs, section titles, page numbers, and record metadata.
Use labels and metadata to understand what each value refers to.
Do not invent facts.
Answer in Korean unless the user asks for another language.
```

User Prompt:

```text
Question:
{{sys.query}}

Query type:
{{Parse Query Plan.query_type}}

Answer style:
{{Parse Query Plan.answer_style}}

Entities:
{{Parse Query Plan.entities_text}}

Sub queries:
{{Parse Query Plan.sub_query_text}}

Structured lookup answerability:
{{Merge Evidence.lookup_answerability}}

Evidence:
{{Merge Evidence.evidence_context}}

Write a concise answer with exact supporting facts.
Use only the provided evidence.
When evidence contains labels, table headers, page numbers, section names, or record metadata, use them to avoid mixing unrelated values.
If structured lookup answerability says answerable=false and the remaining evidence does not directly answer the requested attribute, say that the provided document does not contain enough evidence.
If the evidence is insufficient, say that the provided document does not contain enough evidence.
Answer in Korean unless the user asks for another language.
```

### 3.8 Code: Build Verify Body

입력 변수:

```text
question = {{sys.query}}
answer = {{Final Answer.text}}
evidence_context = {{Merge Evidence.evidence_context}}
```

Python:

```python
import json

def main(question: str, answer: str, evidence_context: str) -> dict:
    body = {
        "question": question,
        "answer": answer,
        "evidence": [evidence_context],
    }
    return {"verify_body_json": json.dumps(body, ensure_ascii=False)}
```

출력 변수 타입:

```text
verify_body_json: string
```

### 3.9 HTTP Request: Verify Answer

노드 이름:

```text
Verify Answer
```

설정:

```text
Method: POST
URL: http://<fastapi-host>:8000/answer/verify
Authorization: None
Body Type: raw
Raw Body Type: JSON
```

Headers:

```text
Content-Type: application/json
```

Raw Body:

```text
{{Build Verify Body.verify_body_json}}
```

### 3.10 Conditional

먼저 `Verify Answer.body`를 파싱하는 Code 노드를 하나 둔다.

노드 이름:

```text
Parse Verify Result
```

입력 변수:

```text
body = {{Verify Answer.body}}
```

Python:

```python
import json

def main(body) -> dict:
    if isinstance(body, str):
        data = json.loads(body) if body.strip() else {}
    else:
        data = body or {}

    valid = bool(data.get("valid", False))
    confidence = float(data.get("confidence") or 0)
    query_type = data.get("query_type") or "semantic"
    issues = data.get("issues") or []

    issue_lines = []
    for issue in issues:
        if isinstance(issue, dict):
            issue_type = issue.get("type", "unknown")
            message = issue.get("message", "")
            terms = issue.get("terms")
            suffix = f" terms={terms}" if terms else ""
            issue_lines.append(f"- {issue_type}: {message}{suffix}".strip())
        else:
            issue_lines.append(f"- {issue}")

    return {
        "valid": valid,
        "confidence": confidence,
        "query_type": query_type,
        "issues_text": "\n".join(issue_lines),
        "should_answer": valid and confidence >= 0.7,
        "should_rewrite": (not valid) or confidence < 0.7,
    }
```

출력 변수 타입:

```text
valid: boolean
confidence: number
query_type: string
issues_text: string
should_answer: boolean
should_rewrite: boolean
```

Conditional 노드 조건:

```text
IF Parse Verify Result.should_answer == true
  -> Answer 노드로 Final Answer.text 출력
ELSE
  -> Rewrite Answer LLM 또는 검증 실패 안내
```

처음에는 검증 실패 시 바로 사용자에게 실패를 보여주기보다, `issues`와 evidence를 넣어 LLM에게 한 번 재작성시키는 편이 낫다.

Rewrite Answer LLM의 System Prompt는 Final Answer LLM의 System Prompt를 그대로 써도 된다.

Rewrite Answer LLM의 User Prompt:

```text
Question:
{{sys.query}}

Previous answer:
{{Final Answer.text}}

Verification issues:
{{Parse Verify Result.issues_text}}

Evidence:
{{Merge Evidence.evidence_context}}

Rewrite the answer using only the evidence.
Fix unsupported numbers, unsupported dates, and claims without evidence.
If the evidence is insufficient, say that the provided document does not contain enough evidence.
Answer in Korean.
```

## 4. 외부 API 계약

### 4.1 POST /convert/rag

요청:

```text
multipart/form-data
pdf 또는 file: PDF 파일
document_id: optional
file_name: optional
```

응답 핵심 필드:

```json
{
  "success": true,
  "document_id": "doc_xxx",
  "file_name": "sample.pdf",
  "markdown": "...",
  "dify_markdown": "...",
  "chunks": [],
  "records": [],
  "document_map": [],
  "stats": {
    "chunk_count": 10,
    "record_count": 5,
    "page_count": 3,
    "section_count": 4,
    "dify_markdown_chars": 12000
  },
  "error": ""
}
```

### 4.2 POST /rag/ingest-markdown

PDF 변환을 거치지 않고 이미 정제된 Markdown을 RAG artifact로 저장할 때 사용한다. 자동 평가나 외부 Markdown 변환기를 붙일 때 유용하다.

요청:

```json
{
  "document_id": "manual_001",
  "file_name": "manual.md",
  "markdown": "# Manual\n\n..."
}
```

응답은 `/convert/rag`와 같은 구조다.

### 4.3 POST /query/plan

요청:

```json
{
  "query": "컴퓨터정보공학과 모집인원 알려줘",
  "document_id": "optional"
}
```

### 4.4 POST /lookup

요청:

```json
{
  "query": "컴퓨터정보공학과 모집인원 알려줘",
  "document_id": "optional",
  "entities": ["컴퓨터정보공학과", "모집인원"],
  "query_type": "number_lookup",
  "limit": 8
}
```

응답에는 기존 `context` 외에 `answer_style`, `sub_queries`, `evidence_items`, `diagnostics`가 포함된다. Dify에서는 일단 `context`만 써도 되고, 디버깅할 때 `diagnostics.average_coverage`, `diagnostics.selected_count`를 보면 검색 품질을 판단하기 쉽다.

### 4.5 POST /answer/verify

요청:

```json
{
  "question": "컴퓨터정보공학과 모집인원 알려줘",
  "answer": "컴퓨터정보공학과 모집인원은 45명입니다.",
  "evidence": ["..."]
}
```

## 5. 수동 테스트

```powershell
curl.exe http://127.0.0.1:8000/health

curl.exe -F "pdf=@2026학년도 입학전형 신입생 모집요강.pdf" -F "document_id=sample_2026" -F "file_name=2026_admission.pdf" http://127.0.0.1:8000/convert/rag

curl.exe -H "Content-Type: application/json" -d "{\"query\":\"컴퓨터정보공학과 모집인원 알려줘\"}" http://127.0.0.1:8000/query/plan

curl.exe -H "Content-Type: application/json" -d "{\"query\":\"컴퓨터정보공학과 모집인원 알려줘\",\"document_id\":\"sample_2026\",\"entities\":[\"컴퓨터정보공학과\",\"모집인원\"],\"query_type\":\"number_lookup\",\"limit\":8}" http://127.0.0.1:8000/lookup
```

## 6. 중요한 운영 판단

이 구조는 Dify Knowledge Pipeline을 그대로 쓰는 방식이다. 따라서 Knowledge Base 노드는 마지막에 연결되어야 하고, Parent-Child Chunker와 embedding은 Dify가 처리한다.

`/convert/rag`가 반환하는 `dify_markdown`은 Dify에 저장하기 좋은 Markdown이다. 원본 `markdown`은 디버깅과 재처리용이고, `chunks`와 `records`는 FastAPI의 structured lookup과 answer verification을 위한 보조 데이터다.
