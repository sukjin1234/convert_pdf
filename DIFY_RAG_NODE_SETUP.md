# Dify RAG 노드 구성 가이드

이 구성은 Dify를 UI, Knowledge Pipeline, Knowledge Retrieval에 쓰고, 정확도를 좌우하는 문서 정규화, 표 행 확장, 구조화 조회, 답변 검증은 외부 FastAPI가 맡는 방식이다

목표 흐름:

```text
PDF/TXT/Markdown/DOCX/CSV
-> FastAPI /convert/rag
-> Dify Knowledge Base에 dify_markdown 저장
-> Chatflow에서 /query/plan + /lookup + Knowledge Retrieval
-> LLM 답변 + 참조 문서 표시
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
-> Document to RAG
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

### 2.2 HTTP Request: Document to RAG

노드 이름:

```text
Document to RAG
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
file         File  {{File.file}}
document_id  Text  {{File.file.related_id}}
file_name    Text  {{File.file.name}}
```

기존 PDF 전용 노드와의 호환을 위해 `pdf` 필드명도 계속 받지만, 새로 구성할 때는 `file`을 사용한다. 현재 지원 형식은 `PDF`, `TXT`, `MD/Markdown`, `DOCX`, `CSV`, `TSV`다.

### 2.3 Code: Parse RAG Response

HTTP 노드 출력이 `body: string`으로 보이면 이 Code 노드를 반드시 둔다.

입력 변수:

```text
body = {{Document to RAG.body}}
```

Python:

```python
import json

def main(body: str) -> dict:
    data = json.loads(body) if isinstance(body, str) else body
    if not data.get("success"):
        raise ValueError(data.get("error") or "Document RAG conversion failed")

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
Parent mode: Paragraph
Parent delimiter: Markdown heading/paragraph 기준
Parent maximum chunk length: 1200~1800 characters
Child delimiter: newline 또는 sentence 기준
Child maximum chunk length: 250~500 characters
```

Dify 공식 문서 기준으로 `Parent-child Chunker`에는 별도 `Chunk Overlap` 설정이 없다. Overlap이 필요한 경우 `General Chunker`에서만 직접 설정할 수 있고, Parent-child에서는 parent chunk를 약간 크게 잡고 child delimiter를 촘촘하게 잡아 문맥 손실을 줄인다.

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
-> Build Merge Evidence Body
-> Merge Evidence Request
-> Parse Merge Evidence
-> Final Answer LLM, 답변 끝에 참조 문서 표시
-> Build Verify Body
-> Verify Answer
-> Parse Verify Result
-> Build Eval Log Body
-> Eval Log
-> Conditional
```

### 3.1 Start

입력:

```text
query: string
document_id: string, optional, hidden/internal
```

일반 사용자에게 `document_id`를 직접 입력하게 하지 않는다. 기본 UX는 사용자가 질문만 입력하고, 답변 끝에 `참조 문서:`로 실제 사용한 문서, 페이지, 섹션을 보여주는 방식이다.

`document_id`는 특정 파일 보기, 관리자 테스트, 파일 업로드 직후 해당 파일 안에서만 묻는 모드처럼 문서 범위가 이미 정해진 경우에만 hidden/internal 입력으로 넣는다. 예를 들어 업로드형 화면이면 Knowledge Pipeline의 `Document to RAG`에서 사용한 `document_id`를 앱 내부 상태로 보관했다가 Chatflow의 `Start.document_id`에 전달한다.

전체 Knowledge를 대상으로 묻는 공개 채팅에서는 `document_id`를 비워둘 수 있다. 이때 `/lookup`은 여러 문서의 구조화 records를 함께 검색하고, 동점 수준의 후보가 여러 문서에 있으면 단일 값으로 무리하게 단정하지 않도록 출처별 값을 근거에 남긴다. 최종 답변은 `Parse Merge Evidence.source_summary`를 사용해 어떤 문서를 참조했는지 표시한다.

실제 점검 기준:

- 특정 문서 모드인데 Dify App API 호출에서 `inputs.document_id`를 넘겨도 Start 노드에 `document_id` 입력 변수가 없으면 값이 완전히 무시된다.
- 특정 문서 모드에서 스트리밍 이벤트의 `workflow_started.data.inputs`에 `document_id`가 없고 `Query Plan` 요청 body가 `{"query": ...}`만 포함하면 Chatflow 설정 오류다.
- 전체 Knowledge 검색 모드라면 `document_id`가 빈 값인 것이 정상이다. 대신 Final Answer에 반드시 `참조 문서:` 줄이 있어야 한다.

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
Top K: 8~12
Score threshold: 처음에는 낮게 또는 비활성
Rerank: 가능하면 활성화
```

Knowledge에는 `/convert/rag` 응답의 `dify_markdown`만 넣는다. 현재 변환기는 Knowledge 검색용 chunk에서 큰 원본 표와 목차를 줄이고, 행 단위 `structured_record`와 `answer_hint`를 우선 노출한다. 기존 PDF 원문을 그대로 넣은 문서는 긴 숫자 표와 목차가 상위 검색을 오염시키므로, compact RAG Markdown으로 재적재한 뒤 기존 PDF 문서는 비활성화한다.

### 3.6 HTTP Request: Merge Evidence Request

노드 이름:

```text
Merge Evidence Request
```

설정:

```text
Method: POST
URL: http://<fastapi-host>:8000/chatflow/merge-evidence
Authorization: None
Body Type: raw
Raw Body Type: JSON
```

Headers:

```text
Content-Type: application/json
```

권장 방식은 바로 앞에 `Build Merge Evidence Body` Code 노드를 두고 raw JSON 문자열을 만든 뒤 이 노드에 연결하는 것이다. Dify HTTP Request의 JSON 편집기에서 객체 변수를 직접 섞으면 특정 케이스에서 body가 `[{"lookup_body": ...}]` 배열로 나가 FastAPI/Pydantic 422가 발생할 수 있다.

#### Code: Build Merge Evidence Body

입력 변수:

```text
lookup_body = {{Structured Lookup.body}}
knowledge_result = {{Knowledge Retrieval.result}}
```

Python:

```python
import json

def parse_json_like(value):
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            return value
    return value

def main(lookup_body=None, knowledge_result=None) -> dict:
    body = {
        "lookup_body": parse_json_like(lookup_body),
        "knowledge_result": parse_json_like(knowledge_result),
    }
    return {"merge_body_json": json.dumps(body, ensure_ascii=False)}
```

출력 변수 타입:

```text
merge_body_json: string
```

Merge Evidence Request Raw Body:

```text
{{Build Merge Evidence Body.merge_body_json}}
```

이 노드는 Structured Lookup의 `answer_contract`, `complete_values`, `answerability`와 Knowledge Retrieval 결과를 서버 쪽에서 병합한다. Dify `Knowledge Retrieval.result`가 `data.records`, `segment.content`, `metadata`처럼 중첩되어 와도 FastAPI가 근거 텍스트를 보존한다.

Dify HTTP Request 노드 출력은 다음처럼 고정된다.

```text
body: string
status_code: number
headers: object
files: array[file]
```

따라서 `Merge Evidence Request.body`를 Final Answer LLM에 직접 연결하지 않는다. 바로 뒤에 `Parse Merge Evidence` Code 노드를 두고, `body`에서 실제 출력 변수를 추출한다.

### 3.7 Code: Parse Merge Evidence

입력 변수:

```text
body = {{Merge Evidence Request.body}}
status_code = {{Merge Evidence Request.status_code}}
```

Python:

```python
import json

def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def main(body, status_code=200) -> dict:
    code = int(status_code or 0)
    if code < 200 or code >= 300:
        raise ValueError(f"Merge Evidence Request failed. status_code={code} body={as_text(body)[:500]}")

    data = json.loads(body) if isinstance(body, str) else body
    if not isinstance(data, dict):
        raise ValueError("Merge Evidence Request body is not a JSON object")

    evidence_context = as_text(data.get("evidence_context"))
    if not evidence_context.strip():
        evidence_context = "제공된 구조화/Knowledge 근거가 비어 있습니다. 문서에 충분한 근거가 없다고 답하세요."

    return {
        "structured_context": as_text(data.get("structured_context")),
        "knowledge_context": as_text(data.get("knowledge_context")),
        "evidence_context": evidence_context,
        "evidence_priority": as_text(data.get("evidence_priority")),
        "source_summary": as_text(data.get("source_summary")),
        "answer_contract": as_text(data.get("answer_contract")),
        "answer_contract_text": as_text(data.get("answer_contract_text")),
        "lookup_diagnostics": as_text(data.get("lookup_diagnostics")),
        "lookup_answerability": as_text(data.get("lookup_answerability")),
    }
```

출력 변수 타입:

```text
structured_context: string
knowledge_context: string
evidence_context: string
evidence_priority: string
source_summary: string
answer_contract: string
answer_contract_text: string
lookup_diagnostics: string
lookup_answerability: string
```

이 구조가 기존의 긴 Dify Code 병합보다 낫다. 정확도에 영향을 주는 근거 병합, `required_values`, Knowledge Retrieval 중첩 포맷 파싱은 FastAPI 한 곳에서 관리하고, Dify Code 노드는 HTTP `body`를 검증하고 펼치는 역할만 한다.

### 3.8 LLM: Final Answer

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
{{Parse Merge Evidence.lookup_answerability}}

Answer contract:
{{Parse Merge Evidence.answer_contract_text}}

Evidence priority:
{{Parse Merge Evidence.evidence_priority}}

Source summary:
{{Parse Merge Evidence.source_summary}}

Evidence:
{{Parse Merge Evidence.evidence_context}}

Write a concise answer with exact supporting facts.
Use only the provided evidence.
Structured Lookup is authoritative exact evidence from parsed tables, records, and field-value pairs.
If Structured Lookup contains an answer_candidate or directly relevant evidence, answer from Structured Lookup even when Knowledge Retrieval is empty, broad, or less specific.
If evidence contains "[Direct Answer - complete structured result]" and "complete_values", include every value in complete_values exactly once. Do not omit values from complete_values and do not add values that are not listed there.
If Answer contract contains required_values, include every required value exactly once before adding any supplementary explanation.
Use Knowledge Retrieval only as supplementary context.
For table, list, field, value, number, date, identifier, and policy questions, do not ignore Structured Lookup just because Knowledge Retrieval has no exact match.
When evidence contains labels, table headers, page numbers, section names, or record metadata, use them to avoid mixing unrelated values.
End the answer with one short line that starts with "참조 문서:" and uses Source summary. If Source summary says there is no evidence, write "참조 문서: 제공된 근거 없음".
If structured lookup answerability says answerable=false and the remaining evidence does not directly answer the requested attribute, say that the provided document does not contain enough evidence.
If the evidence is insufficient, say that the provided document does not contain enough evidence.
Answer in Korean unless the user asks for another language.
```

### 3.9 Code: Build Verify Body

입력 변수:

```text
question = {{sys.query}}
answer = {{Final Answer.text}}
evidence_context = {{Parse Merge Evidence.evidence_context}}
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

### 3.10 HTTP Request: Verify Answer

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

### 3.11 Code: Parse Verify Result

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

### 3.12 Code: Build Eval Log Body

Dify HTTP Request의 JSON body 편집기에는 boolean 변수나 `Knowledge Retrieval.result` 같은 object/list 변수가 직접 들어가지 않는 경우가 많다. 그래서 로그 payload는 Code 노드에서 JSON 문자열로 만든 뒤, HTTP 노드에는 raw body로 전달한다.

노드 이름:

```text
Build Eval Log Body
```

입력 변수:

```text
query = {{sys.query}}
document_id = {{Parse Query Plan.document_id}}
query_plan_body = {{Query Plan.body}}
structured_lookup_body = {{Structured Lookup.body}}
knowledge_context = {{Parse Merge Evidence.knowledge_context}}
evidence_context = {{Parse Merge Evidence.evidence_context}}
evidence_priority = {{Parse Merge Evidence.evidence_priority}}
source_summary = {{Parse Merge Evidence.source_summary}}
answer_contract_text = {{Parse Merge Evidence.answer_contract_text}}
lookup_answerability = {{Parse Merge Evidence.lookup_answerability}}
final_answer = {{Final Answer.text}}
verify_answer_body = {{Verify Answer.body}}
```

중요: `knowledge_context`에는 `Knowledge Retrieval.result`를 직접 넣지 말고, 반드시 `Parse Merge Evidence.knowledge_context`를 넣는다. 이 값은 앞선 Merge Evidence Request 응답을 파싱해 검색 결과를 문자열로 정리한 출력이다.

Python:

```python
import json

def parse_json(value, default=None):
    if value is None:
        return default
    if isinstance(value, (dict, list, bool, int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except Exception:
            return default
    return default

def issue_lines(issues):
    lines = []
    for issue in issues or []:
        if isinstance(issue, dict):
            issue_type = issue.get("type", "unknown")
            message = issue.get("message", "")
            terms = issue.get("terms")
            suffix = f" terms={terms}" if terms else ""
            lines.append(f"- {issue_type}: {message}{suffix}".strip())
        else:
            lines.append(f"- {issue}")
    return "\n".join(lines)

def main(
    query="",
    document_id="",
    query_plan_body="",
    structured_lookup_body="",
    knowledge_context="",
    evidence_context="",
    evidence_priority="",
    source_summary="",
    answer_contract_text="",
    lookup_answerability="",
    final_answer="",
    verify_answer_body="",
) -> dict:
    query_plan = parse_json(query_plan_body, {}) or {}
    verify = parse_json(verify_answer_body, {}) or {}
    answerability = parse_json(lookup_answerability, {}) or {}
    valid = bool(verify.get("valid", False))
    confidence = float(verify.get("confidence") or 0)
    issues = verify.get("issues", [])

    payload = {
        "query": query,
        "document_id": document_id or query_plan.get("document_id", ""),
        "query_plan": {
            "query_type": query_plan.get("query_type", ""),
            "answer_style": query_plan.get("answer_style", ""),
            "entities": query_plan.get("entities", []),
            "sub_queries": query_plan.get("sub_queries", []),
            "expanded_query": query_plan.get("expanded_query", ""),
        },
        "nodes": {
            "query_plan_body": query_plan_body,
            "structured_lookup_body": structured_lookup_body,
            "knowledge_retrieval_text": knowledge_context,
            "merge_evidence": {
                "evidence_context": evidence_context,
                "evidence_priority": evidence_priority,
                "source_summary": source_summary,
                "answer_contract_text": answer_contract_text,
                "lookup_answerability": answerability,
            },
            "final_answer": final_answer,
            "verify_answer_body": verify_answer_body,
            "parse_verify_result": {
                "valid": valid,
                "confidence": confidence,
                "query_type": verify.get("query_type", ""),
                "issues": issues,
                "issues_text": issue_lines(issues),
                "should_answer": valid and confidence >= 0.7,
                "should_rewrite": (not valid) or confidence < 0.7,
            },
        },
    }

    return {
        "eval_log_body_json": json.dumps(payload, ensure_ascii=False)
    }
```

출력 변수 타입:

```text
eval_log_body_json: string
```

### 3.13 HTTP Request: Eval Log

노드 이름:

```text
Eval Log
```

설정:

```text
Method: POST
URL: http://<fastapi-host>:8000/eval/log
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
{{Build Eval Log Body.eval_log_body_json}}
```

`Eval Log` 응답은 답변 판단에 쓰지 않는다. HTTP 노드가 실패해도 운영 답변 흐름을 막지 않게 하려면 Dify에서 실패 시 계속 진행하도록 설정하거나, 최종 Answer 경로와 분리한다.

로그 저장 위치:

```text
기본: OS temp/dify-rag-eval-logs
변경: EVAL_LOG_DIR 환경 변수
```

조회:

```bash
curl http://<fastapi-host>:8000/eval/logs?limit=20
curl http://<fastapi-host>:8000/eval/logs/<run_id>
```

### 3.14 Conditional

조건:

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

Evidence priority:
{{Parse Merge Evidence.evidence_priority}}

Source summary:
{{Parse Merge Evidence.source_summary}}

Evidence:
{{Parse Merge Evidence.evidence_context}}

Rewrite the answer using only the evidence.
If Structured Lookup contains an answer_candidate or directly relevant evidence, answer from Structured Lookup even when Knowledge Retrieval is empty or less specific.
If evidence contains complete_values, include every value in complete_values exactly once.
Fix unsupported numbers, unsupported dates, and claims without evidence.
End the answer with one short line that starts with "참조 문서:" and uses Source summary. If Source summary says there is no evidence, write "참조 문서: 제공된 근거 없음".
If the evidence is insufficient, say that the provided document does not contain enough evidence.
Answer in Korean.
```

## 4. 외부 API 계약

### 4.1 POST /convert/rag

요청:

```text
multipart/form-data
file: PDF/TXT/Markdown/DOCX/CSV/TSV 파일
document_id: optional
file_name: optional
```

기존 호환용으로 `pdf` 필드도 받는다.

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

파일 업로드를 거치지 않고 이미 정제된 Markdown을 RAG artifact로 저장할 때 사용한다. 자동 평가나 외부 Markdown 변환기를 붙일 때 유용하다.

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

응답에는 기존 `context` 외에 `direct_answer`, `answer_items`, `answer_field`, `filter_terms`, `answer_contract`, `answer_style`, `sub_queries`, `evidence_items`, `diagnostics`가 포함된다.

목록/표 질문에서는 `context` 맨 위에 다음 블록이 자동으로 붙는다.

```text
[Direct Answer - complete structured result]
answer_candidate: ...
answer_field: ...
filter_terms: ...
complete_values:
- ...
- ...
Instruction: include every value in complete_values; do not omit or add values.
```

Dify에서는 일단 `context`만 써도 된다. 디버깅할 때는 `direct_answer`가 비어 있는지, `answer_items`에 빠진 행이 없는지, `diagnostics.average_coverage`, `diagnostics.selected_count`를 확인한다.

표 행에 `♣`, `†`, `◈` 같은 표시가 있고 `♣표시 : ...`, `† indicates ...`, `◈ = ...` 형태의 범례가 있으면 `/lookup`이 표시 의미를 행 필드에 자동 주입한다. 범례가 표 바로 아래가 아니라 다음 페이지나 다른 섹션으로 분리되어도, 문서 안에서 해당 기호의 의미가 하나로 명확하면 연결된다. 예를 들어 `♣표시 : 4년제 학사학위(전공심화)과정 개설 학과`가 있으면 `♣`가 붙은 모집단위만 `전공심화 개설 학과`의 후보로 집계된다.

### 4.5 POST /answer/verify

요청:

```json
{
  "question": "컴퓨터정보공학과 모집인원 알려줘",
  "answer": "컴퓨터정보공학과 모집인원은 45명입니다.",
  "evidence": ["..."]
}
```

### 4.6 POST /chatflow/merge-evidence

Dify `Merge Evidence Request` HTTP 노드에서 사용한다. `Structured Lookup.body`와 `Knowledge Retrieval.result`를 받아 최종 LLM에 넣을 근거 패키지를 만든다.

요청:

```json
{
  "lookup_body": {
    "query": "전공심화 개설 학과 알려줘",
    "direct_answer": "개설 학과: 컴퓨터정보공학과, 기계공학과",
    "answer_contract": {
      "required_values": ["컴퓨터정보공학과", "기계공학과"]
    },
    "context": "..."
  },
  "knowledge_result": []
}
```

응답 핵심 필드:

```json
{
  "structured_context": "...",
  "knowledge_context": "...",
  "evidence_context": "...",
  "evidence_priority": "Use Structured Lookup first...",
  "source_summary": "참조 문서: sample.pdf (p.5, 전형일정)",
  "answer_contract": "{...}",
  "answer_contract_text": "[Answer Contract - obey before writing]\\n...",
  "lookup_answerability": "{...}"
}
```

Dify HTTP Request 노드에서는 위 JSON이 `body` 문자열로 들어온다. 따라서 `Parse Merge Evidence` Code 노드에서 `body`를 파싱한 뒤 `answer_contract_text`, `evidence_priority`, `source_summary`, `evidence_context`를 Final Answer LLM 프롬프트에 연결한다.

### 4.7 POST /chatflow/debug

Dify UI에서 답변이 이상할 때 같은 질문으로 API 쪽 흐름을 먼저 확인한다.

요청:

```json
{
  "query": "전공심화 개설 학과 알려줘",
  "document_id": "sample_2026",
  "knowledge_result": [],
  "answer": "제공된 문서에는 충분한 근거가 없습니다."
}
```

핵심 확인값:

```text
query_plan.query_type == table_lookup
node_status.structured_lookup_has_context == true
merge_evidence.evidence_context contains "Structured Lookup - authoritative exact evidence"
merge_evidence.source_summary starts with "참조 문서:"
draft_verification.valid == true
supplied_answer_verification.valid == false
```

위 값이 맞으면 FastAPI의 structured lookup, evidence merge, verifier는 정상이다. 그 상태에서 Dify 답변만 실패하면 `Merge Evidence` 코드가 최신인지, `evidence_priority`와 `source_summary` 출력 변수를 만들었는지, Final Answer LLM User Prompt에 `Evidence priority`, `Source summary`, `Evidence`가 모두 연결됐는지 확인한다.

### 4.8 POST /eval/log

Dify 실제 실행 trace를 로컬 JSONL로 저장한다. 요청 body는 자유 형식이며, API가 `run_id`, `created_at`, `_http`, `size`를 추가한다.

요청 예:

```json
{
  "query": "전공심화 개설 학과 알려줘",
  "document_id": "sample_2026",
  "final_answer": "...",
  "verify_result": {
    "valid": true
  }
}
```

응답:

```json
{
  "success": true,
  "run_id": "...",
  "created_at": "...",
  "log_path": "...",
  "jsonl_path": "...",
  "size": 1234
}
```

조회:

```text
GET /eval/logs?limit=50
GET /eval/logs?date=2026-06-10
GET /eval/logs/{run_id}
```

## 5. 수동 테스트

```powershell
curl.exe http://127.0.0.1:8000/health

curl.exe -F "file=@2026학년도 입학전형 신입생 모집요강.pdf" -F "document_id=sample_2026" -F "file_name=2026_admission.pdf" http://127.0.0.1:8000/convert/rag

curl.exe -F "file=@manual.md" -F "document_id=manual" http://127.0.0.1:8000/convert/rag

curl.exe -H "Content-Type: application/json" -d "{\"query\":\"컴퓨터정보공학과 모집인원 알려줘\"}" http://127.0.0.1:8000/query/plan

curl.exe -H "Content-Type: application/json" -d "{\"query\":\"컴퓨터정보공학과 모집인원 알려줘\",\"document_id\":\"sample_2026\",\"entities\":[\"컴퓨터정보공학과\",\"모집인원\"],\"query_type\":\"number_lookup\",\"limit\":8}" http://127.0.0.1:8000/lookup

curl.exe -H "Content-Type: application/json" -d "{\"query\":\"전공심화 개설 학과 알려줘\",\"document_id\":\"sample_2026\",\"knowledge_result\":[],\"answer\":\"제공된 문서에는 충분한 근거가 없습니다.\"}" http://127.0.0.1:8000/chatflow/debug
```

## 6. 중요한 운영 판단

이 구조는 Dify Knowledge Pipeline을 그대로 쓰는 방식이다. 따라서 Knowledge Base 노드는 마지막에 연결되어야 하고, Parent-Child Chunker와 embedding은 Dify가 처리한다.

`/convert/rag`가 반환하는 `dify_markdown`은 Dify에 저장하기 좋은 Markdown이다. 원본 `markdown`은 디버깅과 재처리용이고, `chunks`와 `records`는 FastAPI의 structured lookup과 answer verification을 위한 보조 데이터다.
