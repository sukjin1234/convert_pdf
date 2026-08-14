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

운영 배포에서는 FastAPI worker를 2개로 시작하고 부하 테스트 결과와 메모리 사용량을 보고 4개까지 늘린다. 각 worker는 RAG artifact와 조회 캐시를 별도로 메모리에 올리므로 CPU 수만 보고 과도하게 늘리지 않는다.

```powershell
$env:RAG_STORE_DIR="D:\dify-rag\rag-store"
$env:RAG_STORE_REFRESH_INTERVAL_SECONDS="0.5"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 --proxy-headers
```

모든 worker와 재시작된 컨테이너가 같은 영속 `RAG_STORE_DIR`을 사용해야 한다. 서버는 공유 디렉터리의 artifact 변경을 감지해 worker별 메모리와 조회 캐시를 갱신한다. PDF 변환은 worker 수와 무관하게 프로세스 간 파일 잠금으로 한 번에 하나만 수행해 OCR/GPU 메모리 경합을 막고, `/lookup`, `/query/plan`, `/chatflow/merge-evidence`는 병렬 처리한다.

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
-> Build Session Context
-> Session Follow-up Guard
-> IF Previous Source Follow-up
   TRUE  -> Previous Source Answer, 답변 출력 후 이 분기는 종료
   FALSE -> Rewrite Standalone Query
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
            -> Conditional
               TRUE  -> Build Conversation State(Final Answer)
                        -> Variable Assigner
                        -> Build Eval Log Body
                        -> Eval Log
                        -> Answer
               FALSE -> Rewrite Answer LLM
                        -> Build Conversation State(Rewrite Answer)
                        -> Variable Assigner
                        -> Build Eval Log Body
                        -> Eval Log
                        -> Answer
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
- API 클라이언트는 첫 요청에서 `conversation_id`를 비우고, 응답으로 받은 `conversation_id`를 같은 `user`와 함께 다음 요청에 넣어야 한다. `user`가 바뀌면 같은 `conversation_id`라도 다른 사용자 세션으로 취급되어 히스토리 조회/연결이 깨진다.

### 3.1.1 Session Follow-up Guard

`3.1.1`은 기존 RAG 답변 생성 경로를 바꾸는 본문 검색 노드가 아니라, `Start`와 `Rewrite Standalone Query` 사이에 추가한 세션 보호 분기다. 새로 넣은 이유는 두 가지다.

- Dify 실제 UI에서 Code 노드 입력 변수로 conversation variable이 직접 연결되지 않는 경우가 있어, Template 노드로 먼저 세션 값을 읽어 Code 노드에 넘겨야 한다.
- 사용자가 “방금 답변에서 참조한 문서 이름만 말해줘”처럼 직전 답변의 출처를 물으면 Knowledge Retrieval을 다시 돌리지 않고 직전 답변에 저장된 출처를 그대로 답해야 한다.

Chatflow API에서 `conversation_id`가 유지되어도, 현재 질문을 그대로 Knowledge Retrieval에 넣으면 “방금 답변에서 참조한 문서 이름만 말해줘” 같은 후속질문이 독립 검색 질의가 된다. 이 경우 Knowledge가 임의의 문서를 높은 점수로 반환하고, 이전 답변의 출처를 잃을 수 있다.

따라서 Chatflow에 다음 conversation variables를 만든다.

```text
last_question: string
last_answer_brief: string
last_source_summary: string
last_document_names: string
```

일부 Dify 환경에서는 Code 노드의 입력 변수 선택 UI에서 conversation variable이 직접 연결되지 않는다. 따라서 `Start` 바로 뒤에 Template 노드를 먼저 두고, conversation variable을 하나의 upstream 문자열 출력으로 만든 다음 Code 노드가 그 출력만 받게 한다.

노드 이름:

```text
Build Session Context
```

노드 타입:

```text
Template
```

입력 변수:

```text
query               = User Input의 query 변수
last_question       = conversation.last_question
last_answer_brief   = conversation.last_answer_brief
last_source_summary = conversation.last_source_summary
last_document_names = conversation.last_document_names
```

Dify 버전에 따라 사용자 질문 변수는 `sys.query`, `userinput.query`, 또는 Start/User Input의 `query`로 보일 수 있다. Template 본문에 `{{sys.query}}`를 직접 쓰지 말고, Template 노드의 입력 변수 `query`에 Dify 변수 선택기로 현재 질문을 먼저 연결한다. 그렇지 않으면 Template 노드의 Jinja 실행 환경에서 `sys`가 정의되지 않아 `'sys' is undefined` 오류가 난다.

`last_question` 같은 값이 변수 선택기에 보이지 않으면, 먼저 Chatflow 캔버스의 Conversation Variables 패널에서 아래 4개 변수를 직접 생성한다. Variable Assigner 노드를 만든다고 conversation variable이 자동 생성되는 것이 아니다.

```text
last_question: string, default ""
last_answer_brief: string, default ""
last_source_summary: string, default ""
last_document_names: string, default ""
```

생성 후에도 Template 노드 입력 변수 선택기에 보이지 않으면 화면을 새로고침하거나 앱을 저장한 뒤 다시 연다. 그래도 보이지 않는 Dify 버전에서는 `Build Session Context`와 `Session Follow-up Guard`를 일단 제거하고, `Start -> Rewrite Standalone Query`로 바로 연결한다. 이 경우 “방금 답변의 출처/문서명만 알려줘” 같은 후속질문 최적화만 빠지고, 일반 RAG 답변 흐름은 그대로 동작한다.

Template 본문:

```text
<<<DIFY_SESSION_FIELD:query>>>
{{ query | default("") }}
<<<DIFY_SESSION_FIELD:last_question>>>
{{ last_question | default("") }}
<<<DIFY_SESSION_FIELD:last_answer_brief>>>
{{ last_answer_brief | default("") }}
<<<DIFY_SESSION_FIELD:last_source_summary>>>
{{ last_source_summary | default("") }}
<<<DIFY_SESSION_FIELD:last_document_names>>>
{{ last_document_names | default("") }}
<<<DIFY_SESSION_END>>>
```

위 변수들은 직접 텍스트로 억지 입력하지 말고 Template 노드의 입력 변수 설정에서 Dify 변수 선택기 또는 `/` 입력으로 연결한다. 화면에서 표시되는 변수명이 예시와 다르면, 입력 변수명은 예시처럼 짧게 맞추고 값 쪽에 Dify가 선택한 실제 변수를 연결한다. UI에서 변수 뒤에 `| default("")` 필터를 붙일 수 없으면 필터를 생략해도 된다. Code 노드에서 빈 값을 다시 안전하게 처리한다. Template 노드에서도 conversation variable이 보이지 않으면 Chatflow가 아니거나, Conversation Variables 패널에 `last_question`, `last_answer_brief`, `last_source_summary`, `last_document_names`가 생성되지 않은 상태다.

출력 변수:

```text
output: string
```

그 바로 뒤에 Code 노드를 둔다.

노드 이름:

```text
Session Follow-up Guard
```

입력 변수:

```text
session_context = {{Build Session Context.output}}
```

Code 노드의 입력 변수 목록에는 위 한 줄만 있어야 한다. 변수 이름이 비어 있는 행이 하나라도 남아 있으면 Dify가 `main`에 빈 keyword argument를 넘겨 `TypeError: main() got an unexpected keyword argument ''` 오류가 난다. 빈 입력 변수 행은 휴지통 아이콘으로 삭제한다.

Python:

```python
import re

def text(value) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if value in {"", '""', "''", "null", "None"}:
        return ""
    return value

def parse_session_context(raw: str) -> dict:
    raw = text(raw)
    fields = {}
    pattern = re.compile(
        r"<<<DIFY_SESSION_FIELD:([a-zA-Z0-9_]+)>>>\s*(.*?)\s*(?=<<<DIFY_SESSION_FIELD:|<<<DIFY_SESSION_END>>>|\Z)",
        re.S,
    )
    for key, value in pattern.findall(raw):
        fields[key] = value.strip()
    return fields

def contains_any(value: str, terms) -> bool:
    value = value.lower()
    return any(term in value for term in terms)

def main(session_context="", **kwargs) -> dict:
    if not text(session_context).strip():
        session_context = kwargs.get("", "")
    data = parse_session_context(session_context)
    q = text(data.get("query")).strip()
    q_lower = q.lower()
    last_question = text(data.get("last_question")).strip()
    last_answer_brief = text(data.get("last_answer_brief")).strip()
    last_source_summary = text(data.get("last_source_summary")).strip()
    last_document_names = text(data.get("last_document_names")).strip()

    asks_source = contains_any(q_lower, ("참조", "출처", "문서", "파일", "페이지", "근거", "source", "document", "file", "page"))
    refers_previous = contains_any(q_lower, ("방금", "이전", "위 답변", "앞서", "그 답변", "저 답변", "last", "previous", "above", "that"))
    is_source_followup = asks_source and refers_previous and bool(last_source_summary)

    direct_answer = ""
    if is_source_followup:
        if contains_any(q_lower, ("이름", "문서명", "파일명", "name")) and last_document_names:
            direct_answer = last_document_names
        else:
            direct_answer = last_source_summary

    return {
        "is_source_followup": is_source_followup,
        "direct_answer": direct_answer,
        "has_previous_answer": bool(last_answer_brief),
        "query": q,
        "last_question": last_question,
        "last_answer_brief": last_answer_brief,
        "last_source_summary": last_source_summary,
        "last_document_names": last_document_names,
    }
```

출력 변수 타입:

```text
is_source_followup: boolean
direct_answer: string
has_previous_answer: boolean
query: string
last_question: string
last_answer_brief: string
last_source_summary: string
last_document_names: string
```

바로 뒤에 Conditional 노드를 둔다. 이 노드는 RAG 검색으로 보낼지, 직전 답변 출처만 바로 답할지 나누는 분기다.

노드 이름:

```text
IF Previous Source Follow-up
```

분기:

```text
IF Session Follow-up Guard.is_source_followup == true
  -> Previous Source Answer
ELSE
  -> Rewrite Standalone Query
```

`true` 쪽에는 Answer 노드를 하나 만들고 이름을 `Previous Source Answer`로 둔다.

Answer 내용:

```text
{{Session Follow-up Guard.direct_answer}}
```

`Previous Source Answer` 뒤에는 다른 노드를 연결하지 않는다. 이 분기는 여기서 답변을 내고 끝난다. `Save Conversation State`에도 연결하지 않는다. 이 분기는 새 검색 답변이 아니라 직전 답변의 출처를 다시 보여주는 용도이기 때문이다.

`false` 쪽은 기존 RAG 경로인 `Rewrite Standalone Query`로 연결한다. 일반 질문, 새 질문, “그럼 수시2차는?” 같은 맥락 후속질문은 모두 이 경로로 간다.

이 분기를 넣으면 출처/문서명/페이지를 묻는 후속질문은 Knowledge Retrieval을 다시 타지 않고 직전 답변의 출처 상태에서 즉시 답한다.

### 3.1.2 Rewrite Standalone Query

출처 확인이 아닌 일반 후속질문은 검색 전에 독립 질문으로 바꾼다. 예를 들어 “그럼 수시2차는?”은 “직전 질문의 주제와 같은 항목에서 수시2차 값은?”처럼 검색 가능한 질문이어야 한다.

노드 이름:

```text
Rewrite Standalone Query
```

LLM System Prompt:

```text
Rewrite the user's current Korean question into a standalone retrieval query.
Use the previous answer only to resolve pronouns and omitted subjects.
Do not answer the question.
Keep exact document names, years, departments, dates, and fields if mentioned.
If the current question is already standalone, return it unchanged.
Return only the rewritten query.
```

LLM User Prompt:

```text
Previous user question:
{{Session Follow-up Guard.last_question}}

Previous answer:
{{Session Follow-up Guard.last_answer_brief}}

Previous source summary:
{{Session Follow-up Guard.last_source_summary}}

Current question:
{{sys.query}}
```

출력 변수:

```text
standalone_query: string
```

이후 `Query Plan`, `Structured Lookup`, `Final Answer`에서 검색용 질문은 `{{Rewrite Standalone Query.standalone_query}}`를 사용한다. `Knowledge Retrieval`에는 긴 질문 대신 `Query Plan`이 만든 짧은 `knowledge_query`를 사용한다. 최종 답변 프롬프트에는 원 질문과 검색용 질문을 둘 다 넣어 사용자의 실제 의도를 유지한다.

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
  "query": "{{Rewrite Standalone Query.standalone_query}}",
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
    knowledge_queries = data.get("knowledge_queries") or []

    return {
        "query": data.get("query", ""),
        "document_id": data.get("document_id") or "",
        "query_type": data.get("query_type", "semantic"),
        "answer_style": data.get("answer_style", "grounded_explanation"),
        "entities": entities,
        "keywords": keywords,
        "knowledge_query": data.get("knowledge_query") or data.get("query", ""),
        "knowledge_query_secondary": data.get("knowledge_query_secondary") or "",
        "knowledge_queries": knowledge_queries,
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
knowledge_query: string
knowledge_query_secondary: string
knowledge_queries: array[string]
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
Body Type: raw
Raw Body Type: JSON
```

Headers:

```text
Content-Type: application/json
```

권장 방식은 `Structured Lookup` 바로 앞에 `Build Lookup Body` Code 노드를 두고 raw JSON 문자열을 만든 뒤 HTTP Request에 연결하는 것이다. Dify JSON 편집기에서 `entities` 배열 변수를 직접 섞으면 배열이 문자열로 전달될 수 있고, 이 경우 `/lookup`의 `entities: list[string]` 계약과 어긋난다.

#### Code: Build Lookup Body

입력 변수:

```text
query = {{Rewrite Standalone Query.standalone_query}}
document_id = {{Parse Query Plan.document_id}}
entities = {{Parse Query Plan.entities}}
query_type = {{Parse Query Plan.query_type}}
```

Python:

```python
import json

def parse_json_like(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, dict, int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default

def main(query="", document_id="", entities=None, query_type="semantic", **kwargs) -> dict:
    parsed_entities = parse_json_like(entities, [])
    if isinstance(parsed_entities, str):
        parsed_entities = [part.strip() for part in parsed_entities.split(",") if part.strip()]
    if not isinstance(parsed_entities, list):
        parsed_entities = []

    body = {
        "query": str(query or "").strip(),
        "document_id": str(document_id or "").strip(),
        "entities": [str(item) for item in parsed_entities if str(item).strip()],
        "query_type": str(query_type or "semantic").strip(),
        "limit": 8,
    }
    return {"lookup_body_json": json.dumps(body, ensure_ascii=False)}
```

출력 변수 타입:

```text
lookup_body_json: string
```

Structured Lookup Raw Body:

```text
{{Build Lookup Body.lookup_body_json}}
```

### 3.5 Knowledge Retrieval

검색 질의:

```text
{{Parse Query Plan.knowledge_query}}
```

대체 질의:

```text
{{Parse Query Plan.knowledge_query_secondary}}
```

긴 자연어 질문이나 여러 확장 질의를 한 문자열로 합쳐 Knowledge 노드에 넣지 않는다. 임베딩·키워드 양쪽에서 문서 원문과의 정확 일치가 약해질 수 있다. 예를 들어 `수시1차 전형의 원서접수 시작 및 마감 기간은 어떻게 되나요?`는 `원서접수`와 `수시1차`로 축약된다.

기본 구성은 `knowledge_query`를 쓰는 Knowledge Retrieval 노드 하나다. 재현율이 중요한 운영 구성에서는 `knowledge_query_secondary`가 비어 있지 않을 때만 두 번째 Knowledge Retrieval 노드를 조건 분기로 병렬 실행하고, 두 결과를 `Merge Evidence Request`에 함께 전달한다. 병렬 실행이므로 직렬 노드 두 개보다 지연 증가가 작고, 구체 엔터티와 속성 용어 중 한쪽이 임베딩 검색에서 누락되는 경우를 보완한다.

권장 설정:

```text
Retrieval mode: Hybrid Search
Top K: 10
Score threshold: 비활성
Rerank: 비활성
```

현재 `ipsi` 데이터셋의 Ollama `qwen3-embedding:8b` 조합은 가중 재랭킹을 켰을 때 정확 용어 질의도 0건이 되는 현상이 확인됐다. 재랭킹은 운영 평가셋에서 Recall@10과 MRR이 유지되는 모델을 별도로 구성한 뒤에만 활성화한다. Knowledge API의 데이터셋 기본값도 `hybrid_search`, `top_k=10`, `score_threshold_enabled=false`, `reranking_enable=false`로 맞춘다.

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

Standalone retrieval question:
{{Rewrite Standalone Query.standalone_query}}

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

### 3.8.1 Save Conversation State

conversation variables는 `Final Answer` 직후가 아니라 `Parse Verify Result` 뒤의 Conditional 분기에서 최종 출력할 답변이 확정된 뒤 갱신한다. 검증 실패한 `Final Answer.text`를 먼저 저장하면 다음 질문에서 “방금 답변의 출처”를 물었을 때 실패 답변의 상태가 재사용될 수 있다.

정상 분기에서는 `Final Answer.text`를 저장하고, 검증 실패 후 재작성 분기에서는 `Rewrite Answer.text`를 저장한다. Dify의 Variable Assigner 노드를 사용하고, 노드 입력으로 필요한 값을 만들기 위해 Code 노드를 하나 둔다.

노드 이름:

```text
Build Conversation State
```

입력 변수:

```text
question = {{sys.query}}
answer = {{Final Answer.text}} 또는 {{Rewrite Answer.text}}
source_summary = {{Parse Merge Evidence.source_summary}}
```

Python:

```python
import re

def text(value) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if value in {"", '""', "''", "null", "None"}:
        return ""
    return value

def main(question="", answer="", source_summary="") -> dict:
    answer = text(answer).strip()
    source_summary = text(source_summary).strip()
    document_names = []
    for part in re.split(r"[,;]\s*", source_summary.replace("참조 문서:", "")):
        name = part.strip()
        if not name or "제공된 근거 없음" in name:
            continue
        name = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()
        if name and name not in document_names:
            document_names.append(name)

    return {
        "last_question": text(question).strip()[:1000],
        "last_answer_brief": answer[:2000],
        "last_source_summary": source_summary[:2000],
        "last_document_names": ", ".join(document_names)[:1000],
    }
```

Variable Assigner에서 다음처럼 저장한다.

```text
last_question       <- Build Conversation State.last_question
last_answer_brief   <- Build Conversation State.last_answer_brief
last_source_summary <- Build Conversation State.last_source_summary
last_document_names <- Build Conversation State.last_document_names
```

API 점검 기준:

```text
1차 요청: conversation_id = ""
2차 요청: conversation_id = 1차 응답 conversation_id
GET /messages?conversation_id=<id>&user=<same-user> 결과에 두 메시지가 있어야 한다.
GET /conversations/<id>/variables?user=<same-user> 결과에 last_source_summary가 있어야 한다.
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

## 7. API 점검 결과와 필수 수정

2026-07-29 Dify API 기준 점검 결과:

```text
Knowledge API:
- dataset: ipsi
- status: 200
- document_count: 7
- documents:
  - ch01_소프트웨어 공학과 개발 프로세스.pdf: completed/available, enabled=true, segments=116
  - ch03_계획.pdf: completed/available, enabled=true, segments=106
  - ch04_요구분석.pdf: completed/available, enabled=true, segments=100
  - ch05_설계.pdf: completed/available, enabled=true, segments=74
  - ch06_아키텍처 설계와 클래스 설계.pdf: completed/available, enabled=true, segments=91
  - 2.  2027학년도 입학전형 수시·정시 모집요강.pdf: completed/available, enabled=true, segments=158
  - document.pdf: completed/available, enabled=true, segments=270

Knowledge Retrieval API:
- "수시1차 원서접수 기간 알려줘"는 2027 모집요강이 top result로 검색됨
- "총 장학금 얼마야"는 top score가 0.1575로 낮고 document.pdf 장학금/등록금 혼합 chunk가 상위에 옴
- "소프트웨어 개발 프로세스", "요구분석", "아키텍처 설계" 질문은 소프트웨어 문서가 정상 검색됨
- 활성 문서 7개, segment 915개 전수 점검에서 section/page metadata 문제 135개 검출
- 2027 모집요강 112개, document.pdf 22개, ch03_계획.pdf 1개에 집중됨
- 구조화 자식 chunk의 source_page/source_section은 정상 metadata로 인정한 수치임
- 주요 유형은 Untitled/목차/빈 section 123개와 번호 계층 역전 12개임

Chatflow App API:
- /chat-messages status=200
- session conversation_id 재사용 정상
- conversation variables 조회 정상
- 혼합 도메인 평가 4/4 통과
- 입학 문서/연도 평가 8/6 통과
```

판단:

```text
1. Knowledge 적재와 Chatflow 실행은 현재 운영 가능한 상태다.
2. 입학/소프트웨어 혼합 도메인 테스트는 통과했으므로 특정 도메인으로만 쏠리는 문제는 현재 재현되지 않는다.
3. 남은 실패는 `inputs.document_id`가 Chatflow Start 입력으로 들어가지 않아 특정 문서 모드가 완전히 강제되지 않는 문제다.
4. 스트리밍 trace에서 `workflow_started.data.inputs`에 `document_id`가 없고 `Query Plan` 응답의 `document_id`도 null이다.
5. 일부 chunk에 목차/빈 섹션/섹션 경로 ">"가 남아 있어 재적재 전 정규화 상태를 다시 확인해야 한다.
```

2026-08-07 재점검 판단:

```text
1. App API 키는 /info, /parameters, /meta 조회와 /chat-messages 실행에는 유효하다.
2. 같은 App API 키로 /console/api/apps, /console/api/apps/{app_id}, /console/api/apps/{app_id}/workflows/draft를 GET 조회하면 모두 401 Invalid token이다.
3. 따라서 현재 제공된 App API Key와 Knowledge API Key만으로는 Dify 캔버스 노드 구조를 직접 수정할 수 없다.
4. document_id 입력을 보낸 진단에서도 workflow_started.inputs에 document_id가 없고 Query Plan document_id가 null이다.
5. 배포 Chatflow 품질 평가는 12개 중 5개 통과, 7개 실패다. 주요 실패는 document_id 필터 누락, 2026/2027 문서 혼합, 일부 소프트웨어 문서 출처 불일치다.
6. 코드 커밋 15b320d는 중복 적재된 최신 학년도 문서가 여러 document_id로 잡혀도 최신 학년도 후보만 direct answer에 쓰도록 보정했다. 이 효과는 배포 서버 FastAPI에 최신 코드를 반영해야 Chatflow에서 나타난다.
```

필수 수정:

```text
1. 모든 Code 노드에서 Output variables를 실제 return dict와 1:1로 등록한다.
2. Code 노드 입력 변수 목록에 빈 변수 이름 행이 있으면 모두 삭제한다.
3. `main(..., **kwargs)` 형태를 써서 Dify가 예기치 않은 입력 키를 넘겨도 실행이 죽지 않게 한다.
4. `document_id`를 Start 입력 변수로 만들고, 특정 문서 테스트/운영에서는 App API `inputs.document_id`로 반드시 전달한다.
5. `Query Plan` HTTP body와 `Build Lookup Body`가 `Start.document_id`를 받는지 trace에서 확인한다.
6. Knowledge Retrieval에도 같은 범위 필터를 적용한다. 필터 적용이 어렵다면 `document_id`가 있는 특정 문서 모드에서는 Structured Lookup을 authoritative source로 두고 Knowledge Retrieval은 보조 근거로만 쓴다.
7. 전체 Knowledge 모드가 아니라면 2026/2027 문서 중 질문 대상이 아닌 문서를 비활성화하거나 dataset을 분리한다.
8. Knowledge Pipeline에서 최신 `/convert/rag` 산출물로 재적재하고, section이 `>`, `목   차`인 segment가 남는지 Knowledge API로 재확인한다.
9. `Build Lookup Body`, `Build Merge Evidence Body`, `Build Verify Body`, `Build Eval Log Body`는 모두 HTTP JSON 편집기 대신 raw JSON 문자열을 반환하는 Code 노드로 둔다.
```

페이지별 section metadata 보정은 `/convert/rag`에서 다음 순서로 처리한다.

```text
1. PDF parser가 출력한 Markdown heading level을 기본 계층으로 사용한다.
2. Ⅰ/Ⅱ, Chapter/Part, 제N장/절/조, 1./1.1 번호 체계로 잘못된 heading level을 교정한다.
3. 제목 태그가 없는 페이지는 상단의 짧은 시각 제목을 section으로 사용한다.
4. 시각 제목도 없고 표만 있는 페이지는 "Table: <주요 열 이름>"을 범용 section으로 사용한다.
5. 실제 연속 페이지처럼 새 제목이 없는 페이지는 직전 section을 유지한다.
6. 목차, Untitled, 빈 section, 제어문자, 하위 번호 뒤에 상위 번호가 오는 경로는 불량으로 판정한다.
```

배포 및 PDF 재적재 후 다음 명령의 `issues=0`을 확인한다.

```powershell
python -X utf8 scripts\check_dify_api_state.py --sections-only --timeout 180
python -X utf8 scripts\check_dify_api_state.py --timeout 180
```

기존 Knowledge segment는 파서 코드 변경만으로 자동 갱신되지 않는다. 원본 PDF를 Knowledge Pipeline에서 다시 실행해 최신 `/convert/rag`의 `dify_markdown`으로 교체해야 한다. 현재 문서는 `rag-pipeline/local_file` 데이터 소스라 Knowledge API의 ZIP 다운로드 대상이 아니므로, 원본 없이 기존 segment를 직접 덮어쓰지 않는다.

`Not all output parameters are validated` 해결 체크리스트:

```text
Session Follow-up Guard:
- is_source_followup: boolean
- direct_answer: string
- has_previous_answer: boolean
- query: string
- last_question: string
- last_answer_brief: string
- last_source_summary: string
- last_document_names: string

Parse Query Plan:
- query: string
- document_id: string
- query_type: string
- answer_style: string
- entities: array[string]
- keywords: array[string]
- sub_queries: array[string]
- expanded_queries: array[string]
- entities_text: string
- sub_query_text: string
- expanded_query_text: string

Build Lookup Body:
- lookup_body_json: string

Build Merge Evidence Body:
- merge_body_json: string

Parse Merge Evidence:
- structured_context: string
- knowledge_context: string
- evidence_context: string
- evidence_priority: string
- source_summary: string
- answer_contract: string
- answer_contract_text: string
- lookup_diagnostics: string
- lookup_answerability: string

Build Verify Body:
- verify_body_json: string

Parse Verify Result:
- valid: boolean
- confidence: number
- query_type: string
- issues_text: string
- should_answer: boolean
- should_rewrite: boolean

Build Eval Log Body:
- eval_log_body_json: string
```

목표 품질을 ChatGPT/Claude에 문서 전체를 넣은 답변에 가깝게 만들려면, Dify Knowledge Retrieval만으로 단정하지 않는다. 현재 구조처럼 `Structured Lookup -> Knowledge Retrieval -> Merge Evidence -> Verify -> Rewrite`를 유지하고, API 점검에서 다음 조건을 통과해야 운영 가능 상태로 본다.

```text
1. /chat-messages가 200을 반환한다.
2. 모든 답변 끝에 "참조 문서:"가 있다.
3. GET /messages에서 같은 user/conversation_id의 히스토리가 이어진다.
4. GET /conversations/{conversation_id}/variables에 last_source_summary가 저장된다.
5. Knowledge Retrieval top results가 질문 대상 연도/document_id와 일치한다.
6. /eval/log 집계에서 missing_document_id=0, missing_direct_answer 감소, verify_failed가 실제 근거 부족 케이스로 제한된다.
```

### 7.1 배포 Chatflow API 품질 평가

배포된 Dify App API만 사용해서 속도, 출처, 연도/문서 혼합을 확인한다. 로컬 FastAPI를 띄우지 않는다.

```powershell
python -X utf8 scripts\dify_app_api_diagnostics.py --document-id c7b1cb1b-91a3-4a6e-a64b-a90837445f3f --out .runtime\dify_app_api_diagnostics_doc2027_latest.json
python -X utf8 scripts\dify_app_api_diagnostics.py --skip-console-probe --fail-on-stale-deployment --timeout 300
python -X utf8 scripts\eval_dify_chat_api.py --timeout 300
python -X utf8 scripts\eval_dify_chat_api.py --suite mixed --timeout 300
python -X utf8 scripts\eval_dify_chat_api.py --suite all --timeout 300
```

`dify_app_api_diagnostics.py`는 다음 값을 반드시 확인한다.

```text
workflow_edit_available: false이면 API 키만으로 노드 편집 불가
document_id_in_workflow_inputs: true여야 특정 문서 모드 정상
query_plan_document_id: App API inputs.document_id와 같아야 정상
structured_direct_answer: document_id 모드에서는 지정 문서 값만 포함되어야 정상
deployment_contract_ready: true여야 knowledge_query, 조회 prefilter, 응답 경량화가 원격 FastAPI에 배포된 상태
structured_response_chars: 배포 전후 크기를 비교하고 불필요한 수만 자 응답이면 실패 원인을 확인
node_timings: Structured Lookup과 Final Answer가 전체 지연의 대부분인지 확인
verify_valid: true여도 structured_direct_answer가 문서를 섞으면 운영 불합격
```

결과 파일:

```text
.runtime/dify_chat_api_eval_latest.json
.runtime/dify_app_api_diagnostics_doc2027_latest.json
```

2026-08-07 재점검 결과:

```text
Dify App API diagnostics:
- workflow_edit_available: False
- document_id_in_workflow_inputs: False
- query_plan_document_id: None
- console draft workflow probe: 401 Invalid token
- doc2027 입력 진단 elapsed: 22.499s, events=186
- structured_direct_answer가 2027 모집요강과 document.pdf 값을 함께 포함함

Chatflow quality suite all:
- total/passed/failed: 12/5/7
- pass_rate: 41.67%
- avg/max latency: 16.496s / 25.724s
```

2026-07-29 재점검 결과:

```text
mixed suite:
- total/passed/failed: 4/4/0
- pass_rate: 100.00%
- avg/max latency: 29.928s / 44.39s

admission suite:
- total/passed/failed: 8/6/2
- pass_rate: 75.00%
- avg/max latency: 48.997s / 57.291s
```

혼합 도메인 통과:

```text
mixed_admission_quota
mixed_software_process
mixed_software_requirements
mixed_software_architecture
```

입학 문서 통과:

```text
date_default
date_doc2027_filter
quota_default
quota_doc2026_filter
explicit_2027_query
scholarship_total_abstains
```

입학 문서 실패:

```text
date_doc2026_filter:
- inputs.document_id를 2026 document.pdf 내부 id로 넣어도 2027학년도 원서접수 날짜를 답함

quota_doc2027_filter:
- inputs.document_id를 2027 모집요강 내부 id로 넣어도 document.pdf의 93명을 답함
```

판단:

```text
1. 혼합 도메인에서는 입학 질문과 소프트웨어 질문이 서로의 문서를 끌고 오지 않는다.
2. 기본 입학 일정 질문은 최신 2027학년도 모집요강을 선택한다.
3. 특정 문서 모드만 남은 문제다. App API `inputs.document_id`가 Start 입력에 등록되지 않아 trace에서 빠진다.
4. `document_id`가 빠지면 Query Plan/Structured Lookup/Knowledge Retrieval이 전체 Knowledge를 대상으로 검색하므로 연도/문서 혼합 위험이 남는다.
5. 일정 표에서는 "원서접수", "등록금 납부기간", "문서등록", "합격자 발표"가 같은 표 근처에 있어 row/field 선택 검증을 계속 유지해야 한다.
```

운영 수정 우선순위:

```text
1. 연도별 Knowledge Base 또는 연도별 App으로 분리한다.
   - 2026 앱: document.pdf만 enabled
   - 2027 앱: 2027 모집요강만 enabled

2. 하나의 App에서 여러 연도를 유지해야 하면 Chatflow Start에 academic_year 또는 document_id를 필수 hidden 입력으로 만들고, Knowledge Retrieval에도 같은 범위 필터를 적용한다.
   - Dify Knowledge Retrieval 노드에서 metadata filtering을 쓸 수 있으면 document_name 또는 academic_year metadata로 필터링한다.
   - 필터를 Knowledge Retrieval에 적용할 수 없으면 Structured Lookup만으로 답해야 하는 질문에서는 Knowledge Retrieval을 보조 근거로만 쓰고, source_summary도 structured source를 우선한다.

3. 일정 질문은 `/lookup`의 answerability와 answer_field를 Final Answer보다 먼저 검사한다.
   - 질문에 "원서접수"가 있으면 answer_field 또는 evidence row label에 원서접수가 있어야 한다.
   - "등록금", "문서등록", "합격자 발표", "면접예약", "면접고사"가 대신 선택되면 answerable=false로 취급한다.

4. 배포 후 아래 명령이 admission 8/8, mixed 4/4를 모두 통과할 때까지 Publish와 API 재테스트를 반복한다.
```

```powershell
python -X utf8 scripts\eval_dify_chat_api.py --suite admission --timeout 300
python -X utf8 scripts\eval_dify_chat_api.py --suite mixed --timeout 300
python -X utf8 scripts\check_dify_api_state.py --timeout 180
```
