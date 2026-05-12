# Dify OpenDataLoader PDF Markdown API

이 프로젝트는 Dify 지식 파이프라인의 API Request/HTTP Request 노드가 호출할 수 있는 PDF -> Markdown 변환 API입니다. 변환은 OpenDataLoader `docling-fast` hybrid를 항상 사용하며, `/convert` 응답 본문은 `success`, `markdown` 두 필드만 반환합니다.

## 실행

```powershell
pip install -r requirements.txt
opendataloader-pdf-hybrid --port 5002 --ocr-lang "ko,en" --enrich-picture-description
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

스캔 PDF가 많으면 하이브리드 백엔드를 아래처럼 실행합니다. OCR 때문에 1분을 넘길 가능성이 더 큽니다.

```powershell
opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang "ko,en" --enrich-picture-description
```

## Dify 노드 설정

지식 파이프라인에서 API Request/HTTP Request 노드를 추가하고 다음 중 하나로 호출합니다.

`multipart/form-data`:

- Method: `POST`
- URL: `http://<converter-host>:8000/convert`
- Body: Form Data, 파일 필드에 PDF 파일 변수 지정
- Response variables: `body.success`, `body.markdown`

Binary body:

- Method: `POST`
- URL: `http://<converter-host>:8000/convert`
- Headers: `Content-Type: application/pdf`, `X-Filename: {{파일명 변수}}`
- Body: Binary, PDF 파일 변수 지정
- Response variables: `body.success`, `body.markdown`

JSON URL:

```json
{
  "file_url": "{{PDF URL 변수}}",
  "filename": "{{파일명 변수}}"
}
```

Dify 노드의 read timeout은 기본 설정 기준 `390~420s`로 설정합니다. API 내부 기본 변환 제한은 360초이고, OpenDataLoader hybrid backend 요청 제한은 300초입니다.

## Parent-child Chunker 추천값

입학요강처럼 표가 많고 질문이 학과명, 모집인원, 기호 설명을 직접 묻는 PDF는 Parent-child 모드를 사용하고 parent는 페이지/섹션 단위, child는 행 요약 단위로 잡습니다.

| 항목 | 추천값 |
| --- | --- |
| Chunk mode | `Parent-child` |
| Index method | `High Quality` |
| Parent mode | `Paragraph` |
| Parent delimiter | `(?m)(?=^--- Page \d+ ---$|^#{1,3}\s+)` |
| Parent maximum length | `3000~4000` |
| Parent overlap | `200` |
| Child delimiter | `\n` |
| Child maximum length | `700` |
| Child overlap | `100` |
| Text preprocessing | 연속 공백/개행 정리 켜기, URL/이메일 제거는 문서 성격에 따라 선택 |

검수 기준:

- Dify Preview Chunks에서 `**표 행 요약**` 아래의 한 줄이 child chunk 하나로 유지되어야 합니다.
- `컴퓨터정보공학과 모집인원` 같은 질문의 child에 `모집단위`, `모집 정원`, 수시/정시 세부 인원이 같이 들어와야 합니다.
- `전공심화 개설 학과` 같은 질문의 child에 `♣ 표시(4년제 학사학위(전공심화)과정 개설 학과)` 목록이 한 줄로 들어와야 합니다.
- parent chunk에는 원본 표와 `표 행 요약`, `표 기호 요약`이 함께 보이는 상태가 좋습니다.

## RAG 저장 품질 설정

- 제목/소제목/본문: OpenDataLoader JSON의 `heading`, `paragraph`, `list` 구조를 Markdown으로 재렌더링합니다.
- 표: OpenDataLoader `--table-method cluster`와 hybrid backend를 사용하고, JSON table rows/cells를 Markdown table로 변환합니다. 표 아래에는 검색용 `표 행 요약`을 추가해 각 행을 `열: 값` 형태로 반복합니다.
- 기호 각주: `♣표시 : ...`처럼 표 밖에 있는 기호 설명을 같은 페이지의 표 행에 붙이고, `표 기호 요약`으로 기호가 붙은 모집단위 목록을 생성합니다.
- 이미지/차트: 이미지 설명까지 필요하면 `ODL_HYBRID_MODE=full`로 실행합니다. 백엔드는 `--enrich-picture-description` 옵션으로 시작해야 이미지 설명이 Markdown 흐름에 들어갑니다.
- Hybrid 강제: 변환 명령에 `--hybrid docling-fast`를 항상 넣고 `--hybrid-fallback`은 사용하지 않습니다.
- 시간 제한: `ODL_CONVERSION_TIMEOUT_SECONDS` 기본값은 `360`, `ODL_HYBRID_TIMEOUT_MS` 기본값은 `300000`입니다.

## 환경 변수

- `ODL_HYBRID_URL`: 기본 `http://localhost:5002`
- `ODL_HYBRID_MODE`: 기본 `auto`, 허용값 `full` 또는 `auto`
- `ODL_CONVERSION_TIMEOUT_SECONDS`: 기본 `360`
- `ODL_HYBRID_TIMEOUT_MS`: 기본 `300000`
- `ODL_MAX_PDF_BYTES`: 기본 `83886080`
- `ODL_TABLE_METHOD`: 기본 `cluster`
- `ODL_USE_STRUCT_TREE`: 기본 `false`

## 테스트

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
