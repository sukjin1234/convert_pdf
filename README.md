# Dify OpenDataLoader PDF Markdown API

이 프로젝트는 Dify 지식 파이프라인의 API Request/HTTP Request 노드가 호출할 수 있는 PDF -> Markdown 변환 API입니다. 변환은 OpenDataLoader `docling-fast` hybrid를 항상 사용하며, `/convert` 응답 본문은 `success`, `markdown` 두 필드만 반환합니다.

## 실행

```powershell
pip install -r requirements.txt
opendataloader-pdf-hybrid --port 5002 --ocr-lang "ko,en" --enrich-picture-description
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`use_ocr=true`로 요청하면 API가 별도 OCR 전용 hybrid 서버를 임시로 실행합니다. 기본 포트는 `5003`이며, 변환이 끝나면 서버 프로세스를 종료합니다.

```powershell
opendataloader-pdf-hybrid --port 5003 --force-ocr --ocr-lang "ko,en" --enrich-picture-description
```

## Dify 노드 설정

지식 파이프라인에서 API Request/HTTP Request 노드를 추가하고 다음 중 하나로 호출합니다.

`multipart/form-data`:

- Method: `POST`
- URL: `http://<converter-host>:8000/convert`
- Body: Form Data, 파일 필드에 PDF 파일 변수 지정
- Optional Form Data: `use_ocr` boolean, 이미지 OCR이 필요할 때 `true`
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

Dify 노드의 read timeout은 `70~90s`로 설정합니다. API 내부 기본 변환 제한은 70초이고, OpenDataLoader hybrid backend 요청 제한은 60초입니다.

## RAG 저장 품질 설정

- 제목/소제목/본문: OpenDataLoader JSON의 `heading`, `paragraph`, `list` 구조를 Markdown으로 재렌더링합니다.
- 표: OpenDataLoader `--table-method cluster`와 hybrid backend를 사용하고, JSON table rows/cells를 Markdown table로 변환합니다.
- 이미지/차트: 클라이언트는 `--hybrid-mode full`로 실행됩니다. 백엔드는 `--enrich-picture-description` 옵션으로 시작해야 이미지 설명이 Markdown 흐름에 들어갑니다.
- OCR: `use_ocr=true`이면 기본 변환 결과의 JSON에서 이미지 요소가 있는 페이지만 OCR 대상으로 삼습니다. OCR 텍스트는 해당 이미지 요소가 있던 Markdown 위치에 삽입됩니다.
- Hybrid 강제: 변환 명령에 `--hybrid docling-fast`를 항상 넣고 `--hybrid-fallback`은 사용하지 않습니다.
- 시간 제한: `ODL_CONVERSION_TIMEOUT_SECONDS` 기본값은 `70`, `ODL_HYBRID_TIMEOUT_MS` 기본값은 `60000`입니다.

## 환경 변수

- `ODL_HYBRID_URL`: 기본 `http://localhost:5002`
- `ODL_HYBRID_MODE`: 기본 `full`, 허용값 `full` 또는 `auto`
- `ODL_CONVERSION_TIMEOUT_SECONDS`: 기본 `70`
- `ODL_HYBRID_TIMEOUT_MS`: 기본 `60000`
- `ODL_MAX_PDF_BYTES`: 기본 `83886080`
- `ODL_TABLE_METHOD`: 기본 `cluster`
- `ODL_USE_STRUCT_TREE`: 기본 `false`
- `ODL_OCR_SERVER_CLI`: 기본 `opendataloader-pdf-hybrid`
- `ODL_OCR_SERVER_HOST`: 기본 `127.0.0.1`
- `ODL_OCR_SERVER_PORT`: 기본 `5003`
- `ODL_OCR_SERVER_WORKERS`: 기본 `0`, `ODL_OCR_SERVER_DEVICES`/`ODL_OCR_SERVER_PORTS`/자동 GPU 감지 개수 사용
- `ODL_OCR_SERVER_PORTS`: 예 `5003,5004,5005`, 지정하지 않으면 `ODL_OCR_SERVER_PORT`부터 연속 포트 사용
- `ODL_OCR_SERVER_DEVICES`: 예 `0,1,2` 또는 `cuda0,cuda1,cuda2`, 각 OCR 서버 프로세스의 `CUDA_VISIBLE_DEVICES`; 지정하지 않으면 `nvidia-smi`로 최대 3개 자동 감지
- `ODL_OCR_AUTO_DETECT_CUDA`: 기본 `true`, `false`이면 GPU 자동 감지 비활성화
- `ODL_OCR_AUTO_DETECT_MAX_GPUS`: 기본 `3`
- `ODL_OCR_HYBRID_URL`: 기본 `http://127.0.0.1:<ODL_OCR_SERVER_PORT>`
- `ODL_OCR_HYBRID_URLS`: 예 `http://127.0.0.1:5003,http://127.0.0.1:5004,http://127.0.0.1:5005`
- `ODL_OCR_LANG`: 기본 `ko,en`
- `ODL_OCR_ENRICH_PICTURE_DESCRIPTION`: 기본 `true`
- `ODL_OCR_SERVER_START_TIMEOUT_SECONDS`: 기본 `60`
- `ODL_OCR_SERVER_SHUTDOWN_TIMEOUT_SECONDS`: 기본 `10`

기본적으로 `nvidia-smi`에서 GPU 3개가 감지되면 OCR 서버를 `5003`, `5004`, `5005`에 자동으로 띄워 병렬 처리합니다. 명시적으로 고정하려면 다음처럼 실행합니다.

```bash
export ODL_OCR_SERVER_DEVICES=0,1,2
export ODL_OCR_SERVER_PORTS=5003,5004,5005
```

## 테스트

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
