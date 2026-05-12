from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any


MARKER_SYMBOLS = (
    "\u2663"  # club suit
    "\u25c6"  # black diamond
    "\u25c7"  # white diamond
    "\u25a0"  # black square
    "\u25a1"  # white square
    "\u25b2"  # black up-pointing triangle
    "\u25b3"  # white up-pointing triangle
    "\u25cf"  # black circle
    "\u25cb"  # white circle
    "\u25ce"  # bullseye
    "\u25c8"  # white diamond containing black small diamond
    "\u2605"  # black star
    "\u2606"  # white star
    "\u25b6"  # black right-pointing triangle
    "\u25b7"  # white right-pointing triangle
    "\u25a3"  # white square containing black small square
)
MARKER_SYMBOL_PATTERN = f"[{re.escape(MARKER_SYMBOLS)}]"
PARENT_OVERLAP_LINE_LIMIT = 2
PARENT_OVERLAP_CHAR_LIMIT = 280
CHILD_CONTEXT_CHAR_LIMIT = 240
TABLE_ROW_OVERLAP_CHAR_LIMIT = 180
SEARCH_FACT_CHAR_LIMIT = 650


def render_document_to_markdown(doc: dict[str, Any] | None, fallback_markdown: str = "") -> str:
    if not doc:
        return _normalize_markdown(fallback_markdown)

    blocks = []
    heading_context: list[str] = []
    for element in _children(doc):
        rendered = _render_element(element, heading_context=heading_context)
        if rendered:
            blocks.append(rendered)
        if _element_type(element) == "heading":
            heading_context = _updated_heading_context(heading_context, element)

    markdown = "\n\n".join(blocks).strip()
    if not markdown:
        markdown = fallback_markdown
    return _normalize_markdown(markdown)


def render_document_pages_to_markdown(doc: dict[str, Any] | None, fallback_markdown: str = "") -> str:
    if not doc:
        return _normalize_markdown(fallback_markdown)

    pages: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for element in _children(doc):
        page_number = _safe_int(element.get("page number"), 0)
        if page_number > 0:
            pages[page_number].append(element)

    if not pages:
        return render_document_to_markdown(doc, fallback_markdown)

    blocks = []
    for page_number in sorted(pages):
        page_markdown = _render_page(pages[page_number])
        blocks.append((page_number, page_markdown))
    return _normalize_markdown("\n\n".join(_render_pages_with_parent_overlap(blocks)))


def _render_pages_with_parent_overlap(pages: list[tuple[int, str]]) -> list[str]:
    blocks = []
    previous_tail = ""
    for page_number, page_markdown in pages:
        page_blocks = [f"--- Page {page_number} ---"]
        if previous_tail:
            page_blocks.append(f"> 이전 parent overlap: {previous_tail}")
        if page_markdown:
            page_blocks.append(page_markdown)
        blocks.append("\n\n".join(page_blocks))
        previous_tail = _parent_overlap_tail(page_markdown)
    return blocks


def _parent_overlap_tail(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        line = _overlap_line_text(line)
        if line:
            lines.append(line)
    if not lines:
        return ""

    tail = []
    total = 0
    for line in reversed(lines):
        line_length = len(line)
        if tail and total + line_length > PARENT_OVERLAP_CHAR_LIMIT:
            break
        tail.append(line)
        total += line_length
        if len(tail) >= PARENT_OVERLAP_LINE_LIMIT:
            break
    return " / ".join(reversed(tail))


def _overlap_line_text(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if line.startswith("|") or re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", line):
        return ""
    if line.startswith("**표 행 요약**") or line.startswith("**표 기호 요약**"):
        return ""
    line = re.sub(r"^#{1,6}\s+", "", line)
    line = re.sub(r"^[-*]\s+", "", line)
    line = re.sub(r"^>\s*", "", line)
    return _record_cell_text(line)


def _render_element(
    element: dict[str, Any],
    list_depth: int = 0,
    symbol_legends: dict[str, str] | None = None,
    heading_context: list[str] | None = None,
) -> str:
    element_type = _element_type(element)

    if element_type in {"header", "footer"}:
        return ""
    if element_type in {"document", "page", "text block", "block", "section"}:
        return _render_children(element, symbol_legends, heading_context)
    if element_type == "heading":
        return _render_heading(element)
    if element_type in {"paragraph", "text"}:
        return _clean_text(element.get("content", ""))
    if element_type == "caption":
        content = _clean_text(element.get("content", ""))
        return f"*{content}*" if content else ""
    if element_type == "list":
        return _render_list(element, list_depth, symbol_legends, heading_context)
    if element_type == "list item":
        return _render_list_item(element, list_depth, symbol_legends, heading_context)
    if element_type == "table":
        return _render_table(element, symbol_legends, heading_context)
    if element_type in {"image", "picture", "figure"}:
        return _render_image(element)

    content = _clean_text(element.get("content", ""))
    child_content = _render_children(element, symbol_legends, heading_context)
    return "\n\n".join(part for part in (content, child_content) if part)


def _render_heading(element: dict[str, Any]) -> str:
    content = _clean_text(element.get("content", ""))
    if not content:
        return ""
    level = _heading_level(element)
    return f"{'#' * level} {content}"


def _render_children(
    element: dict[str, Any],
    symbol_legends: dict[str, str] | None = None,
    heading_context: list[str] | None = None,
) -> str:
    blocks = []
    for child in _children(element):
        rendered = _render_element(child, symbol_legends=symbol_legends, heading_context=heading_context)
        if rendered:
            blocks.append(rendered)
    return "\n\n".join(blocks)


def _render_list(
    element: dict[str, Any],
    list_depth: int,
    symbol_legends: dict[str, str] | None = None,
    heading_context: list[str] | None = None,
) -> str:
    items = element.get("list items") or element.get("items") or element.get("kids") or []
    if not isinstance(items, list):
        return ""

    style = str(element.get("numbering style", "")).lower()
    ordered = any(token in style for token in ("ordered", "decimal", "number", "arabic"))
    lines = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        text = _render_list_item(item, list_depth + 1, symbol_legends, heading_context)
        if not text:
            continue
        prefix = f"{index}. " if ordered else "- "
        indent = "  " * list_depth
        continuation = "\n" + indent + "  "
        lines.append(f"{indent}{prefix}{text.replace(chr(10), continuation)}")
    return "\n".join(lines)


def _render_list_item(
    element: dict[str, Any],
    list_depth: int,
    symbol_legends: dict[str, str] | None = None,
    heading_context: list[str] | None = None,
) -> str:
    content = _clean_text(element.get("content", ""))
    child_blocks = []
    for child in _children(element):
        rendered = _render_element(child, list_depth, symbol_legends, heading_context)
        if rendered:
            child_blocks.append(rendered)
    return "\n".join(part for part in [content, *child_blocks] if part)


def _render_table(
    element: dict[str, Any],
    symbol_legends: dict[str, str] | None = None,
    heading_context: list[str] | None = None,
) -> str:
    rows = element.get("rows") or []
    if not isinstance(rows, list):
        return ""

    matrix = [_table_row_to_cells(row) for row in rows if isinstance(row, dict)]
    matrix = [row for row in matrix if any(cell.strip() for cell in row)]
    if not matrix:
        return ""

    width = max(len(row) for row in matrix)
    matrix = [row + [""] * (width - len(row)) for row in matrix]

    header_rows = _table_header_rows(matrix)
    header = header_rows[0]
    body = matrix[len(header_rows) :]
    if not any(cell.strip() for cell in header):
        header = [f"Column {index}" for index in range(1, width + 1)]
        header_rows = [header]
        body = matrix

    lines = [
        "| " + " | ".join(_escape_table_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |")

    records = _render_table_records(header_rows, body, symbol_legends or {}, heading_context or [])
    if records:
        lines.extend(["", *records])
    return "\n".join(lines)


def _table_header_rows(matrix: list[list[str]]) -> list[list[str]]:
    header_rows = [matrix[0]]
    if len(matrix) > 2 and _looks_like_header_continuation(matrix[0], matrix[1]):
        header_rows.append(matrix[1])
    return header_rows


def _looks_like_header_continuation(header: list[str], row: list[str]) -> bool:
    values = [_record_cell_text(cell) for cell in row if _record_cell_text(cell)]
    if len(values) < 2:
        return False

    short_values = [value for value in values if len(value) <= 24]
    numeric_values = [value for value in values if re.search(r"\d", value)]
    has_blank_header_cells = any(not cell.strip() for cell in header)
    has_leading_blanks = sum(1 for cell in row[: min(4, len(row))] if not cell.strip()) >= 2
    mostly_short_text = len(short_values) / len(values) >= 0.75 and len(numeric_values) / len(values) <= 0.35
    return mostly_short_text and (has_blank_header_cells or has_leading_blanks)


def _render_table_records(
    header_rows: list[list[str]],
    body: list[list[str]],
    symbol_legends: dict[str, str],
    heading_context: list[str],
) -> list[str]:
    if not body:
        return []

    labels = _table_column_labels(header_rows)
    context = _format_child_context(heading_context)
    record_entries = []
    search_facts = []
    symbol_hits: dict[str, list[str]] = defaultdict(list)
    carried_context: dict[int, str] = {}

    for row in body:
        values = []
        for index, cell in enumerate(row):
            value = _record_cell_text(cell)
            if value:
                carried_context[index] = value
            elif index == 0 and carried_context.get(index):
                value = carried_context[index]
            values.append(value)

        if not any(values):
            continue

        row_text = " ".join(values)
        row_label = _table_row_label(labels, values)
        parts = []
        if context:
            parts.append(f"문맥: {context}")
        parts.extend(
            f"{label}: {value}"
            for label, value in zip(labels, values)
            if label and value and not _label_repeats_value(label, value)
        )
        for symbol, legend in symbol_legends.items():
            if symbol in row_text:
                parts.append(f"{symbol}: {legend}")
                symbol_labels = _symbol_row_labels(values, symbol)
                symbol_hits[symbol].extend(symbol_labels or ([row_label] if row_label else []))

        if parts:
            record = "; ".join(parts)
            record_entries.append(record)
            search_fact = _render_search_fact(context, labels, values, record)
            if search_fact:
                search_facts.append(search_fact)

    lines = ["**표 행 요약**", *_render_table_records_with_overlap(record_entries)]

    legend_lines = []
    for symbol, labels_for_symbol in symbol_hits.items():
        names = _unique_preserving_order(labels_for_symbol)
        if names:
            legend_lines.append(f"- {symbol} 표시({symbol_legends[symbol]}): {', '.join(names)}")

    if legend_lines:
        lines.extend(["", "**표 기호 요약**", *legend_lines])
        search_facts.extend(_render_symbol_search_facts(symbol_legends, symbol_hits))

    if search_facts:
        lines.extend(["", "**검색 보강**", *_unique_preserving_order(search_facts)])

    return lines if len(lines) > 1 else []


def _render_table_records_with_overlap(records: list[str]) -> list[str]:
    lines = []
    for index, record in enumerate(records):
        overlap_parts = []
        if index > 0:
            overlap_parts.append(f"이전 행 overlap: {_table_row_overlap_text(records[index - 1])}")
        if index + 1 < len(records):
            overlap_parts.append(f"다음 행 overlap: {_table_row_overlap_text(records[index + 1])}")

        suffix = f"; {'; '.join(overlap_parts)}" if overlap_parts else ""
        lines.append(f"- {record}{suffix}")
    return lines


def _table_row_overlap_text(record: str) -> str:
    compact = _record_cell_text(record)
    if len(compact) <= TABLE_ROW_OVERLAP_CHAR_LIMIT:
        return compact
    return compact[: TABLE_ROW_OVERLAP_CHAR_LIMIT - 1].rstrip() + "…"


def _format_child_context(heading_context: list[str]) -> str:
    context = " > ".join(_unique_preserving_order([item for item in heading_context if item]))
    if len(context) <= CHILD_CONTEXT_CHAR_LIMIT:
        return context
    return context[: CHILD_CONTEXT_CHAR_LIMIT - 1].rstrip() + "…"


def _render_search_fact(context: str, labels: list[str], values: list[str], record: str) -> str:
    row_label = _table_row_label(labels, values)
    keywords = _search_keywords(context, labels, values, row_label)
    if not keywords:
        return ""
    fact = f"- 검색 키워드: {', '.join(keywords)} | 답변 근거: {record}"
    return _truncate_search_fact(fact)


def _render_symbol_search_facts(symbol_legends: dict[str, str], symbol_hits: dict[str, list[str]]) -> list[str]:
    facts = []
    for symbol, labels_for_symbol in symbol_hits.items():
        names = _unique_preserving_order(labels_for_symbol)
        if not names:
            continue
        legend = symbol_legends[symbol]
        keywords = _unique_preserving_order(
            [
                legend,
                _compact_text(legend),
                "상세 정보",
                "자세한 내용",
                "세부 내용",
                "관련 정보",
                "목록",
                "대상",
                "표시",
                *names,
            ]
        )
        evidence = f"{symbol} 표시({legend}): {', '.join(names)}"
        facts.append(_truncate_search_fact(f"- 검색 키워드: {', '.join(keywords)} | 답변 근거: {evidence}"))
    return facts


def _search_keywords(context: str, labels: list[str], values: list[str], row_label: str) -> list[str]:
    keywords = [
        context,
        _compact_text(context),
        row_label,
        _compact_text(row_label),
        "상세 정보",
        "자세한 내용",
        "세부 내용",
        "관련 정보",
        "표 상세",
        "표 내용",
        "값",
        "항목",
        "details",
    ]
    for label, value in zip(labels, values):
        if label:
            keywords.extend([label, _compact_text(label)])
        if _looks_like_row_label_value(value):
            cleaned_value = _strip_marker_symbols(value)
            keywords.extend([cleaned_value, _compact_text(cleaned_value)])
    return _unique_preserving_order([keyword for keyword in keywords if keyword])


def _truncate_search_fact(fact: str) -> str:
    if len(fact) <= SEARCH_FACT_CHAR_LIMIT:
        return fact
    return fact[: SEARCH_FACT_CHAR_LIMIT - 1].rstrip() + "…"


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def _table_column_labels(header_rows: list[list[str]]) -> list[str]:
    width = max(len(row) for row in header_rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in header_rows]
    top_labels = _propagate_group_headers(normalized_rows[0], normalized_rows[1:])

    labels = []
    for index in range(width):
        parts = []
        for value in [top_labels[index], *[row[index] for row in normalized_rows[1:]]]:
            value = _record_cell_text(value)
            if value and value not in parts:
                parts.append(value)
        labels.append(" ".join(parts) if parts else f"Column {index + 1}")
    return labels


def _propagate_group_headers(header: list[str], continuation_rows: list[list[str]]) -> list[str]:
    propagated = []
    last = ""
    for index, cell in enumerate(header):
        value = _record_cell_text(cell)
        if value:
            last = value
            propagated.append(value)
            continue

        continuation_has_value = any(index < len(row) and _record_cell_text(row[index]) for row in continuation_rows)
        propagated.append(last if last and continuation_has_value else "")
    return propagated


def _table_row_label(labels: list[str], values: list[str]) -> str:
    candidates = [_strip_marker_symbols(value) for value in values if _looks_like_row_label_value(value)]
    if not candidates:
        return ""
    return max(candidates, key=_row_label_score)


def _symbol_row_labels(values: list[str], symbol: str) -> list[str]:
    labels_for_symbol = []
    for value in values:
        if symbol not in value:
            continue
        labels_for_symbol.extend(_split_symbol_marked_labels(value, symbol))
    return _unique_preserving_order(labels_for_symbol)


def _split_symbol_marked_labels(value: str, symbol: str) -> list[str]:
    labels = []
    start = 0
    for match in re.finditer(re.escape(symbol), value):
        label = _strip_marker_symbols(value[start : match.start()])
        if label:
            labels.append(label)
        start = match.end()
    return labels


def _record_cell_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _looks_like_row_label_value(value: str) -> bool:
    value = _strip_marker_symbols(value)
    if not value:
        return False
    if re.fullmatch(r"[\d\s.,%()·/\-+]+", value):
        return False
    return bool(re.search(r"[A-Za-z\uac00-\ud7a3]", value))


def _row_label_score(value: str) -> tuple[int, int]:
    fingerprint = _content_fingerprint(value)
    return (len(fingerprint), len(value))


def _label_repeats_value(label: str, value: str) -> bool:
    return _content_fingerprint(label) == _content_fingerprint(value)


def _strip_marker_symbols(value: str) -> str:
    value = re.sub(rf"\s*{MARKER_SYMBOL_PATTERN}\s*", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        key = _content_fingerprint(value)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _table_row_to_cells(row: dict[str, Any]) -> list[str]:
    cells = row.get("cells") or row.get("kids") or []
    if not isinstance(cells, list):
        return []

    rendered: list[str] = []
    next_column = 1
    for cell in sorted((cell for cell in cells if isinstance(cell, dict)), key=_cell_sort_key):
        column = _safe_int(cell.get("column number"), next_column)
        while next_column < column:
            rendered.append("")
            next_column += 1

        rendered.append(_render_table_cell(cell))
        next_column += 1

        for _ in range(max(_safe_int(cell.get("column span"), 1) - 1, 0)):
            rendered.append("")
            next_column += 1
    return rendered


def _render_table_cell(cell: dict[str, Any]) -> str:
    content = _clean_text(cell.get("content", ""))
    child_parts = []
    for child in _children(cell):
        rendered = _render_element(child)
        if rendered:
            child_parts.append(_strip_block_markdown(rendered))
    return "<br>".join(part for part in [content, *child_parts] if part)


def _render_image(element: dict[str, Any]) -> str:
    description = _clean_text(element.get("description", "") or element.get("content", ""))
    if description:
        return f"**Image summary:** {description}"
    return ""


def _render_page(elements: list[dict[str, Any]]) -> str:
    has_visual_only_content = any(_element_has_image(element) and not _element_has_text(element) for element in elements)
    elements = [element for element in elements if _is_page_content_element(element)]
    symbol_legends = _extract_symbol_legends(elements)

    timeline = _render_timeline_page(elements)
    if timeline:
        return timeline

    metric_grid = _render_metric_grid_page(elements)
    if metric_grid:
        return metric_grid

    blocks = []
    has_image = False
    heading_context: list[str] = []
    for element in elements:
        if _element_type(element) in {"image", "picture", "figure"}:
            has_image = True
            continue
        rendered = _render_element(element, symbol_legends=symbol_legends, heading_context=heading_context)
        if rendered:
            blocks.append(rendered)
        if _element_type(element) == "heading":
            heading_context = _updated_heading_context(heading_context, element)

    if not blocks and has_image:
        return "> Image-only page. No embedded text layer was available."
    if has_visual_only_content and len(_content_fingerprint("\n".join(blocks))) < 40:
        blocks.append("> \uc774\ubbf8\uc9c0/\ub3c4\uc2dd \uc911\uc2ec \ud398\uc774\uc9c0\ub85c, PDF \ud14d\uc2a4\ud2b8 \ub808\uc774\uc5b4\uc5d0\uc11c \ud655\uc778\ub418\ub294 \ubb38\uad6c\ub9cc \ud3ec\ud568\ud588\uc2b5\ub2c8\ub2e4.")
    return "\n\n".join(blocks)


def _updated_heading_context(context: list[str], element: dict[str, Any]) -> list[str]:
    content = _clean_text(element.get("content", ""))
    if not content:
        return context

    level = max(_heading_level(element), 1)
    index = min(level - 1, len(context))
    return [*context[:index], content][-4:]


def _is_page_content_element(element: dict[str, Any]) -> bool:
    element_type = _element_type(element)
    if element_type in {"header", "footer"}:
        return False
    if _looks_like_running_page_label(_clean_text(element.get("content", ""))):
        return False
    if element_type == "table" and not _element_has_text(element):
        return False
    return True


def _looks_like_running_page_label(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return False
    if (
        len(normalized) <= 60
        and re.search(r"(?:19|20)\d{2}", normalized)
        and re.search(r"\s\d{1,4}$", normalized)
        and not re.search(r"[.!?;:]", normalized)
    ):
        return True
    return False


def _extract_symbol_legends(elements: list[dict[str, Any]]) -> dict[str, str]:
    text = "\n".join(_element_text_lines(element) for element in elements)
    legends = {}
    marker_words = r"(?:표시|기호|mark|marker|symbol|legend)?"
    for match in re.finditer(rf"({MARKER_SYMBOL_PATTERN})\s*{marker_words}\s*[:：\-\u2013]\s*([^\n]+)", text, re.IGNORECASE):
        symbol = match.group(1)
        legend = _clean_text(match.group(2))
        if legend:
            legends[symbol] = legend
    return legends


def _element_text_lines(element: dict[str, Any]) -> str:
    parts = []
    content = _clean_text(element.get("content", ""))
    if content:
        parts.append(content)

    for child in _children(element):
        text = _element_text_lines(child)
        if text:
            parts.append(text)

    if _element_type(element) == "list":
        for child in element.get("list items") or element.get("items") or []:
            if isinstance(child, dict):
                text = _element_text_lines(child)
                if text:
                    parts.append(text)

    return "\n".join(parts)


def _render_metric_grid_page(elements: list[dict[str, Any]]) -> str:
    items = _text_items(elements)
    labels = [item for item in items if _is_metric_label(item)]
    values = [item for item in items if _is_metric_value(item)]

    pairs = []
    used_values: set[int] = set()
    for label in labels:
        candidates = [
            value
            for value in values
            if value["index"] not in used_values
            and _same_visual_column(label, value, tolerance=95.0)
            and 0.0 < label["cy"] - value["cy"] < 95.0
        ]
        if not candidates:
            continue
        value = min(candidates, key=lambda candidate: abs(label["cy"] - candidate["cy"]) + abs(label["cx"] - candidate["cx"]) / 2)
        used_values.add(value["index"])
        pairs.append((label, value))

    if len(pairs) < 4:
        return ""

    paired_indexes = {item["index"] for pair in pairs for item in pair}
    lead_items = [item for item in items if item["index"] not in paired_indexes and item["index"] < min(paired_indexes)]

    blocks = _render_lead_items(lead_items)
    rows = sorted(pairs, key=lambda pair: (-pair[0]["cy"], pair[0]["cx"]))
    item_header = "\ud56d\ubaa9"
    value_header = "\ub0b4\uc6a9"
    table = [f"| {item_header} | {value_header} |", "| --- | --- |"]
    for label, value in rows:
        table.append(f"| {_escape_table_cell(label['text'])} | {_escape_table_cell(value['text'])} |")
    blocks.append("\n".join(table))
    return "\n\n".join(block for block in blocks if block)


def _render_timeline_page(elements: list[dict[str, Any]]) -> str:
    items = _text_items(elements)
    years = [item for item in items if _is_year(item["text"])]
    events = [item for item in items if _is_event_text(item["text"])]

    if len(years) < 4 or len(events) < 4:
        return ""

    first_timeline_index = min(item["index"] for item in [*years, *events])
    lead_items = [item for item in items if item["index"] < first_timeline_index and not _is_page_noise(item["text"])]

    entries: dict[str, list[str]] = defaultdict(list)
    for event in events:
        year = _nearest_timeline_year(event, years)
        if not year:
            continue
        event_text = re.sub(r"^\s*[-\u2022]\s*", "", event["text"]).strip()
        if event_text and event_text not in entries[year["text"]]:
            entries[year["text"]].append(event_text)

    if len(entries) < 4:
        return ""

    blocks = _render_lead_items(lead_items)
    for year in sorted(entries, key=lambda value: int(value)):
        events_text = "; ".join(entries[year])
        blocks.append(f"- {year}: {events_text}")
    return "\n\n".join(block for block in blocks if block)


def _render_lead_items(items: list[dict[str, Any]]) -> list[str]:
    blocks = []
    for index, item in enumerate(items):
        text = item["text"]
        if not text or _is_page_noise(text):
            continue
        if index == 0 and len(text) <= 60:
            blocks.append(f"## {text}")
        else:
            blocks.append(text)
    return blocks


def _nearest_timeline_year(event: dict[str, Any], years: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        year
        for year in years
        if _same_visual_column(year, event, tolerance=125.0)
        and 0.0 <= year["cy"] - event["cy"] < 145.0
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda year: abs(year["cy"] - event["cy"]) + abs(year["x0"] - event["x0"]) / 2)


def _text_items(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for element in elements:
        items.extend(_text_items_from_element(element))

    rendered = []
    for index, item in enumerate(items):
        text = item["text"]
        if not text or _is_page_noise(text):
            continue
        bbox = item["bbox"]
        rendered.append(
            {
                **item,
                "index": index,
                "x0": bbox[0],
                "y0": bbox[1],
                "x1": bbox[2],
                "y1": bbox[3],
                "cx": (bbox[0] + bbox[2]) / 2,
                "cy": (bbox[1] + bbox[3]) / 2,
            }
        )
    return rendered


def _text_items_from_element(element: dict[str, Any]) -> list[dict[str, Any]]:
    element_type = _element_type(element)
    if element_type in {"image", "picture", "figure", "table", "header", "footer"}:
        return []

    items = []
    content = _clean_text(element.get("content", ""))
    bbox = _bounding_box(element)
    if content and bbox:
        items.append(
            {
                "text": content,
                "type": element_type,
                "bbox": bbox,
                "font_size": _safe_float(element.get("font size"), 0.0),
            }
        )

    for child in _children(element):
        items.extend(_text_items_from_element(child))
    if element_type == "list":
        for child in element.get("list items") or element.get("items") or []:
            if isinstance(child, dict):
                items.extend(_text_items_from_element(child))
    return items


def _is_metric_label(item: dict[str, Any]) -> bool:
    text = item["text"]
    return (
        item["type"] in {"paragraph", "text"}
        and 1 <= len(text) <= 24
        and not _is_year(text)
        and not _is_event_text(text)
        and not re.search(r"\d", text)
    )


def _is_metric_value(item: dict[str, Any]) -> bool:
    text = item["text"]
    return (
        item["type"] == "heading"
        and item["font_size"] >= 14
        and bool(re.search(r"\d", text))
    )


def _same_visual_column(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> bool:
    return (
        abs(left["cx"] - right["cx"]) <= tolerance
        or abs(left["x0"] - right["x0"]) <= tolerance
        or left["x0"] - 20 <= right["x0"] <= left["x1"] + 20
        or right["x0"] - 20 <= left["x0"] <= right["x1"] + 20
    )


def _is_year(text: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d{2}", text.strip()))


def _is_event_text(text: str) -> bool:
    return bool(re.match(r"^\s*[-\u2022]\s*\S+", text))


def _is_page_noise(text: str) -> bool:
    return _looks_like_running_page_label(text)


def _content_fingerprint(value: str) -> str:
    return "".join(re.findall(r"[0-9A-Za-z\uac00-\ud7a3]+", value.lower()))


def _element_has_text(element: dict[str, Any]) -> bool:
    if _clean_text(element.get("content", "")):
        return True
    nested_keys = ("kids", "children", "rows", "cells", "list items", "items")
    for key in nested_keys:
        values = element.get(key) or []
        if not isinstance(values, list):
            continue
        if any(isinstance(child, dict) and _element_has_text(child) for child in values):
            return True
    return False


def _element_has_image(element: dict[str, Any]) -> bool:
    if _element_type(element) in {"image", "picture", "figure"}:
        return True
    nested_keys = ("kids", "children", "rows", "cells", "list items", "items")
    for key in nested_keys:
        values = element.get(key) or []
        if not isinstance(values, list):
            continue
        if any(isinstance(child, dict) and _element_has_image(child) for child in values):
            return True
    return False


def _element_type(element: dict[str, Any]) -> str:
    return str(element.get("type", "")).strip().lower()


def _bounding_box(element: dict[str, Any]) -> list[float] | None:
    bbox = element.get("bounding box")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        return [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None


def _children(element: dict[str, Any]) -> list[dict[str, Any]]:
    kids = element.get("kids") or element.get("children") or []
    if not isinstance(kids, list):
        return []
    return [kid for kid in kids if isinstance(kid, dict)]


def _heading_level(element: dict[str, Any]) -> int:
    if "heading level" in element:
        raw_level = _safe_int(element.get("heading level"), 1)
        if raw_level > 6:
            font_size = _safe_float(element.get("font size"), 0.0)
            if font_size >= 16:
                return 2
            if font_size >= 11:
                return 3
            return 4
        return min(max(raw_level, 1), 6)

    raw = str(element.get("level", "")).lower()
    match = re.search(r"(\d+)", raw)
    if match:
        return min(max(int(match.group(1)), 1), 6)
    return 2


def _cell_sort_key(cell: dict[str, Any]) -> tuple[int, int]:
    return (
        _safe_int(cell.get("row number"), 1),
        _safe_int(cell.get("column number"), 1),
    )


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _strip_block_markdown(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^#{1,6}\s+", "", value)
    return value.replace("\n\n", "<br>").replace("\n", "<br>")


def _escape_table_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", "<br>").strip()


def _normalize_markdown(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
