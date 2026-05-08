from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def render_document_to_markdown(doc: dict[str, Any] | None, fallback_markdown: str = "") -> str:
    if not doc:
        return _normalize_markdown(fallback_markdown)

    blocks = []
    for element in _children(doc):
        rendered = _render_element(element)
        if rendered:
            blocks.append(rendered)

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
        if page_markdown:
            blocks.append(f"--- Page {page_number} ---\n\n{page_markdown}")
        else:
            blocks.append(f"--- Page {page_number} ---")
    return _normalize_markdown("\n\n".join(blocks))


def _render_element(element: dict[str, Any], list_depth: int = 0) -> str:
    element_type = _element_type(element)

    if element_type in {"header", "footer"}:
        return ""
    if element_type in {"document", "page", "text block", "block", "section"}:
        return _render_children(element)
    if element_type == "heading":
        return _render_heading(element)
    if element_type in {"paragraph", "text"}:
        return _clean_text(element.get("content", ""))
    if element_type == "caption":
        content = _clean_text(element.get("content", ""))
        return f"*{content}*" if content else ""
    if element_type == "list":
        return _render_list(element, list_depth)
    if element_type == "list item":
        return _render_list_item(element, list_depth)
    if element_type == "table":
        return _render_table(element)
    if element_type in {"image", "picture", "figure"}:
        return _render_image(element)

    content = _clean_text(element.get("content", ""))
    child_content = _render_children(element)
    return "\n\n".join(part for part in (content, child_content) if part)


def _render_heading(element: dict[str, Any]) -> str:
    content = _clean_text(element.get("content", ""))
    if not content:
        return ""
    level = _heading_level(element)
    return f"{'#' * level} {content}"


def _render_children(element: dict[str, Any]) -> str:
    blocks = []
    for child in _children(element):
        rendered = _render_element(child)
        if rendered:
            blocks.append(rendered)
    return "\n\n".join(blocks)


def _render_list(element: dict[str, Any], list_depth: int) -> str:
    items = element.get("list items") or element.get("items") or element.get("kids") or []
    if not isinstance(items, list):
        return ""

    style = str(element.get("numbering style", "")).lower()
    ordered = any(token in style for token in ("ordered", "decimal", "number", "arabic"))
    lines = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        text = _render_list_item(item, list_depth + 1)
        if not text:
            continue
        prefix = f"{index}. " if ordered else "- "
        indent = "  " * list_depth
        continuation = "\n" + indent + "  "
        lines.append(f"{indent}{prefix}{text.replace(chr(10), continuation)}")
    return "\n".join(lines)


def _render_list_item(element: dict[str, Any], list_depth: int) -> str:
    content = _clean_text(element.get("content", ""))
    child_blocks = []
    for child in _children(element):
        rendered = _render_element(child, list_depth)
        if rendered:
            child_blocks.append(rendered)
    return "\n".join(part for part in [content, *child_blocks] if part)


def _render_table(element: dict[str, Any]) -> str:
    rows = element.get("rows") or []
    if not isinstance(rows, list):
        return ""

    matrix = [_table_row_to_cells(row) for row in rows if isinstance(row, dict)]
    matrix = [row for row in matrix if any(cell.strip() for cell in row)]
    if not matrix:
        return ""

    width = max(len(row) for row in matrix)
    matrix = [row + [""] * (width - len(row)) for row in matrix]

    header = matrix[0]
    body = matrix[1:]
    if not any(cell.strip() for cell in header):
        header = [f"Column {index}" for index in range(1, width + 1)]
        body = matrix

    lines = [
        "| " + " | ".join(_escape_table_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


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

    timeline = _render_timeline_page(elements)
    if timeline:
        return timeline

    metric_grid = _render_metric_grid_page(elements)
    if metric_grid:
        return metric_grid

    blocks = []
    has_image = False
    for element in elements:
        if _element_type(element) in {"image", "picture", "figure"}:
            has_image = True
            rendered = _render_element(element)
            if rendered:
                blocks.append(rendered)
            continue
        rendered = _render_element(element)
        if rendered:
            blocks.append(rendered)

    if not blocks and has_image:
        return "> Image-only page. No embedded text layer was available."
    if has_visual_only_content and len(_content_fingerprint("\n".join(blocks))) < 40:
        blocks.append("> \uc774\ubbf8\uc9c0/\ub3c4\uc2dd \uc911\uc2ec \ud398\uc774\uc9c0\ub85c, PDF \ud14d\uc2a4\ud2b8 \ub808\uc774\uc5b4\uc5d0\uc11c \ud655\uc778\ub418\ub294 \ubb38\uad6c\ub9cc \ud3ec\ud568\ud588\uc2b5\ub2c8\ub2e4.")
    return "\n\n".join(blocks)


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
    college_name = "\uc778\ud558\uacf5\uc5c5\uc804\ubb38\ub300\ud559"
    admission_guide = "\ubaa8\uc9d1\uc694\uac15"
    if re.fullmatch(rf"\d+\s+{re.escape(college_name)}(?:\s+\d{{4}}\S*\s+{re.escape(admission_guide)}\s+\d+)?", normalized):
        return True
    if re.fullmatch(rf"\d{{4}}\S*\s+{re.escape(admission_guide)}\s+\d+", normalized):
        return True
    return False


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
    text = str(value)
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
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
