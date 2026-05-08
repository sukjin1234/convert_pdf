from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_PARENT_DELIMITER = "<<<PARENT_BREAK>>>"
DEFAULT_CHILD_DELIMITER = "<<<CHILD_BREAK>>>"
DEFAULT_PARENT_MAX_CHARS = 4500
DEFAULT_CHILD_TARGET_CHARS = 900
DEFAULT_CHILD_OVERLAP_CHARS = 180

_PAGE_RE = re.compile(r"^--- Page (\d+) ---$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class ChunkingOptions:
    parent_delimiter: str = DEFAULT_PARENT_DELIMITER
    child_delimiter: str = DEFAULT_CHILD_DELIMITER
    parent_max_chars: int = DEFAULT_PARENT_MAX_CHARS
    child_target_chars: int = DEFAULT_CHILD_TARGET_CHARS
    child_overlap_chars: int = DEFAULT_CHILD_OVERLAP_CHARS


@dataclass(frozen=True)
class Block:
    text: str
    page: int | None = None
    heading_level: int | None = None
    heading_text: str | None = None

    @property
    def is_page_marker(self) -> bool:
        return _PAGE_RE.fullmatch(self.text.strip()) is not None


@dataclass(frozen=True)
class ParentSection:
    blocks: list[Block]
    title: str


def prepare_parent_child_markdown(markdown: str, options: ChunkingOptions | None = None) -> str:
    options = options or ChunkingOptions()
    markdown = _normalize(markdown)
    if not markdown:
        return ""
    if options.parent_delimiter in markdown or options.child_delimiter in markdown:
        return markdown

    parents = _build_parent_sections(_split_blocks(markdown), options.parent_max_chars)
    rendered_parents = []
    for parent in parents:
        children = _build_child_sections(parent.blocks, options.child_target_chars)
        if not children:
            continue

        rendered_children = []
        for index, child_blocks in enumerate(children):
            previous_context = _context_snippet(
                children[index - 1] if index > 0 else [],
                options.child_overlap_chars,
                from_end=True,
            )
            rendered_children.append(options.child_delimiter)
            rendered_children.append(
                _render_child(parent, child_blocks, previous_context)
            )

        rendered_parents.append(
            "\n\n".join([options.parent_delimiter, *rendered_children])
        )

    return _normalize("\n\n".join(rendered_parents)) or markdown


def _build_parent_sections(blocks: list[Block], max_chars: int) -> list[ParentSection]:
    parents: list[ParentSection] = []
    current: list[Block] = []
    heading_path: dict[int, str] = {}
    fallback_title = "Document section"

    def finish_current() -> None:
        nonlocal current
        if not _has_body_content(current):
            return
        parents.append(ParentSection(blocks=current, title=_title_for_blocks(current, fallback_title)))
        current = []

    for block in blocks:
        if block.heading_level is not None and block.heading_text:
            next_heading_path = _updated_heading_path(heading_path, block.heading_level, block.heading_text)
            starts_new_parent = block.heading_level <= 2 and _has_body_content(current)
            if starts_new_parent:
                carry = []
                if current and current[-1].is_page_marker:
                    carry.append(current.pop())
                finish_current()
                current = carry
                if not current and block.page is not None:
                    _append_page_marker(current, block.page)
                fallback_title = _heading_path_title(next_heading_path)
            heading_path = next_heading_path
            if fallback_title == "Document section":
                fallback_title = _heading_path_title(heading_path)
            current.append(block)
            continue

        if (
            max_chars > 0
            and _blocks_length(current) + len(block.text) > max_chars
            and _has_body_content(current)
            and not block.is_page_marker
        ):
            finish_current()
            if block.page is not None:
                _append_page_marker(current, block.page)

        current.append(block)

    finish_current()
    return parents


def _build_child_sections(blocks: list[Block], target_chars: int) -> list[list[Block]]:
    children: list[list[Block]] = []
    current: list[Block] = []

    def finish_current() -> None:
        nonlocal current
        if not _has_body_content(current):
            return
        children.append(current)
        current = []

    for block in blocks:
        if (
            target_chars > 0
            and _blocks_length(current) + len(block.text) > target_chars
            and _has_body_content(current)
            and not block.is_page_marker
        ):
            finish_current()
            if block.page is not None:
                _append_page_marker(current, block.page)

        current.append(block)

    finish_current()
    return children


def _render_child(
    parent: ParentSection,
    child_blocks: list[Block],
    previous_context: str,
) -> str:
    parts = [
        f"Section: {parent.title}",
        f"Source: {_source_pages(child_blocks) or _source_pages(parent.blocks) or 'Unknown page'}",
    ]
    if previous_context:
        parts.append(f"> Previous context: {previous_context}")
    parts.append("Content:")
    parts.append(_blocks_markdown(child_blocks))
    return "\n\n".join(part for part in parts if part).strip()


def _split_blocks(markdown: str) -> list[Block]:
    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()]
    blocks: list[Block] = []
    current_page: int | None = None

    for raw_block in raw_blocks:
        page_match = _PAGE_RE.fullmatch(raw_block)
        if page_match:
            current_page = int(page_match.group(1))
            blocks.append(Block(text=raw_block, page=current_page))
            continue

        heading_match = _HEADING_RE.match(raw_block)
        heading_level = None
        heading_text = None
        if heading_match:
            heading_level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

        blocks.append(
            Block(
                text=raw_block,
                page=current_page,
                heading_level=heading_level,
                heading_text=heading_text,
            )
        )

    return blocks


def _append_page_marker(blocks: list[Block], page: int) -> None:
    marker = f"--- Page {page} ---"
    if blocks and blocks[-1].text == marker:
        return
    blocks.append(Block(text=marker, page=page))


def _updated_heading_path(heading_path: dict[int, str], level: int, text: str) -> dict[int, str]:
    updated = {key: value for key, value in heading_path.items() if key < level}
    updated[level] = text
    return updated


def _heading_path_title(heading_path: dict[int, str]) -> str:
    values = [heading_path[level] for level in sorted(heading_path)]
    return " > ".join(values) if values else "Document section"


def _title_for_blocks(blocks: list[Block], fallback: str) -> str:
    heading_path: dict[int, str] = {}
    for block in blocks:
        if block.heading_level is not None and block.heading_text:
            heading_path = _updated_heading_path(heading_path, block.heading_level, block.heading_text)
    return _heading_path_title(heading_path) if heading_path else fallback


def _source_pages(blocks: list[Block]) -> str:
    pages = sorted({block.page for block in blocks if block.page is not None})
    if not pages:
        return ""
    ranges: list[str] = []
    start = pages[0]
    previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(_format_page_range(start, previous))
        start = page
        previous = page
    ranges.append(_format_page_range(start, previous))
    return "Page " + ", ".join(ranges)


def _format_page_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def _has_body_content(blocks: list[Block]) -> bool:
    for block in blocks:
        if block.is_page_marker or block.heading_level is not None:
            continue
        if _content_fingerprint(block.text):
            return True
    return False


def _blocks_length(blocks: list[Block]) -> int:
    return len(_blocks_markdown(blocks))


def _blocks_markdown(blocks: list[Block]) -> str:
    return "\n\n".join(block.text for block in blocks).strip()


def _context_snippet(blocks: list[Block], max_chars: int, *, from_end: bool) -> str:
    if max_chars <= 0 or not blocks:
        return ""
    text = _plain_context_text(blocks)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    if from_end:
        return "..." + text[-max_chars:].lstrip()
    return text[:max_chars].rstrip() + "..."


def _plain_context_text(blocks: list[Block]) -> str:
    text = _blocks_markdown(
        [block for block in blocks if not block.is_page_marker]
    )
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _content_fingerprint(value: str) -> str:
    return "".join(re.findall(r"[0-9A-Za-z\uac00-\ud7a3]+", value.lower()))


def _normalize(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
