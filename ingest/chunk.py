"""Heading-aware, token-bounded Markdown chunks with citable breadcrumbs."""
import re
from collections.abc import Callable

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"\s+", "-", text.strip())


def strip_mdx(text: str) -> str:
    text = re.sub(r"\A---\r?\n.*?\r?\n---(?:\r?\n|$)", "", text, flags=re.S)
    result, fence = [], None
    for line in text.splitlines():
        match = FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            result.append(line)
        elif fence is not None or not re.match(r"^(import|export)\s", line):
            result.append(line)
    return "\n".join(result)


def embedding_text(chunk: dict) -> str:
    return f"{chunk['breadcrumb']}\n{chunk['text']}"


def _blocks(lines: list[str]) -> list[str]:
    blocks, cur, fence = [], [], None
    for line in lines:
        match = FENCE.match(line)
        if match and fence is None:
            if cur:
                blocks.append("\n".join(cur).strip("\n"))
                cur = []
            fence = match.group(1)
        elif match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
            cur.append(line)
            blocks.append("\n".join(cur))
            cur, fence = [], None
            continue
        cur.append(line)
        if not fence and not line.strip():
            blocks.append("\n".join(cur).strip("\n"))
            cur = []
    if cur:
        if fence:
            cur.append(fence)
        blocks.append("\n".join(cur).strip("\n"))
    return [b for b in blocks if b.strip()]


def _split_text(text: str, fits: Callable[[str], bool]) -> list[str]:
    """Prefer complete lines/words, but bound even a single huge command line."""
    out = []
    while text:
        if fits(text):
            out.append(text)
            break
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if fits(text[:mid]):
                low = mid
            else:
                high = mid - 1
        if not low:
            raise ValueError("Chunk budget is too small for the heading and code fence")
        boundary = max(text.rfind("\n", 0, low), text.rfind(" ", 0, low))
        cut = boundary + 1 if boundary >= low // 2 else low
        # WordPiece counts need not be monotonic when a word is shortened.
        while cut and not fits(text[:cut]):
            cut -= 1
        if not cut:
            raise ValueError("Chunk budget cannot fit a source character")
        out.append(text[:cut])
        text = text[cut:]
    return out


def _split_block(block: str, fits: Callable[[str], bool]) -> list[str]:
    if fits(block):
        return [block]
    lines = block.splitlines()
    if FENCE.match(lines[0]):
        opening = lines[0]
        marker = FENCE.match(opening).group(1)
        content = "\n".join(lines[1:-1])
        wrap = lambda part: f"{opening}\n{part}\n{marker}"
        return [wrap(part) for part in _split_text(content, lambda part: fits(wrap(part)))]
    return _split_text(block, fits)


def chunk_markdown(
    text: str, url: str, title: str, source_type: str,
    product: str = "general", max_chars: int = 2800, min_chars: int = 40,
    *, max_tokens: int = 384, token_counter: Callable[[str], int] | None = None,
) -> list[dict]:
    # UTF-8 bytes give a deliberately conservative bound without loading models.
    count = token_counter or (lambda value: len(value.encode("utf-8")) + 2)
    sections = []
    stack, cur_lines, cur_crumb, cur_anchor, fence = [], [], [title], "", None

    def flush():
        if any(line.strip() for line in cur_lines):
            sections.append((list(cur_crumb), cur_anchor, list(cur_lines)))

    for line in strip_mdx(text).splitlines():
        match = FENCE.match(line)
        if match and fence is None:
            fence = match.group(1)
        elif match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
            fence = None
        heading = HEADING.match(line) if not fence and not match else None
        if heading:
            flush()
            level, name = len(heading.group(1)), heading.group(2)
            stack = [(level_, name_) for level_, name_ in stack if level_ < level] + [(level, name)]
            cur_crumb = [title] + [name_ for _, name_ in stack]
            cur_anchor, cur_lines = slugify(name), []
        else:
            cur_lines.append(line)
    flush()

    chunks = []
    for crumb, anchor, lines in sections:
        breadcrumb = " > ".join(dict.fromkeys(crumb))
        while breadcrumb and count(f"{breadcrumb}\n```\nx\n```") > max_tokens // 2:
            breadcrumb = breadcrumb[:max(0, len(breadcrumb) * 3 // 4)]
        fits = lambda body: len(body.strip()) <= max_chars and count(f"{breadcrumb}\n{body.strip()}") <= max_tokens
        bodies, buf = [], ""
        for block in _blocks(lines):
            for part in _split_block(block, fits):
                candidate = f"{buf}\n\n{part}" if buf else part
                if buf and not fits(candidate):
                    bodies.append(buf)
                    buf = part
                else:
                    buf = candidate
        if buf:
            bodies.append(buf)
        for body in bodies:
            if not fits(body):
                raise ValueError("Chunk exceeds the configured token or character budget")
            if len(body.strip()) < min_chars and len(bodies) == 1:
                continue
            chunks.append({
                "url": url, "anchor": anchor, "title": title,
                "breadcrumb": breadcrumb, "text": body.strip(),
                "source_type": source_type, "product": product,
            })
    return chunks
