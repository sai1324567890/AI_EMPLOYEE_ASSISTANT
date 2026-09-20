
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    doc_path: str          # relative path, e.g. "policies/expense_policy.md"
    doc_title: str         # top-level "# Title" of the file
    section: str           # "## Section" heading this chunk belongs to
    text: str              # section body text
    route: str = ""        # which route/knowledge source this chunk belongs to
    citation: str = field(init=False)

    def __post_init__(self):
        fname = Path(self.doc_path).name
        if self.section:
            self.citation = f"{fname}#{self.section}"
        else:
            self.citation = fname


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _split_sections(text: str):
    """Split a markdown document into (level, heading, body) tuples."""
    matches = list(_HEADER_RE.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((level, heading, body))
    return sections


def load_markdown_file(path: Path, route: str) -> List[Chunk]:
    raw = path.read_text(encoding="utf-8")
    sections = _split_sections(raw)
    if not sections:
        return [Chunk(doc_path=str(path), doc_title=path.stem, section="", text=raw.strip(), route=route)]

    # first section (level 1) is the document title
    doc_title = sections[0][1] if sections[0][0] == 1 else path.stem
    chunks = []
    for level, heading, body in sections:
        if level == 1:
            # Title-level section with its own leading text (rare) -> skip if empty
            if body:
                chunks.append(Chunk(doc_path=str(path), doc_title=doc_title,
                                     section=heading, text=body, route=route))
            continue
        if not body:
            continue
        chunks.append(Chunk(doc_path=str(path), doc_title=doc_title,
                             section=heading, text=body, route=route))
    return chunks


def load_corpus(route_sources: dict) -> List[Chunk]:
    """
    route_sources: mapping of route_name -> list[Path] (folders)
    Returns a flat list of Chunk objects across all routes.
    """
    all_chunks: List[Chunk] = []
    for route, folders in route_sources.items():
        for folder in folders:
            if not folder.exists():
                continue
            for md_file in sorted(folder.glob("*.md")):
                all_chunks.extend(load_markdown_file(md_file, route))
    return all_chunks
