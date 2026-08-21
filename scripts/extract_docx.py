"""Extract plain text + table content from a .docx using only the stdlib.

Used once, during project setup, to read the source manuscript so the pipeline
can be built against the paper's actual terminology, architecture and reported
numbers. Does not modify the manuscript.
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def para_text(p: ET.Element) -> str:
    parts = []
    for node in p.iter():
        if node.tag == f"{W}t":
            parts.append(node.text or "")
        elif node.tag == f"{W}tab":
            parts.append("\t")
        elif node.tag == f"{W}br":
            parts.append("\n")
    return "".join(parts)


def walk_body(body: ET.Element):
    """Yield paragraphs and tables in document order."""
    for child in body:
        if child.tag == f"{W}p":
            txt = para_text(child).strip()
            if txt:
                yield ("p", txt)
        elif child.tag == f"{W}tbl":
            rows = []
            for tr in child.findall(f"{W}tr"):
                cells = []
                for tc in tr.findall(f"{W}tc"):
                    cell = " ".join(
                        para_text(p).strip() for p in tc.findall(f"{W}p")
                    ).strip()
                    cells.append(cell)
                if any(cells):
                    rows.append(cells)
            if rows:
                yield ("tbl", rows)


def main() -> int:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src) as z:
        xml = z.read("word/document.xml")
        media = [n for n in z.namelist() if n.startswith("word/media/")]

    root = ET.fromstring(xml)
    body = root.find(f"{W}body")

    lines = [f"# EXTRACTED FROM: {src.name}", f"# embedded media files: {len(media)}", ""]
    n_tbl = 0
    for kind, payload in walk_body(body):
        if kind == "p":
            lines.append(payload)
        else:
            n_tbl += 1
            lines.append("")
            lines.append(f"<<<TABLE {n_tbl}>>>")
            for row in payload:
                lines.append(" | ".join(c.replace("\n", " ") for c in row))
            lines.append(f"<<<END TABLE {n_tbl}>>>")
            lines.append("")

    text = "\n".join(lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}  ({len(text):,} chars, {n_tbl} tables, {len(media)} media)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
