"""Dump the contents of an .xlsx using only the stdlib (zip + XML).

Used to inspect dataset1's `patient details.xlsx` before any dependency is
installed, so we can decide whether patient-level splitting is possible.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def col_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def main() -> int:
    path = Path(sys.argv[1])
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{NS}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))

        sheets = [n for n in names if n.startswith("xl/worksheets/") and n.endswith(".xml")]
        print(f"workbook: {path.name}")
        print(f"sheets: {sheets}")
        print(f"sharedStrings: {len(shared)}")

        for sheet in sheets:
            print(f"\n===== {sheet} =====")
            root = ET.fromstring(z.read(sheet))
            rows = root.iter(f"{NS}row")
            n_total = 0
            for i, row in enumerate(rows):
                n_total += 1
                if i >= max_rows:
                    continue
                cells: dict[int, str] = {}
                for c in row.findall(f"{NS}c"):
                    ref = c.get("r", "")
                    ctype = c.get("t")
                    v = c.find(f"{NS}v")
                    isel = c.find(f"{NS}is")
                    if ctype == "s" and v is not None:
                        val = shared[int(v.text)]
                    elif ctype == "inlineStr" and isel is not None:
                        val = "".join(t.text or "" for t in isel.iter(f"{NS}t"))
                    elif v is not None:
                        val = v.text or ""
                    else:
                        val = ""
                    if val.strip():
                        cells[col_index(ref)] = val.strip()
                if cells:
                    width = max(cells) + 1
                    print(" | ".join(cells.get(k, "") for k in range(width)))
            print(f"-- total rows in sheet: {n_total} --")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
