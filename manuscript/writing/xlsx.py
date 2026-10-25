"""A minimal, dependency-free writer of Office Open XML workbooks (.xlsx) for the supplemental tables (Tables S2-S4).

No spreadsheet package is installed here and none is downloaded: a workbook is a zip of a few XML parts, written directly.
Cells are numbers or inline strings (no shared-string table); the first row of a data sheet is a bold, frozen header; a
legend sheet wraps its text. The zip entries carry a fixed date and order, so a rebuild is byte-identical. read() parses a
written workbook back (every part must be well-formed XML) so the builder can compare it with what it wrote.

    write(path, [Sheet("Legend", rows=[...], widths=[...], wrap=True), Sheet("Tests", header=[...], rows=[...])])
"""
import math
import re
import zipfile
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
DATE = (1980, 1, 1, 0, 0, 0)                     # every zip entry: a fixed timestamp, for byte-identical rebuilds
BAD = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")  # characters XML 1.0 forbids
STYLES = (f'<styleSheet xmlns="{NS}">'
          '<fonts count="2"><font><sz val="10"/><name val="Arial"/></font><font><b/><sz val="10"/><name val="Arial"/></font></fonts>'
          '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
          '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
          '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
          '<cellXfs count="4">'
          '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
          '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
          '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>'
          '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1">'
          '<alignment wrapText="1" vertical="top"/></xf>'
          '</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
PLAIN, BOLD, WRAP, BOLD_WRAP = 0, 1, 2, 3


@dataclass
class Sheet:
    name: str
    rows: list                                    # list of rows; a row is a list of str, int, float, bool or None
    header: list = None                           # bold and frozen when given
    widths: list = field(default_factory=list)    # column widths in characters
    wrap: bool = False                            # wrap text (legend sheets)


def col(i):
    """0 -> A, 25 -> Z, 26 -> AA."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _cell(ref, v, style):
    if hasattr(v, "item") and not isinstance(v, str):         # numpy scalars
        v = v.item()
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, bool):
        v = "yes" if v else "no"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if not math.isfinite(v):
            raise ValueError(f"{ref}: a non-finite number cannot be written ({v})")
        num = repr(int(v)) if isinstance(v, int) or float(v).is_integer() and abs(v) < 1e15 else repr(float(v))
        return f'<c r="{ref}" s="{style}"><v>{num}</v></c>'
    text = BAD.sub("", str(v))
    space = ' xml:space="preserve"' if text != text.strip() or "\n" in text else ""
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t{space}>{escape(text)}</t></is></c>'


def _sheet(s):
    rows = ([s.header] if s.header else []) + list(s.rows)
    out = [f'<worksheet xmlns="{NS}">']
    if s.header:
        out.append('<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" '
                   'state="frozen"/></sheetView></sheetViews>')
    out.append('<sheetFormatPr defaultRowHeight="13"/>')
    if s.widths:
        out.append("<cols>" + "".join(f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>' for i, w in enumerate(s.widths))
                   + "</cols>")
    out.append("<sheetData>")
    for r, row in enumerate(rows, 1):
        head = bool(s.header) and r == 1
        style = (BOLD_WRAP if s.wrap else BOLD) if head else (WRAP if s.wrap else PLAIN)
        cells = "".join(_cell(f"{col(c)}{r}", v, style) for c, v in enumerate(row))
        out.append(f'<row r="{r}">{cells}</row>')
    out.append("</sheetData></worksheet>")
    return "".join(out)


def write(path, sheets):
    names = [s.name for s in sheets]
    if len(set(names)) != len(names) or any(len(n) > 31 or re.search(r"[\[\]:*?/\\]", n) for n in names):
        raise ValueError(f"sheet names must be unique, at most 31 characters and free of []:*?/\\: {names}")
    head = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    parts = {
        "[Content_Types].xml": head + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                  'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(sheets) + 1))
        + '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>",
        "_rels/.rels": head + f'<Relationships xmlns="{PKG}"><Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>",
        "xl/workbook.xml": head + f'<workbook xmlns="{NS}" xmlns:r="{REL}"><sheets>'
        + "".join(f'<sheet name="{escape(n)}" sheetId="{i}" r:id="rId{i}"/>' for i, n in enumerate(names, 1)) + "</sheets></workbook>",
        "xl/_rels/workbook.xml.rels": head + f'<Relationships xmlns="{PKG}">'
        + "".join(f'<Relationship Id="rId{i}" Type="{REL}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1))
        + f'<Relationship Id="rId{len(sheets) + 1}" Type="{REL}/styles" Target="styles.xml"/></Relationships>',
        "xl/styles.xml": head + STYLES,
    }
    for i, s in enumerate(sheets, 1):
        parts[f"xl/worksheets/sheet{i}.xml"] = head + _sheet(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for name, text in parts.items():
            info = zipfile.ZipInfo(name, date_time=DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            z.writestr(info, text.encode("utf-8"))


def read(path):
    """{sheet name: rows of cell values (str or float)}; raises if any part is not well-formed XML."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            ET.fromstring(z.read(name))                           # well-formed, or ParseError
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        for i, sh in enumerate(wb.iter(f"{{{NS}}}sheet"), 1):
            root = ET.fromstring(z.read(f"xl/worksheets/sheet{i}.xml"))
            rows = []
            for row in root.iter(f"{{{NS}}}row"):
                vals = {}
                for c in row.iter(f"{{{NS}}}c"):
                    ci = 0
                    for ch in re.match(r"[A-Z]+", c.get("r")).group(0):
                        ci = ci * 26 + ord(ch) - 64
                    if c.get("t") == "inlineStr":
                        vals[ci - 1] = "".join(t.text or "" for t in c.iter(f"{{{NS}}}t"))
                    else:
                        vals[ci - 1] = float(c.find(f"{{{NS}}}v").text)
                rows.append([vals.get(k) for k in range(max(vals) + 1)] if vals else [])
            out[sh.get("name")] = rows
    return out
