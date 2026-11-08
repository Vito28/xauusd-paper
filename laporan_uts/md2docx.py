# ruff: noqa: E501  (potongan XML Word sengaja ditulis utuh per baris)
"""Markdown sederhana -> XML Word dengan gaya BUSINESS UNDERSTANDING.docx.

Sintaks yang didukung (satu blok per baris kosong):
  # / ## / ###            Heading1 / Heading2 / Heading3
  %title Teks             paragraf Title;  %subtitle Teks -> Subtitle
  %toc                    field daftar isi (Update Field di Word)
  %pagebreak              pemisah halaman
  %image rIdN             salin paragraf gambar rIdN dari dokumen template
  %figure path [cm]       sisipkan PNG dari path (relatif terhadap file markdown), lebar cm
  %cover ... %endcover    halaman sampul (format draf skripsi Mikroskil), baris "kunci: nilai"
  Tabel/Kode/Gambar N.M   caption (miring, rata tengah, keepNext)
  Sumber: ...             catatan sumber kecil
  > teks                  kotak catatan (latar kuning muda)
  $$ rumus $$ (n.m)       rumus OMML linear dengan nomor di kanan
  1) teks / - teks        butir bernomor / butir strip (indentasi gantung)
  | a | b |               tabel (baris pertama = header); <!-- widths: 30,70 --> opsional
  ``` ... ```             blok kode (Consolas, latar abu-abu)
Inline: **tebal**, *miring*, `kode`.
"""

import json
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

TABLE_WIDTH = 9026
CAPTION_RE = re.compile(r"^(Tabel|Kode|Gambar) \d+(\.\d+)? ")
INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)")


def text_run(text: str, bold=False, italic=False, code=False, size=None) -> str:
    props = ""
    if code:
        props += '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>'
    if bold:
        props += "<w:b/><w:bCs/>"
    if italic:
        props += "<w:i/><w:iCs/>"
    if size is not None:
        props += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def inline_runs(text: str, size=None, bold=False, italic=False) -> str:
    runs = []
    for part in INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            runs.append(text_run(part[2:-2], bold=True, italic=italic, size=size))
        elif part.startswith("`") and part.endswith("`"):
            runs.append(text_run(part[1:-1], code=True, bold=bold, italic=italic, size=size))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            runs.append(text_run(part[1:-1], italic=True, bold=bold, size=size))
        else:
            runs.append(text_run(part, bold=bold, italic=italic, size=size))
    return "".join(runs)


def paragraph(inner: str, ppr: str = "") -> str:
    return f"<w:p><w:pPr>{ppr}</w:pPr>{inner}</w:p>" if ppr else f"<w:p>{inner}</w:p>"


def body_paragraph(text: str) -> str:
    return paragraph(inline_runs(text), '<w:jc w:val="both"/>')


def heading(text: str, level: int) -> str:
    # keepNext: style heading template tidak memilikinya, jadi heading bisa tertinggal di bawah halaman.
    return paragraph(inline_runs(text), f'<w:pStyle w:val="Heading{level}"/><w:keepNext/>')


def caption(text: str) -> str:
    ppr = '<w:keepNext/><w:spacing w:before="120" w:after="60"/><w:jc w:val="center"/>'
    return paragraph(inline_runs(text, italic=True), ppr)


def source(text: str) -> str:
    return paragraph(inline_runs(text, size=18), '<w:spacing w:before="0" w:after="160"/>')


def callout(text: str) -> str:
    ppr = (
        '<w:pBdr><w:left w:val="single" w:sz="18" w:space="6" w:color="BF8F00"/></w:pBdr>'
        '<w:shd w:val="clear" w:color="auto" w:fill="FFF2CC"/>'
        '<w:spacing w:before="120" w:after="120"/><w:ind w:left="142" w:right="142"/>'
        '<w:jc w:val="both"/>'
    )
    return paragraph(inline_runs(text), ppr)


def list_item(marker: str, text: str) -> str:
    ppr = '<w:ind w:left="567" w:hanging="283"/><w:jc w:val="both"/>'
    return paragraph(text_run(marker) + "<w:r><w:tab/></w:r>" + inline_runs(text), ppr)


def code_lines(lines: list[str]) -> str:
    ppr = (
        '<w:keepLines/><w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>'
        '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:ind w:left="142"/>'
    )
    out = []
    for index, line in enumerate(lines):
        extra = ppr
        if index == len(lines) - 1:
            extra = ppr.replace('w:after="0"', 'w:after="160"')
        out.append(paragraph(text_run(line, code=True, size=16), extra))
    return "".join(out)


MATH_FONT = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'
SUBSCRIPT_RE = re.compile(r"([^\W_]+\*?)_\{?([^\W_]+)\}?")
WORD_RE = re.compile(r"([A-Za-z]{2,}[0-9]*)")


def math_runs(text: str) -> str:
    """Teks rumus -> run OMML; kata >= 2 huruf (ATR, TP, log) ditulis tegak."""
    runs = []
    for part in WORD_RE.split(text):
        if not part:
            continue
        style = '<m:rPr><m:sty m:val="p"/></m:rPr>' if WORD_RE.fullmatch(part) else ""
        runs.append(f'<m:r>{style}{MATH_FONT}<m:t xml:space="preserve">{escape(part)}</m:t></m:r>')
    return "".join(runs)


def omml(formula: str) -> str:
    """Rumus linear dengan subscript x_t atau x_{ik} menjadi m:sSub."""
    xml = []
    position = 0
    for match in SUBSCRIPT_RE.finditer(formula):
        xml.append(math_runs(formula[position : match.start()]))
        xml.append(
            f"<m:sSub><m:e>{math_runs(match.group(1))}</m:e>"
            f"<m:sub>{math_runs(match.group(2))}</m:sub></m:sSub>"
        )
        position = match.end()
    xml.append(math_runs(formula[position:]))
    return "<m:oMath>" + "".join(xml) + "</m:oMath>"


def equation(formula: str, number: str) -> str:
    ppr = (
        '<w:tabs><w:tab w:val="center" w:pos="4513"/><w:tab w:val="right" w:pos="9026"/></w:tabs>'
        '<w:spacing w:before="120" w:after="120"/>'
    )
    math = omml(formula)
    tail = f"<w:r><w:tab/><w:t>{escape(number)}</w:t></w:r>" if number else ""
    return paragraph("<w:r><w:tab/></w:r>" + math + tail, ppr)


def table(rows: list[list[str]], widths: list[float] | None) -> str:
    columns = max(len(row) for row in rows)
    for row in rows:
        while len(row) < columns:
            row.append("")
    if widths is None:
        lengths = []
        for column in range(columns):
            longest = max(len(row[column]) for row in rows)
            lengths.append(min(max(longest, 6), 60))
        widths = lengths
    total = sum(widths)
    grid = [int(TABLE_WIDTH * value / total) for value in widths]
    grid[-1] = TABLE_WIDTH - sum(grid[:-1])

    xml = (
        '<w:tbl><w:tblPr><w:tblStyle w:val="GridTable4-Accent4"/>'
        f'<w:tblW w:w="{TABLE_WIDTH}" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
        '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" '
        'w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>'
    )
    xml += "".join(f'<w:gridCol w:w="{width}"/>' for width in grid) + "</w:tblGrid>"
    for row_index, row in enumerate(rows):
        trpr = (
            "<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>"
            if row_index == 0
            else "<w:trPr><w:cantSplit/></w:trPr>"
        )
        xml += f"<w:tr>{trpr}"
        for column, cell in enumerate(row):
            xml += f'<w:tc><w:tcPr><w:tcW w:w="{grid[column]}" w:type="dxa"/></w:tcPr>'
            parts = cell.split("<br>")
            for part in parts:
                xml += paragraph(
                    inline_runs(part.strip(), size=19), '<w:spacing w:before="20" w:after="20"/>'
                )
            xml += "</w:tc>"
        xml += "</w:tr>"
    return xml + "</w:tbl>" + paragraph("", '<w:spacing w:before="0" w:after="60"/>')


def toc_entry(level: int, text: str, page) -> str:
    indent = 0 if level == 1 else 360
    ppr = (
        f'<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{TABLE_WIDTH}"/></w:tabs>'
        f'<w:spacing w:before="{80 if level == 1 else 0}" w:after="20"/><w:ind w:left="{indent}"/>'
    )
    number = "" if page is None else str(page)
    runs = inline_runs(text, bold=level == 1) + f"<w:r><w:tab/><w:t>{number}</w:t></w:r>"
    return f"<w:pPr>{ppr}</w:pPr>{runs}"


def toc(entries: list[tuple[int, str, int | None]]) -> str:
    """Daftar isi berupa field TOC yang hasilnya sudah terisi (bisa di-Update Field di Word)."""
    title = paragraph(text_run("Daftar Isi"), '<w:pStyle w:val="TOCHeading"/>')
    begin = (
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
    )
    end = '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    paragraphs = []
    for index, (level, text, page) in enumerate(entries):
        body = toc_entry(level, text, page)
        ppr_end = body.index("</w:pPr>") + len("</w:pPr>")
        prefix = begin if index == 0 else ""
        suffix = end if index == len(entries) - 1 else ""
        paragraphs.append(f"<w:p>{body[:ppr_end]}{prefix}{body[ppr_end:]}{suffix}</w:p>")
    return title + "".join(paragraphs)


def heading_entries(markdown: str) -> list[tuple[int, str]]:
    entries = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            entries.append((1, stripped[2:]))
        elif stripped.startswith("## "):
            entries.append((2, stripped[3:]))
    return entries


# ---------------------------------------------------------------------------
# Gambar dari file PNG dan halaman sampul
# ---------------------------------------------------------------------------

EMU_PER_CM = 360_000


def png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Hanya PNG yang didukung")
    return struct.unpack(">II", data[16:24])


class Media:
    """Kumpulan gambar baru: rId, nama file di word/media/, dan isi file."""

    def __init__(self):
        self.items: list[tuple[str, str, bytes]] = []

    def add(self, data: bytes) -> str:
        rid = f"rIdLap{len(self.items) + 1}"
        self.items.append((rid, f"laporan_{len(self.items) + 1}.png", data))
        return rid


def read_image(reference: str, base: Path) -> bytes:
    """path.png atau dokumen.docx#word/media/imageN.png (gambar diambil dari docx lain)."""
    if "#" in reference:
        archive, member = reference.split("#", 1)
        with zipfile.ZipFile(base / archive) as source:
            return source.read(member)
    return (base / reference).read_bytes()


def drawing(rid: str, width: int, height: int, number: int, name: str) -> str:
    return (
        '<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{width}" cy="{height}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{number}" name="{escape(name)}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{number}" name="{escape(name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width}" cy="{height}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        "</wp:inline></w:drawing></w:r>"
    )


def figure(reference: str, width_cm: float, base: Path, media: Media) -> str:
    data = read_image(reference, base)
    pixels_wide, pixels_high = png_size(data)
    width = int(width_cm * EMU_PER_CM)
    height = int(width * pixels_high / pixels_wide)
    rid = media.add(data)
    number = 1000 + len(media.items)
    ppr = '<w:keepNext/><w:spacing w:before="120" w:after="0"/><w:jc w:val="center"/>'
    return paragraph(drawing(rid, width, height, number, Path(reference).name), ppr)


COVER_FONT = (
    '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
)


def cover_line(text: str, size: int, before: int = 0) -> str:
    ppr = f'<w:spacing w:before="{before}" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/>'
    run = (
        f'<w:r><w:rPr>{COVER_FONT}<w:b/><w:bCs/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
        f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    )
    return paragraph(run, ppr)


def cover(fields: list[tuple[str, str]], base: Path, media: Media, section: str) -> str:
    """Sampul mengikuti halaman pertama draf skripsi: judul, jenis, oleh, anggota, logo, instansi.

    Diakhiri section break sehingga sampul tidak bernomor halaman.
    """
    out = []
    members = [value for key, value in fields if key == "anggota"]
    institution = [value for key, value in fields if key == "instansi"]
    values = dict(fields)
    out.append(cover_line(values["judul"], 28))
    out.append(cover_line(values["jenis"], 24, before=1100))
    if "keterangan" in values:
        out.append(cover_line(values["keterangan"], 24, before=120))
    out.append(cover_line("Oleh:", 24, before=1000))
    if "kelompok" in values:
        out.append(cover_line(values["kelompok"], 24, before=240))
    for index, member in enumerate(members):
        name, nim = (part.strip() for part in member.split("|"))
        out.append(cover_line(name, 24, before=240 if index == 0 else 120))
        out.append(cover_line(f"NIM. {nim}", 24))
    data = read_image(values["logo"], base)
    pixels_wide, pixels_high = png_size(data)
    width = int(4.0 * EMU_PER_CM)  # sama dengan draf skripsi: lebar logo 4 cm
    height = int(width * pixels_high / pixels_wide)
    rid = media.add(data)
    logo_ppr = '<w:spacing w:before="900" w:after="0"/><w:jc w:val="center"/>'
    out.append(paragraph(drawing(rid, width, height, 999, "Logo Universitas Mikroskil"), logo_ppr))
    for index, line in enumerate(institution):
        out.append(cover_line(line, 24, before=900 if index == 0 else 0))
    # Section break: sectPr di paragraf terakhir sampul (tanpa footer -> tanpa nomor halaman).
    cover_section = re.sub(r"<w:footerReference [^>]*/>", "", section)
    cover_section = re.sub(r"<w:pgNumType [^>]*/>", "", cover_section)
    out.append(f"<w:p><w:pPr>{cover_section}</w:pPr></w:p>")
    return "".join(out)


MAX_IMAGE_WIDTH = 5_300_000  # EMU, sedikit di bawah lebar teks A4 (9026 dxa = 5.731.510 EMU)
MAX_IMAGE_HEIGHT = 3_700_000


def image_paragraphs(template_xml: str) -> dict[str, str]:
    """rId -> paragraf gambar INLINE rata tengah, dibuat dari drawing di dokumen template.

    Gambar template berupa anchor (floating) dengan wrap teks; bila disalin apa adanya,
    caption dan paragraf ikut terlipat di sekeliling gambar. Karena itu drawing diubah
    menjadi wp:inline dan diperbesar sampai lebar maksimum dengan rasio aspek tetap.
    """
    found = {}
    for match in re.finditer(r'r:embed="(rId\d+)"', template_xml):
        rid = match.group(1)
        start = template_xml.rfind("<w:drawing>", 0, match.start())
        end = template_xml.find("</w:drawing>", match.start()) + len("</w:drawing>")
        drawing = template_xml[start:end]
        width, height = (
            int(value) for value in re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"', drawing).groups()
        )
        scale = min(MAX_IMAGE_WIDTH / width, MAX_IMAGE_HEIGHT / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        doc_pr = re.search(r"<wp:docPr [^>]*/>", drawing).group(0)
        frame = re.search(r"<wp:cNvGraphicFramePr>.*?</wp:cNvGraphicFramePr>", drawing, re.S)
        graphic = re.search(r"<a:graphic .*?</a:graphic>", drawing, re.S).group(0)
        graphic = re.sub(
            r'<a:ext cx="\d+" cy="\d+"/>', f'<a:ext cx="{new_width}" cy="{new_height}"/>', graphic
        )
        inline = (
            '<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{new_width}" cy="{new_height}"/>'
            '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f"{doc_pr}{frame.group(0) if frame else ''}{graphic}</wp:inline></w:drawing>"
        )
        ppr = '<w:keepNext/><w:spacing w:before="120" w:after="0"/><w:jc w:val="center"/>'
        found[rid] = paragraph(f"<w:r><w:rPr><w:noProof/></w:rPr>{inline}</w:r>", ppr)
    return found


def convert(
    markdown: str,
    images: dict[str, str] | None = None,
    base: Path | None = None,
    media: Media | None = None,
    toc_entries: list | None = None,
    section: str = "",
) -> str:
    lines = markdown.splitlines()
    out = []
    index = 0
    pending_widths = None
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("<!-- widths:"):
            pending_widths = [float(value) for value in stripped[12:-3].split(",")]
            index += 1
            continue
        if stripped.startswith("<!--"):
            # Komentar bisa beberapa baris: lewati sampai baris yang memuat "-->".
            while index < len(lines) and "-->" not in lines[index]:
                index += 1
            index += 1
            continue
        if stripped.startswith("```"):
            block = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index].rstrip("\n"))
                index += 1
            index += 1
            out.append(code_lines(block))
            continue
        if stripped.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row_text = lines[index].strip()
                if not re.fullmatch(r"\|[\s:\-|]+\|", row_text):
                    rows.append([cell.strip() for cell in row_text[1:-1].split("|")])
                index += 1
            out.append(table(rows, pending_widths))
            pending_widths = None
            continue
        if stripped.startswith("### "):
            out.append(heading(stripped[4:], 3))
        elif stripped.startswith("## "):
            out.append(heading(stripped[3:], 2))
        elif stripped.startswith("# "):
            out.append(heading(stripped[2:], 1))
        elif stripped.startswith("%title "):
            out.append(paragraph(inline_runs(stripped[7:]), '<w:pStyle w:val="Title"/>'))
        elif stripped.startswith("%subtitle "):
            out.append(paragraph(inline_runs(stripped[10:]), '<w:pStyle w:val="Subtitle"/>'))
        elif stripped == "%toc":
            out.append(toc(toc_entries or []))
        elif stripped == "%cover":
            fields = []
            index += 1
            while lines[index].strip() != "%endcover":
                key, value = lines[index].split(":", 1)
                fields.append((key.strip(), value.strip()))
                index += 1
            out.append(cover(fields, base, media, section))
        elif stripped.startswith("%figure "):
            parts = stripped.split()
            width_cm = float(parts[2]) if len(parts) > 2 else 15.5
            out.append(figure(parts[1], width_cm, base, media))
        elif stripped == "%pagebreak":
            out.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        elif stripped.startswith("%image "):
            rid = stripped.split()[1]
            out.append(images[rid])
        elif stripped.startswith("$$"):
            match = re.fullmatch(r"\$\$(.+?)\$\$\s*(\(.+\))?", stripped)
            out.append(equation(match.group(1).strip(), match.group(2) or ""))
        elif CAPTION_RE.match(stripped):
            out.append(caption(stripped))
        elif stripped.startswith("Sumber:"):
            out.append(source(stripped))
        elif stripped.startswith("> "):
            text = stripped[2:]
            index += 1
            while index < len(lines) and lines[index].strip().startswith("> "):
                text += " " + lines[index].strip()[2:]
                index += 1
            out.append(callout(text))
            continue
        elif re.match(r"^\d+\) ", stripped):
            marker, text = stripped.split(" ", 1)
            out.append(list_item(marker, text))
        elif stripped.startswith("- "):
            out.append(list_item("–", stripped[2:]))
        else:
            # Paragraf biasa: gabungkan baris berikutnya sampai baris kosong / blok lain.
            text = stripped
            index += 1
            while index < len(lines):
                following = lines[index].strip()
                if not following or following.startswith(
                    ("#", "|", "```", "> ", "- ", "%", "$$", "<!--")
                ):
                    break
                if re.match(r"^\d+\) ", following) or CAPTION_RE.match(following):
                    break
                text += " " + following
                index += 1
            out.append(body_paragraph(text))
            continue
        index += 1
    return "".join(out)


def expand_directives(markdown: str, repo: Path, base: Path) -> str:
    """%code file awal akhir -> blok kode bernomor dari repo; %include file -> isi file."""
    lines = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("%code "):
            _, path, start, end = stripped.split()
            source_lines = (repo / path).read_text(encoding="utf-8").splitlines()
            width = len(end)
            lines.append("```")
            for number in range(int(start), int(end) + 1):
                lines.append(f"{str(number).rjust(width)}  {source_lines[number - 1]}")
            lines.append("```")
        elif stripped.startswith("%include "):
            path = base / stripped.split(maxsplit=1)[1]
            if path.exists():
                lines.append(path.read_text(encoding="utf-8"))
            else:
                lines.append(
                    f"> **DATA BELUM ADA.** Bagian ini dibuat otomatis dari folder runs/ dengan "
                    f"skrip ringkas_runs.py (berkas `{path.name}` belum tersedia)."
                )
        else:
            lines.append(line)
    return "\n".join(lines)


FOOTER_RID = "rIdLapFooter"
FOOTER_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
    '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
    '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
    '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r>'
    '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>'
)
FOOTER_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer"
IMAGE_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"


def build_document(
    template: Path, markdown_path: Path, output: Path, repo: Path, toc_pages: Path | None = None
) -> None:
    """Dokumen baru: styles/theme/numbering dari template, isi dari markdown.

    Isi utama diberi footer nomor halaman (mulai 1 setelah sampul). toc_pages (JSON dari
    perintah toc-pages) mengisi nomor halaman daftar isi.
    """
    with zipfile.ZipFile(template) as source_zip:
        template_xml = source_zip.read("word/document.xml").decode("utf-8")
        rels = source_zip.read("word/_rels/document.xml.rels").decode("utf-8")
        names = source_zip.namelist()
        images = image_paragraphs(template_xml)
        markdown = expand_directives(
            markdown_path.read_text(encoding="utf-8"), repo, markdown_path.parent
        )
        used = set(re.findall(r"^%image (rId\d+)", markdown, re.M))
        pages = json.loads(toc_pages.read_text(encoding="utf-8")) if toc_pages else {}
        entries = [(level, text, pages.get(text)) for level, text in heading_entries(markdown)]

        head = template_xml[: template_xml.find("<w:body>") + len("<w:body>")]
        section = re.search(r"<w:sectPr.*?</w:sectPr>", template_xml, re.S).group(0)
        main_section = section.replace(
            "<w:pgSz",
            f'<w:footerReference w:type="default" r:id="{FOOTER_RID}"/><w:pgSz',
            1,
        )
        # Urutan skema sectPr: ... pgSz, pgMar, ..., pgNumType, cols, ... (pgNumType sebelum cols).
        main_section = main_section.replace("<w:cols", '<w:pgNumType w:start="1"/><w:cols', 1)
        media = Media()
        body = convert(markdown, images, markdown_path.parent, media, entries, section)
        document = head + body + main_section + "</w:body></w:document>"

        # Buang relasi gambar yang tidak dipakai beserta file media-nya.
        kept_media = set()

        def keep_relationship(match):
            relationship = match.group(0)
            rid = re.search(r'Id="(rId\d+)"', relationship).group(1)
            if "/image" in relationship and rid not in used:
                return ""
            if "/image" in relationship:
                kept_media.add("word/" + re.search(r'Target="([^"]+)"', relationship).group(1))
            return relationship

        rels = re.sub(r"<Relationship [^>]*/>", keep_relationship, rels)
        extra = [
            f'<Relationship Id="{FOOTER_RID}" Type="{FOOTER_TYPE}" Target="footer_laporan.xml"/>'
        ]
        for rid, filename, _ in media.items:
            extra.append(
                f'<Relationship Id="{rid}" Type="{IMAGE_TYPE}" Target="media/{filename}"/>'
            )
        rels = rels.replace("</Relationships>", "".join(extra) + "</Relationships>")

        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target_zip:
            for name in names:
                if name.endswith("/"):
                    continue
                if name.startswith("word/media/") and name not in kept_media:
                    continue
                if name == "word/document.xml":
                    target_zip.writestr(name, document)
                elif name == "word/_rels/document.xml.rels":
                    target_zip.writestr(name, rels)
                elif name == "[Content_Types].xml":
                    types = source_zip.read(name).decode("utf-8")
                    override = (
                        '<Override PartName="/word/footer_laporan.xml" ContentType="application/'
                        'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
                    )
                    target_zip.writestr(name, types.replace("</Types>", override + "</Types>"))
                elif name == "docProps/core.xml":
                    core = source_zip.read(name).decode("utf-8")
                    title = output.stem
                    core = re.sub(
                        r"<dc:title>.*?</dc:title>", f"<dc:title>{escape(title)}</dc:title>", core
                    )
                    target_zip.writestr(name, core)
                else:
                    target_zip.writestr(name, source_zip.read(name))
            target_zip.writestr("word/footer_laporan.xml", FOOTER_XML)
            for _, filename, data in media.items:
                target_zip.writestr(f"word/media/{filename}", data)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().upper()


def toc_pages(pdf: Path, markdown_path: Path, repo: Path, output: Path, offset: int = 1) -> None:
    """Cari halaman setiap heading pada PDF hasil render; nomor = halaman PDF − offset (sampul).

    Heading dicari berurutan: setiap heading dicari mulai dari halaman heading sebelumnya,
    dan pencarian dimulai setelah halaman daftar isi.
    """
    markdown = expand_directives(
        markdown_path.read_text(encoding="utf-8"), repo, markdown_path.parent
    )
    entries = heading_entries(markdown)
    count = int(
        re.search(
            r"Pages:\s+(\d+)",
            subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout,
        ).group(1)
    )
    texts = []
    for page in range(1, count + 1):
        result = subprocess.run(
            ["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(pdf), "-"],
            capture_output=True,
            text=True,
        )
        texts.append(normalize(result.stdout))
    # Halaman daftar isi memuat semua judul; mulai mencari setelah halaman terakhir yang memuat "DAFTAR ISI".
    start = 0
    for index, text in enumerate(texts):
        if "DAFTAR ISI" in text:
            start = index + 1
    pages = {}
    current = start
    for _, text in entries:
        target = normalize(text)
        for index in range(current, len(texts)):
            if target in texts[index]:
                pages[text] = index + 1 - offset
                current = index
                break
    output.write_text(json.dumps(pages, indent=1, ensure_ascii=False), encoding="utf-8")
    missing = [text for _, text in entries if text not in pages]
    print(f"{len(pages)} heading ditemukan; tidak ditemukan: {missing}")


def insert_section(
    document_xml_path: Path, markdown_path: Path, after_text: str, before_text: str
) -> None:
    """Sisipkan hasil konversi markdown di antara paragraf yang memuat after_text dan before_text."""
    xml = document_xml_path.read_text(encoding="utf-8")
    anchor = xml.find(before_text)
    if anchor < 0 or xml.find(after_text) < 0 or xml.find(after_text) > anchor:
        raise SystemExit("Penanda tidak ditemukan / urutannya salah")
    paragraph_start = xml.rfind("<w:p ", 0, anchor)
    fragment = convert(markdown_path.read_text(encoding="utf-8"))
    xml = xml[:paragraph_start] + fragment + xml[paragraph_start:]
    document_xml_path.write_text(xml, encoding="utf-8")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "build":
        toc_json = Path(sys.argv[6]) if len(sys.argv) > 6 else None
        build_document(
            Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), toc_json
        )
    elif command == "toc-pages":
        toc_pages(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    elif command == "insert":
        insert_section(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        raise SystemExit("perintah: build | toc-pages | insert")
