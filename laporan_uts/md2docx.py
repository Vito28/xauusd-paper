"""Markdown sederhana -> XML Word dengan gaya BUSINESS UNDERSTANDING.docx.

Sintaks yang didukung (satu blok per baris kosong):
  # / ## / ###            Heading1 / Heading2 / Heading3
  %title Teks             paragraf Title;  %subtitle Teks -> Subtitle
  %toc                    field daftar isi (Update Field di Word)
  %pagebreak              pemisah halaman
  %image rIdN             salin paragraf gambar rIdN dari dokumen template
  Tabel/Kode/Gambar N.M   caption (miring, rata tengah, keepNext)
  Sumber: ...             catatan sumber kecil
  > teks                  kotak catatan (latar kuning muda)
  $$ rumus $$ (n.m)       rumus OMML linear dengan nomor di kanan
  1) teks / - teks        butir bernomor / butir strip (indentasi gantung)
  | a | b |               tabel (baris pertama = header); <!-- widths: 30,70 --> opsional
  ``` ... ```             blok kode (Consolas, latar abu-abu)
Inline: **tebal**, *miring*, `kode`.
"""

import re
import shutil
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
        trpr = "<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>" if row_index == 0 else "<w:trPr><w:cantSplit/></w:trPr>"
        xml += f"<w:tr>{trpr}"
        for column, cell in enumerate(row):
            xml += f'<w:tc><w:tcPr><w:tcW w:w="{grid[column]}" w:type="dxa"/></w:tcPr>'
            parts = cell.split("<br>")
            for part in parts:
                xml += paragraph(inline_runs(part.strip(), size=19), '<w:spacing w:before="20" w:after="20"/>')
            xml += "</w:tc>"
        xml += "</w:tr>"
    return xml + "</w:tbl>" + paragraph("", '<w:spacing w:before="0" w:after="60"/>')


def toc() -> str:
    title = paragraph(text_run("Daftar Isi"), '<w:pStyle w:val="TOCHeading"/>')
    field = (
        '<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        + text_run("Klik kanan di sini lalu pilih Update Field untuk menampilkan daftar isi.", italic=True)
        + '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
    )
    return title + field


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
        width, height = (int(value) for value in re.search(
            r'<wp:extent cx="(\d+)" cy="(\d+)"', drawing).groups())
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


def convert(markdown: str, images: dict[str, str] | None = None) -> str:
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
            out.append(toc())
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
                if not following or following.startswith(("#", "|", "```", "> ", "- ", "%", "$$", "<!--")):
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


def build_document(template: Path, markdown_path: Path, output: Path, repo: Path) -> None:
    """Dokumen baru: styles/theme/numbering dari template, isi dari markdown."""
    with zipfile.ZipFile(template) as source_zip:
        template_xml = source_zip.read("word/document.xml").decode("utf-8")
        rels = source_zip.read("word/_rels/document.xml.rels").decode("utf-8")
        names = source_zip.namelist()
        images = image_paragraphs(template_xml)
        markdown = expand_directives(
            markdown_path.read_text(encoding="utf-8"), repo, markdown_path.parent
        )
        used = set(re.findall(r"^%image (rId\d+)", markdown, re.M))
        body = convert(markdown, images)

        head = template_xml[: template_xml.find("<w:body>") + len("<w:body>")]
        section = re.search(r"<w:sectPr.*?</w:sectPr>", template_xml, re.S).group(0)
        document = head + body + section + "</w:body></w:document>"

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
                elif name == "docProps/core.xml":
                    core = source_zip.read(name).decode("utf-8")
                    title = output.stem
                    core = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{escape(title)}</dc:title>", core)
                    target_zip.writestr(name, core)
                else:
                    target_zip.writestr(name, source_zip.read(name))


def insert_section(document_xml_path: Path, markdown_path: Path, after_text: str, before_text: str) -> None:
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
        build_document(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    elif command == "insert":
        insert_section(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        raise SystemExit("perintah: build | insert")
