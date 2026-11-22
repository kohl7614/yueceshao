#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""软著源程序排版导出：从 git 工作树生成可提交的 PDF / DOCX。

用途
----
为「粤策哨」计算机软件著作权登记准备源程序材料。按中国版权保护中心口径，
源程序提交「前 30 页 + 后 30 页」（总页数不足 60 页则全量提交），每页 50 行，
页眉标注软件全称与版本号（须与申请表逐字一致），页码置于右上角，A4 版式。
本脚本只读取**公开代码仓**内的源码，不读取任何数据仓 / 评测 / 标注数据。

读取范围
--------
仓库根下的 app/、rag/、rules/、gen/、eval/ 五个模块中的 .py 文件，外加
config.example.json（配置样例，不含任何密钥）。文件清单来自 ``git ls-files``，
按路径排序，保证可复现；config.json / .env / 索引 / 模型权重一律不读。

用法
----
    # 在仓库根目录执行（默认生成 PDF + DOCX 到 build/cpcc/）
    python scripts/cpcc_export.py

    # 指定软件全称（与申请表一致）与版本
    python scripts/cpcc_export.py --name "粤策哨县域涉农资金申报前自查辅助系统" --version V1.0

    # 只生成 PDF，指定输出目录
    python scripts/cpcc_export.py --format pdf --out-dir /path/to/out

参数：
    --repo REPO        仓库根目录，默认取脚本所在仓库根
    --out-dir DIR      输出目录，默认 <repo>/build/cpcc
    --format {pdf,docx,both}   输出格式，默认 both
    --name NAME        软件全称，默认取模块常量 SOFTWARE_FULL_NAME（占位，待组长定）
    --version VER      版本号，默认 V1.0
    --lines-per-page N 每页行数，默认 50（软著要求）
    --head-tail-pages N        前/后各取页数，默认 30（合计 60 页）

依赖
----
标准库 + 可选依赖：
    - PDF：reportlab（pip install reportlab）
    - DOCX：python-docx（pip install python-docx）
两种依赖彼此独立；缺失某一种时该格式跳过并提示，另一种仍可生成。
中文字体按候选顺序自动探测（SimHei / SimSun / PingFang / Noto Sans CJK）。

输出
----
    <out-dir>/源程序-<软件全称>-<版本>.pdf
    <out-dir>/源程序-<软件全称>-<版本>.docx
    <out-dir>/source_manifest.txt    本次纳入的文件清单与页数统计（供核对/贴 PR）

注意
----
软件全称与版本号的默认值均为占位常量，正式提交前必须由组长按申请表定稿值
通过 --name / --version 传入；脚本本身不填写任何待定项。
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

# ===== 软件身份：软著申请表字段，顶部常量，默认占位（待组长定） =====
SOFTWARE_FULL_NAME = "【待组长定：软件全称】"
SOFTWARE_VERSION = "V1.0"

# ===== 源程序排版口径（软著要求） =====
LINES_PER_PAGE = 50
HEAD_TAIL_PAGES = 30
MODULES = ("app", "rag", "rules", "gen", "eval")
EXTRA_FILES = ("config.example.json",)

# 中文字体候选（按平台顺序探测，取第一个存在的）
FONT_CANDIDATES = (
    ("SimHei", Path(r"C:\Windows\Fonts\simhei.ttf")),
    ("SimSun", Path(r"C:\Windows\Fonts\simsun.ttc")),
    ("PingFang", Path("/System/Library/Fonts/PingFang.ttc")),
    ("Noto Sans CJK SC", Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")),
    ("WenQuanYi", Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")),
)


def repo_root() -> Path:
    """脚本所在仓库的根目录（scripts/ 的上一级）。"""
    return Path(__file__).resolve().parent.parent


def list_source_files(repo: Path) -> list[str]:
    """用 git ls-files 列出纳入的源码文件，按路径排序（可复现）。"""
    patterns = [f"{m}/*.py" for m in MODULES] + list(EXTRA_FILES)
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--", *patterns],
            check=True, capture_output=True, text=True, encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"无法从 git 工作树读取文件清单：{exc}") from exc
    files = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not files:
        raise RuntimeError("git ls-files 未返回任何源码文件，请确认在正确的仓库根运行")
    return sorted(files)


def collect_rows(repo: Path, files: list[str]) -> list[tuple[int, str]]:
    """拼装源码行流：每项为 (全局行号, 展示文本)，文件间插入一行路径标记。"""
    rows: list[tuple[int, str]] = []
    for rel in files:
        path = repo / rel
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append((0, f"# ===== 文件: {rel} ====="))
        for lineno, line in enumerate(text.splitlines(), start=1):
            rows.append((lineno, line.rstrip()))
    return rows


def safe_filename(name: str) -> str:
    """把软件全称清洗成可作文件名的字符串。"""
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", name).strip()
    return cleaned or "yueceshao"


def pick_font() -> tuple[str, Path | None]:
    """探测可用中文字体，返回 (展示名, 路径)；均无则返回 (Courier, None)。"""
    for label, path in FONT_CANDIDATES:
        if path.exists():
            return label, path
    return "Courier", None


def display_line(lineno: int, text: str) -> str:
    """单行展示文本：行号左对齐 + 源码内容。"""
    prefix = f"{lineno:5d}  " if lineno else "      "
    return prefix + text


def export_pdf(rows: list[tuple[int, str]], pages: list[int], out: Path,
               lines_per_page: int, name: str, version: str) -> None:
    """用 reportlab 生成 A4 PDF：每页 50 行、页眉、右上角页码。"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - 依赖缺失时提示
        print(f"[SKIP] PDF 未生成：reportlab 未安装（{exc}）", file=sys.stderr)
        return

    font_label, font_path = pick_font()
    if font_path is not None:
        pdfmetrics.registerFont(TTFont("CPCC-CJK", str(font_path)))
        font = "CPCC-CJK"
    else:
        font = "Courier"

    page_w, page_h = A4
    left = 18 * mm
    right_margin = 18 * mm
    top = 22 * mm
    line_h = (page_h - top - 16 * mm) / lines_per_page
    font_size = min(9.0, line_h * 0.78)
    usable_w = page_w - left - right_margin

    c = canvas.Canvas(str(out), pagesize=A4,
                      title=f"源程序 {name} {version}", author="粤策哨项目组")
    for out_index, page_no in enumerate(pages, start=1):
        start = (page_no - 1) * lines_per_page
        chunk = rows[start:start + lines_per_page]

        # 页眉：软件全称 + 版本号（与申请表逐字一致）
        c.setFont(font, 9)
        c.drawString(left, page_h - 12 * mm, f"{name}  {version}")
        # 右上角页码（连续输出序号）
        c.drawRightString(page_w - right_margin, page_h - 12 * mm, str(out_index))
        c.setStrokeColorRGB(0.75, 0.75, 0.75)
        c.line(left, page_h - 14 * mm, page_w - right_margin, page_h - 14 * mm)

        c.setFont(font, font_size)
        y = page_h - top
        for lineno, text in chunk:
            line = display_line(lineno, text)
            # 自适应字号：超宽行缩小字号，最小 6pt；再长才截断（保留源码可读）
            size = font_size
            while size > 6 and pdfmetrics.stringWidth(line, font, size) > usable_w:
                size -= 0.5
            while pdfmetrics.stringWidth(line, font, size) > usable_w and len(line) > 8:
                line = line[:-1]
            c.setFont(font, size)
            c.drawString(left, y, line)
            y -= line_h
        c.showPage()
    c.save()
    print(f"[OK] PDF 已生成：{out}（{len(pages)} 页）")


def add_page_number(paragraph) -> None:
    """在段落中插入 Word PAGE 域（右上角页码）。"""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    run = paragraph.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.addnext(fld)


def export_docx(rows: list[tuple[int, str]], pages: list[int], out: Path,
                lines_per_page: int, name: str, version: str) -> None:
    """用 python-docx 生成 A4 DOCX：每页 50 行、页眉、右上角页码。"""
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Mm, Pt
    except ImportError as exc:  # pragma: no cover - 依赖缺失时提示
        print(f"[SKIP] DOCX 未生成：python-docx 未安装（{exc}）", file=sys.stderr)
        return

    font_label, _ = pick_font()
    doc = Document()
    for section in doc.sections:
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.left_margin = Mm(18)
        section.right_margin = Mm(18)
        section.top_margin = Mm(22)
        section.bottom_margin = Mm(16)

        header = section.header.paragraphs[0]
        header.text = f"{name}  {version}"
        header.alignment = WD_ALIGN_PARAGRAPH.LEFT

        footer_p = section.footer.paragraphs[0]
        footer_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        add_page_number(footer_p)

    normal = doc.styles["Normal"]
    normal.font.name = font_label
    normal.font.size = Pt(8.5)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.0

    for idx, page_no in enumerate(pages):
        if idx > 0:
            doc.add_page_break()
        start = (page_no - 1) * lines_per_page
        for lineno, text in rows[start:start + lines_per_page]:
            p = doc.add_paragraph(display_line(lineno, text))
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
    doc.save(str(out))
    print(f"[OK] DOCX 已生成：{out}（{len(pages)} 页）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="软著源程序排版导出（前30+后30页，每页50行）")
    ap.add_argument("--repo", type=Path, default=repo_root(), help="仓库根目录")
    ap.add_argument("--out-dir", type=Path, default=None, help="输出目录，默认 <repo>/build/cpcc")
    ap.add_argument("--format", choices=("pdf", "docx", "both"), default="both", help="输出格式")
    ap.add_argument("--name", default=SOFTWARE_FULL_NAME, help="软件全称（须与申请表一致）")
    ap.add_argument("--version", default=SOFTWARE_VERSION, help="版本号")
    ap.add_argument("--lines-per-page", type=int, default=LINES_PER_PAGE, help="每页行数")
    ap.add_argument("--head-tail-pages", type=int, default=HEAD_TAIL_PAGES,
                    help="前/后各取页数（合计 60 页口径）")
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    out_dir = (args.out_dir or (repo / "build" / "cpcc")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if "待组长定" in args.name:
        print("[提示] 软件全称为占位值；正式提交前请用 --name 传入定稿全称", file=sys.stderr)

    files = list_source_files(repo)
    rows = collect_rows(repo, files)
    total_pages = math.ceil(len(rows) / args.lines_per_page)
    head_tail = args.head_tail_pages
    if total_pages <= head_tail * 2:
        pages = list(range(1, total_pages + 1))
    else:
        pages = list(range(1, head_tail + 1)) + \
            list(range(total_pages - head_tail + 1, total_pages + 1))
    stem = f"源程序-{safe_filename(args.name)}-{args.version}"
    manifest = out_dir / "source_manifest.txt"
    manifest.write_text(
        "\n".join([
            f"软件全称：{args.name}",
            f"版本号：{args.version}",
            f"纳入文件数：{len(files)}",
            f"源码总行数（含文件标记行）：{len(rows)}",
            f"总页数（每页 {args.lines_per_page} 行）：{total_pages}",
            f"实际导出页数（前{args.head_tail_pages}+后{args.head_tail_pages}）：{len(pages)}",
            "文件清单：",
            *[f"  {f}" for f in files],
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] 清单已生成：{manifest}")
    print(f"源码行数={len(rows)} 总页数={total_pages} 导出页数={len(pages)}")

    if args.format in ("pdf", "both"):
        export_pdf(rows, pages, out_dir / f"{stem}.pdf",
                   args.lines_per_page, args.name, args.version)
    if args.format in ("docx", "both"):
        export_docx(rows, pages, out_dir / f"{stem}.docx",
                    args.lines_per_page, args.name, args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
