from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import fitz
import pytesseract
from PIL import Image, ImageOps


def _clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _preprocess(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    # 简单阈值化以提升印刷文字识别稳定性
    return gray.point(lambda p: 255 if p > 165 else 0)


@dataclass(slots=True)
class OcrRecord:
    page: int
    text: str
    char_count: int
    elapsed_sec: float


def ocr_pdf(
    pdf_path: str | Path,
    start_page: int = 1,
    end_page: int | None = None,
    scale: float = 2.0,
    lang: str = "chi_sim+eng",
    psm: int = 6,
    preprocess: bool = True,
    progress_every: int = 10,
) -> list[OcrRecord]:
    pdf_path = str(pdf_path)
    doc = fitz.open(pdf_path)
    total = doc.page_count
    if end_page is None:
        end_page = total
    start = max(1, start_page)
    end = min(end_page, total)
    if start > end:
        raise ValueError(f"invalid range: start_page={start_page}, end_page={end_page}, total={total}")

    records: list[OcrRecord] = []
    config = f"--psm {psm}"
    begin = time.time()

    for page_no in range(start, end + 1):
        t0 = time.time()
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if preprocess:
            img = _preprocess(img)

        raw = pytesseract.image_to_string(img, lang=lang, config=config)
        text = _clean_text(raw)
        elapsed = time.time() - t0
        records.append(OcrRecord(page=page_no, text=text, char_count=len(text), elapsed_sec=elapsed))

        idx = page_no - start + 1
        if idx % progress_every == 0 or page_no == end:
            avg = (time.time() - begin) / idx
            print(
                f"[ocr] {idx}/{end - start + 1} pages done, "
                f"avg={avg:.2f}s/page, current_page={page_no}, chars={len(text)}"
            )
    return records


def write_jsonl(records: Iterable[OcrRecord], out_file: str | Path) -> None:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_summary(records: list[OcrRecord], out_file: str | Path) -> None:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    chars = [r.char_count for r in records]
    stats = {
        "pages": len(records),
        "char_total": int(sum(chars)),
        "char_avg": float(sum(chars) / len(chars)) if chars else 0.0,
        "char_min": int(min(chars)) if chars else 0,
        "char_max": int(max(chars)) if chars else 0,
        "slowest_page": int(max(records, key=lambda x: x.elapsed_sec).page) if records else None,
    }
    out_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF 按页 OCR 抽取")
    parser.add_argument("--pdf", required=True, help="PDF 文件路径")
    parser.add_argument("--out-jsonl", required=True, help="输出 JSONL")
    parser.add_argument("--out-summary", required=False, help="输出统计 JSON")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--lang", default="chi_sim+eng")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--no-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = ocr_pdf(
        pdf_path=args.pdf,
        start_page=args.start_page,
        end_page=args.end_page,
        scale=args.scale,
        lang=args.lang,
        psm=args.psm,
        preprocess=not args.no_preprocess,
    )
    write_jsonl(records, args.out_jsonl)
    if args.out_summary:
        write_summary(records, args.out_summary)
    print(f"[ocr] wrote {len(records)} pages -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
