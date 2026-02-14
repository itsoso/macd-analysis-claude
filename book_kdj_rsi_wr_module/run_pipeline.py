from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .ocr_extract import ocr_pdf, write_jsonl as write_ocr_jsonl, write_summary
    from .page_interpret import (
        interpret_pages,
        read_jsonl as read_ocr_jsonl,
        write_jsonl as write_interpret_jsonl,
        write_markdown,
    )
except ImportError:
    from ocr_extract import ocr_pdf, write_jsonl as write_ocr_jsonl, write_summary
    from page_interpret import (
        interpret_pages,
        read_jsonl as read_ocr_jsonl,
        write_jsonl as write_interpret_jsonl,
        write_markdown,
    )


DEFAULT_PDF = (
    "/Users/liqiuhua/Downloads/炒股指标三剑客 KDJ、RSI、WR入门与技巧 三大经典指标灵活运用轻松判断个股顶底 "
    "(永良，韦铭锋著, 永良, author) 9787542952998(已优化).pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="《炒股指标三剑客》独立模块：逐页 OCR + 解读")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="PDF 路径")
    parser.add_argument("--out-dir", default="book_kdj_rsi_wr_module/outputs", help="输出目录")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--skip-ocr", action="store_true", help="跳过 OCR，直接读取已有 ocr_pages.jsonl")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--lang", default="chi_sim+eng")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--no-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr_jsonl = out_dir / "ocr_pages.jsonl"
    ocr_summary = out_dir / "ocr_summary.json"
    page_jsonl = out_dir / "page_interpretation.jsonl"
    page_md = out_dir / "page_interpretation.md"

    if not args.skip_ocr:
        records = ocr_pdf(
            pdf_path=args.pdf,
            start_page=args.start_page,
            end_page=args.end_page,
            scale=args.scale,
            lang=args.lang,
            psm=args.psm,
            preprocess=not args.no_preprocess,
        )
        write_ocr_jsonl(records, ocr_jsonl)
        write_summary(records, ocr_summary)
        print(f"[pipeline] OCR finished: {len(records)} pages -> {ocr_jsonl}")
    else:
        if not ocr_jsonl.exists():
            raise FileNotFoundError(f"missing ocr file: {ocr_jsonl}")
        print(f"[pipeline] skip OCR, using existing {ocr_jsonl}")

    raw = read_ocr_jsonl(ocr_jsonl)
    interpreted = interpret_pages(raw)
    write_interpret_jsonl(interpreted, page_jsonl)
    write_markdown(interpreted, page_md)
    print(f"[pipeline] interpret finished: {len(interpreted)} pages")
    print(f"[pipeline] markdown: {page_md}")
    print(f"[pipeline] jsonl: {page_jsonl}")


if __name__ == "__main__":
    main()
