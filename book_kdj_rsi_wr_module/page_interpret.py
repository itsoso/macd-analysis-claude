from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CHAPTER_RE = re.compile(r"(第[一二三四五六七八九十百0-9]+章[^\n]{0,24})")
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;]\s*")

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "均线": ["均线", "移动平均", "MA", "趋势"],
    "KDJ": ["KDJ", "随机指标", "K线", "D线", "J线", "金叉", "死叉"],
    "RSI": ["RSI", "强弱指标", "相对强弱"],
    "WR": ["WR", "威廉", "Williams", "%R"],
    "超买超卖": ["超买", "超卖", "80", "20", "-20", "-80"],
    "买卖点": ["买入", "卖出", "买点", "卖点", "止损", "止盈"],
    "背离": ["背离", "顶背离", "底背离"],
}

CODE_MAP: dict[str, str] = {
    "KDJ": "book_kdj_rsi_wr_module/indicators.py::calc_kdj",
    "RSI": "book_kdj_rsi_wr_module/indicators.py::calc_rsi",
    "WR": "book_kdj_rsi_wr_module/indicators.py::calc_wr",
    "买卖点": "book_kdj_rsi_wr_module/indicators.py::add_triple_sword_features",
}


@dataclass(slots=True)
class PageInput:
    page: int
    text: str
    char_count: int


@dataclass(slots=True)
class PageInterpretation:
    page: int
    chapter: str
    topics: list[str]
    summary: str
    rules: list[str]
    code_mapping: list[str]
    confidence: float


def _normalize(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_chapter(text: str, prev_chapter: str) -> str:
    m = CHAPTER_RE.search(text)
    if m:
        return m.group(1).strip()
    return prev_chapter


def _detect_topics(text: str) -> list[str]:
    hits: list[str] = []
    for topic, words in TOPIC_KEYWORDS.items():
        for w in words:
            if w in text:
                hits.append(topic)
                break
    return hits


def _score_sentence(sentence: str, topics: list[str]) -> int:
    score = 0
    for t in topics:
        for w in TOPIC_KEYWORDS.get(t, []):
            if w in sentence:
                score += 2
    if 12 <= len(sentence) <= 80:
        score += 1
    return score


def _extract_summary(text: str, topics: list[str]) -> str:
    candidate = []
    for seg in SENTENCE_SPLIT_RE.split(text):
        seg = seg.strip()
        if len(seg) >= 10:
            candidate.append(seg)
    if not candidate:
        return text[:120]

    scored = sorted(candidate, key=lambda s: _score_sentence(s, topics), reverse=True)
    summary_parts = scored[:2]
    return "；".join(summary_parts)[:220]


def _extract_rules(text: str) -> list[str]:
    rules: list[str] = []
    snippets = text.splitlines()
    pattern_pairs = [
        ("金叉", "KDJ出现金叉，优先视为偏多触发条件"),
        ("死叉", "KDJ出现死叉，优先视为空头风险提示"),
        ("超买", "指标进入超买区，避免追高并观察拐点"),
        ("超卖", "指标进入超卖区，等待反转确认"),
        ("背离", "价格与指标背离时，趋势延续概率下降"),
        ("止损", "满足条件时严格执行止损"),
    ]

    seen = set()
    for line in snippets:
        line = line.strip()
        if len(line) < 4:
            continue
        for token, rule in pattern_pairs:
            if token in line and rule not in seen:
                rules.append(rule)
                seen.add(rule)
    return rules[:4]


def _build_code_map(topics: list[str]) -> list[str]:
    refs: list[str] = []
    for t in topics:
        if t in CODE_MAP:
            refs.append(CODE_MAP[t])
    return sorted(set(refs))


def _confidence(char_count: int, topics: list[str]) -> float:
    base = 0.35
    if char_count >= 400:
        base += 0.25
    if char_count >= 800:
        base += 0.20
    base += min(0.2, 0.03 * len(topics))
    return round(min(base, 0.95), 2)


def interpret_pages(pages: Iterable[PageInput]) -> list[PageInterpretation]:
    out: list[PageInterpretation] = []
    current_chapter = "未知章节"
    for p in pages:
        text = _normalize(p.text)
        current_chapter = _extract_chapter(text, current_chapter)
        topics = _detect_topics(text)
        summary = _extract_summary(text, topics)
        rules = _extract_rules(text)
        code_mapping = _build_code_map(topics)
        out.append(
            PageInterpretation(
                page=p.page,
                chapter=current_chapter,
                topics=topics,
                summary=summary,
                rules=rules,
                code_mapping=code_mapping,
                confidence=_confidence(p.char_count, topics),
            )
        )
    return out


def read_jsonl(path: str | Path) -> list[PageInput]:
    records: list[PageInput] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            records.append(
                PageInput(
                    page=int(obj["page"]),
                    text=str(obj.get("text", "")),
                    char_count=int(obj.get("char_count", 0)),
                )
            )
    return records


def write_jsonl(records: Iterable[PageInterpretation], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_markdown(records: list[PageInterpretation], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# 《炒股指标三剑客》逐页解读（自动生成）")
    lines.append("")
    lines.append(f"总页数: {len(records)}")
    lines.append("")

    chapter_counter: dict[str, int] = defaultdict(int)
    for r in records:
        chapter_counter[r.chapter] += 1

    lines.append("## 章节覆盖")
    for chap, cnt in sorted(chapter_counter.items(), key=lambda x: x[0]):
        lines.append(f"- {chap}: {cnt} 页")
    lines.append("")

    lines.append("## 逐页内容")
    for r in records:
        topics = "、".join(r.topics) if r.topics else "未识别"
        rules = "；".join(r.rules) if r.rules else "无明确规则"
        refs = "；".join(r.code_mapping) if r.code_mapping else "无直接映射"
        lines.append(f"### 第 {r.page} 页")
        lines.append(f"- 章节: {r.chapter}")
        lines.append(f"- 主题: {topics}")
        lines.append(f"- 摘要: {r.summary}")
        lines.append(f"- 规则: {rules}")
        lines.append(f"- 代码映射: {refs}")
        lines.append(f"- 置信度: {r.confidence}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按页解读 OCR 结果")
    parser.add_argument("--ocr-jsonl", required=True, help="OCR 输出 JSONL")
    parser.add_argument("--out-jsonl", required=True, help="解读输出 JSONL")
    parser.add_argument("--out-md", required=True, help="解读输出 Markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = read_jsonl(args.ocr_jsonl)
    interpreted = interpret_pages(raw)
    write_jsonl(interpreted, args.out_jsonl)
    write_markdown(interpreted, args.out_md)
    print(f"[interpret] wrote {len(interpreted)} pages")
    print(f"[interpret] jsonl: {args.out_jsonl}")
    print(f"[interpret] markdown: {args.out_md}")


if __name__ == "__main__":
    main()
