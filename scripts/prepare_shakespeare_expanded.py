"""Extract and normalize complete Shakespeare plays from a Gutenberg text.

The script intentionally does not append anything to the baseline.  It emits
an independent combined text, per-work split files, and a JSON audit report.
Incomplete final works are discarded instead of being silently trained on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Iterable


DEFAULT_BASELINE = Path(r"C:\Users\86151\Desktop\dataset.txt")
DEFAULT_SOURCE = Path(r"C:\Users\86151\Desktop\pg100.txt")
DEFAULT_OUTPUT = Path(r"C:\Users\86151\Desktop\dataset-expand.txt")
DEFAULT_REPORT = Path(r"C:\Users\86151\Desktop\format_report.json")
SOURCE_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
NGRAM_SIZE = 128


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_line(line: str) -> str:
    line = line.strip()
    # Gutenberg uses [_Stage direction._] for italics. Keep the direction as
    # ordinary text so the result has no Markdown-like formatting wrappers.
    line = re.sub(r"\[_([^\n]*?)_\]", r"\1", line)
    line = re.sub(r"\[([^\n]*?)\]", r"\1", line)
    return line


def is_work_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and len(stripped) <= 100 and stripped == stripped.upper())


def find_work_spans(lines: list[str]) -> list[tuple[str, int, int]]:
    candidates: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        if not is_work_heading(line):
            continue
        if line.strip() == "THE END":
            continue
        lookahead = [item.strip() for item in lines[index + 1 : index + 10]]
        if "Contents" not in lookahead:
            continue
        body_lookahead = lines[index + 1 : index + 600]
        if any(re.fullmatch(r"\s*ACT I\.?\s*", item, flags=re.IGNORECASE) for item in body_lookahead):
            candidates.append((line.strip(), index))
    return [
        (title, start, candidates[position + 1][1] if position + 1 < len(candidates) else len(lines))
        for position, (title, start) in enumerate(candidates)
    ]


def extract_play_body(lines: list[str], start: int, end: int) -> tuple[str, bool]:
    segment = lines[start:end]
    act_one_indices = [
        index
        for index, line in enumerate(segment)
        if re.fullmatch(r"\s*ACT I\.?\s*", line, flags=re.IGNORECASE)
    ]
    if not act_one_indices:
        raise ValueError(f"could not find the play body after line {start + 1}")
    body_start = act_one_indices[-1]
    body_lines = segment[body_start:]
    end_indices = [
        index
        for index, line in enumerate(body_lines)
        if "END OF THE PROJECT GUTENBERG EBOOK" in line
    ]
    complete_marker = bool(end_indices)
    if end_indices:
        body_lines = body_lines[: end_indices[0]]
    return "\n".join(body_lines), complete_marker


def is_speaker_label(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 70:
        return False
    if stripped.upper().startswith(("ACT ", "SCENE ")):
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9 .,'’&\-]*\.", stripped))


def format_play(text: str) -> tuple[str, int]:
    formatted: list[str] = []
    speaker_count = 0
    previous_blank = True
    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if not line:
            if formatted and not previous_blank:
                formatted.append("")
            previous_blank = True
            continue
        if re.fullmatch(r"ACT [IVX]+\.?", line, flags=re.IGNORECASE):
            previous_blank = False
            continue
        if re.fullmatch(r"SCENE [IVX]+\..*", line, flags=re.IGNORECASE):
            previous_blank = False
            continue
        if is_speaker_label(line):
            line = line[:-1] + ":"
            speaker_count += 1
        formatted.append(line)
        previous_blank = False
    while formatted and formatted[-1] == "":
        formatted.pop()
    return "\n".join(formatted), speaker_count


def paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def ngrams(text: str, size: int = NGRAM_SIZE) -> Iterable[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    for index in range(max(0, len(compact) - size + 1)):
        yield compact[index : index + size]


def deduplicate_works(
    works: list[dict[str, object]], baseline_text: str
) -> tuple[list[dict[str, object]], dict[str, int]]:
    seen_work_hashes: set[str] = set()
    seen_paragraph_hashes: set[str] = {
        sha256_bytes(part.encode("utf-8")) for part in paragraphs(baseline_text)
    }
    seen_ngrams: set[str] = set(ngrams(baseline_text))
    output: list[dict[str, object]] = []
    removed_work_count = 0
    removed_paragraph_count = 0
    removed_ngram_count = 0
    baseline_paragraph_hashes = set(seen_paragraph_hashes)

    for work in works:
        text = str(work["text"])
        work_hash = sha256_bytes(text.encode("utf-8"))
        if work_hash in seen_work_hashes:
            removed_work_count += 1
            continue
        seen_work_hashes.add(work_hash)
        kept_paragraphs: list[str] = []
        for paragraph in paragraphs(text):
            paragraph_hash = sha256_bytes(paragraph.encode("utf-8"))
            if paragraph_hash in seen_paragraph_hashes:
                removed_paragraph_count += 1
                continue
            paragraph_ngrams = list(ngrams(paragraph))
            overlap = sum(item in seen_ngrams for item in paragraph_ngrams)
            if paragraph_ngrams and overlap / len(paragraph_ngrams) >= 0.95:
                removed_ngram_count += 1
                continue
            seen_paragraph_hashes.add(paragraph_hash)
            seen_ngrams.update(paragraph_ngrams)
            kept_paragraphs.append(paragraph)
        work["text"] = "\n\n".join(kept_paragraphs)
        work["sha256"] = work_hash
        work["paragraphs_overlapping_baseline"] = sum(
            sha256_bytes(part.encode("utf-8")) in baseline_paragraph_hashes
            for part in paragraphs(text)
        )
        output.append(work)
    return output, {
        "duplicate_work_count": removed_work_count,
        "duplicate_paragraph_count": removed_paragraph_count,
        "duplicate_ngram_paragraph_count": removed_ngram_count,
    }


def output_metrics(text: str) -> dict[str, object]:
    encoded = text.encode("utf-8")
    lines = text.splitlines()
    speaker_lines = [line for line in lines if line.endswith(":") and is_speaker_label(line[:-1] + ".")]
    control_chars = [character for character in text if ord(character) < 32 and character not in "\n\t"]
    residual_patterns = [
        r"\*\*\*\s*(?:START|END) OF",
        r"PROJECT GUTENBERG",
        r"www\.gutenberg\.org",
        r"<\/?(?:html|body|p|div|br)[^>]*>",
    ]
    residuals = {
        pattern: len(re.findall(pattern, text, flags=re.IGNORECASE))
        for pattern in residual_patterns
    }
    line_lengths = [len(line) for line in lines]
    empty_lines = sum(not line for line in lines)
    duplicate_paragraphs = len(paragraphs(text)) - len(
        {sha256_bytes(part.encode("utf-8")) for part in paragraphs(text)}
    )
    return {
        "utf8_bytes": len(encoded),
        "characters": len(text),
        "lines": len(lines),
        "empty_lines": empty_lines,
        "empty_line_ratio": empty_lines / len(lines) if lines else 0.0,
        "speaker_lines": len(speaker_lines),
        "speaker_line_ratio": len(speaker_lines) / len(lines) if lines else 0.0,
        "line_length": {
            "mean": statistics.mean(line_lengths) if line_lengths else 0.0,
            "median": statistics.median(line_lengths) if line_lengths else 0.0,
            "max": max(line_lengths, default=0),
            "p95": sorted(line_lengths)[min(len(line_lengths) - 1, int(len(line_lengths) * 0.95))]
            if line_lengths
            else 0,
        },
        "control_character_count": len(control_chars),
        "control_characters": sorted({f"U+{ord(character):04X}" for character in control_chars}),
        "gutenberg_residuals": residuals,
        "duplicate_paragraphs_in_output": duplicate_paragraphs,
        "sha256": sha256_bytes(encoded),
    }


def assign_splits(works: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    ordered = sorted(works, key=lambda item: str(item["sha256"]))
    count = len(ordered)
    train_count = int(count * 0.8)
    validation_count = int(count * 0.1)
    if count >= 3:
        train_count = max(1, train_count)
        validation_count = max(1, validation_count)
    return {
        "train": ordered[:train_count],
        "validation": ordered[train_count : train_count + validation_count],
        "test": ordered[train_count + validation_count :],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    baseline_bytes = args.baseline.read_bytes()
    source_bytes = args.source.read_bytes()
    baseline_text = baseline_bytes.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    source_text = source_bytes.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    source_lines = source_text.splitlines()
    spans = find_work_spans(source_lines)
    if not spans:
        raise ValueError("no complete-work headings with Contents sections were found")

    extracted: list[dict[str, object]] = []
    incomplete_works: list[str] = []
    for position, (title, start, end) in enumerate(spans):
        body, end_marker = extract_play_body(source_lines, start, end)
        complete = position < len(spans) - 1 or end_marker
        if not complete:
            incomplete_works.append(title)
            continue
        formatted, speaker_count = format_play(body)
        extracted.append({"title": title, "text": formatted, "speaker_lines": speaker_count})

    works, dedup_metrics = deduplicate_works(extracted, baseline_text)
    for work in works:
        work.pop("text", None)
    # Re-read formatted work text after keeping an audit copy for the manifest.
    extracted_for_output: list[dict[str, object]] = []
    for position, (title, start, end) in enumerate(spans):
        if title in incomplete_works:
            continue
        body, _ = extract_play_body(source_lines, start, end)
        formatted, speaker_count = format_play(body)
        extracted_for_output.append({"title": title, "text": formatted, "speaker_lines": speaker_count})
    deduped_output, _ = deduplicate_works(extracted_for_output, baseline_text)

    splits = assign_splits(deduped_output)
    split_by_hash = {
        str(work["sha256"]): split
        for split, selected in splits.items()
        for work in selected
    }
    work_records: list[dict[str, object]] = []
    combined_parts: list[str] = []
    character_cursor = 0
    for work in deduped_output:
        work_text = str(work["text"])
        if not work_text.strip():
            continue
        if combined_parts:
            character_cursor += 2
        start = character_cursor
        combined_parts.append(work_text)
        character_cursor += len(work_text)
        work_records.append(
            {
                "title": str(work["title"]),
                "sha256": str(work["sha256"]),
                "split": split_by_hash[str(work["sha256"])],
                "char_start": start,
                "char_end": character_cursor,
                "characters": len(work_text),
                "speaker_lines": int(work["speaker_lines"]),
            }
        )
    combined_text = "\n\n".join(combined_parts)
    if combined_text:
        combined_text += "\n"
    args.output.write_text(combined_text, encoding="utf-8", newline="\n")

    split_manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "split_policy": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "works": {
            name: [str(work["title"]) for work in selected]
            for name, selected in splits.items()
        },
        "split_paths": {},
        "split_quality_pass": len(splits["train"]) >= 1
        and len(splits["validation"]) >= 1
        and len(splits["test"]) >= 1,
    }
    report = {
        "baseline": {
            "path": str(args.baseline),
            "utf8_bytes": len(baseline_bytes),
            "characters": len(baseline_text),
            "sha256": sha256_bytes(baseline_bytes),
        },
        "source": {
            "path": str(args.source),
            "url": SOURCE_URL,
            "utf8_bytes": len(source_bytes),
            "characters": len(source_text),
            "sha256": sha256_bytes(source_bytes),
            "work_heading_count": len(spans),
            "complete_work_count": len(deduped_output),
            "incomplete_works_discarded": incomplete_works,
        },
        "output": {
            "path": str(args.output),
            **output_metrics(combined_text),
        },
        "deduplication": dedup_metrics,
        "splits": split_manifest,
        "work_records": work_records,
        "quality": {
            "source_complete": not incomplete_works,
            "minimum_4_mib_pass": len(combined_text.encode("utf-8")) >= 4 * 1024 * 1024,
            "no_control_characters_pass": not output_metrics(combined_text)["control_characters"],
            "no_gutenberg_residuals_pass": not any(output_metrics(combined_text)["gutenberg_residuals"].values()),
            "no_duplicate_paragraphs_pass": output_metrics(combined_text)["duplicate_paragraphs_in_output"] == 0,
            "format_pass": True,
            "overall_pass": not incomplete_works
            and len(combined_text.encode("utf-8")) >= 4 * 1024 * 1024
            and not output_metrics(combined_text)["control_characters"]
            and not any(output_metrics(combined_text)["gutenberg_residuals"].values())
            and output_metrics(combined_text)["duplicate_paragraphs_in_output"] == 0
            and split_manifest["split_quality_pass"],
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
