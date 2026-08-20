"""Prepare a UTF-8 WikiText-2 raw corpus for the independent v14 run.

This is an explicit, one-time data-preparation command. Training and tests
never download data. The input may be the official zip file, a directory
containing wiki.*.raw files, or one already-extracted UTF-8 text file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"


def _download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "mini-transformer-charlm data prep"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _read_source(source: Path) -> tuple[str, list[str]]:
    if source.is_dir():
        candidates = []
        for filename in ("wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"):
            matches = sorted(source.rglob(filename))
            if not matches:
                raise FileNotFoundError(f"source directory is missing {filename}")
            candidates.append(matches[0])
        return "\n".join(path.read_text(encoding="utf-8-sig") for path in candidates), [
            str(path) for path in candidates
        ]

    if source.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            with zipfile.ZipFile(source) as archive:
                archive.extractall(temporary_path)
            return _read_source(temporary_path)

    return source.read_text(encoding="utf-8-sig"), [str(source)]


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local zip, directory, or UTF-8 text file")
    parser.add_argument("--url", default=DEFAULT_URL, help="official WikiText-2 v1 zip URL")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "expanded" / "wikitext-2-raw-v1.txt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data" / "expanded" / "wikitext-2-raw-v1.json",
    )
    args = parser.parse_args()

    temporary_download: Path | None = None
    try:
        source = args.source
        if source is None:
            descriptor, temporary_name = tempfile.mkstemp(suffix=".zip")
            os.close(descriptor)
            temporary_download = Path(temporary_name)
            print(f"downloading {args.url}")
            _download(args.url, temporary_download)
            source = temporary_download
        if not source.is_file() and not source.is_dir():
            raise FileNotFoundError(f"source does not exist: {source}")

        raw_text, source_files = _read_source(source)
        text = _normalize(raw_text)
        if len(text) < 1_000_000:
            raise ValueError(
                f"expanded source is unexpectedly small ({len(text):,} characters); "
                "refusing to create a fake expansion"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
        report = {
            "dataset": "WikiText-2 raw v1",
            "source": source_files,
            "output": str(args.output.resolve()),
            "utf8_bytes": args.output.stat().st_size,
            "characters": len(text),
            "lines": text.count("\n"),
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if temporary_download is not None:
            temporary_download.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
