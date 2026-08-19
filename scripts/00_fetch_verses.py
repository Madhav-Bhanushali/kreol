import argparse
import json
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

# 27 NT books in canonical order -> USFM abbreviation used by bible.com
BOOK_ABBR = [
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH",
    "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAS",
    "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV",
]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


class VerseParser(HTMLParser):
    """Walk a bible.com chapter page and collect {verse_num: text}."""

    def __init__(self):
        super().__init__()
        self.stack = []  # span classes while inside a verse ('verse','label','content','other')
        self.cur_num = None
        self.cur_text = []
        self.verses = {}

    def handle_starttag(self, tag, attrs):
        if tag != "span":
            return
        d = dict(attrs)
        cls = d.get("class", "")
        if "data-usfm" in d:
            self.stack = ["verse"]
            self.cur_num = None
            self.cur_text = []
        elif self.stack and self.stack[0] == "verse":
            self.stack.append("label" if "label" in cls else "content" if "content" in cls else "other")

    def handle_endtag(self, tag):
        if tag != "span" or not self.stack:
            return
        if self.stack[0] == "verse" and len(self.stack) == 1:
            if self.cur_num is not None:
                text = " ".join("".join(self.cur_text).split())
                if text:
                    self.verses[str(self.cur_num)] = text
            self.stack = []
        else:
            self.stack.pop()

    def handle_data(self, data):
        if self.stack and self.stack[0] == "verse":
            if self.stack[-1] == "label":
                m = re.match(r"\s*(\d+)", data)
                if m:
                    self.cur_num = int(m.group(1))
            elif self.stack[-1] == "content":
                self.cur_text.append(data)


def parse_chapter(html: str) -> dict[str, str]:
    """Extract {verse_number: text} from a bible.com chapter page."""
    p = VerseParser()
    p.feed(html)
    return p.verses


def fetch_chapter(version: int, abbr: str, chapter: int, timeout: int = 30) -> dict[str, str]:
    url = f"https://www.bible.com/bible/{version}/{abbr}.{chapter}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse_chapter(r.read().decode("utf-8", errors="ignore"))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Fetch NTKM2009 (bible.com version 344) chapter text verse-by-verse and write "
            "data/verses.json in the format scripts/03_align.py expects. "
            "NOTE: bible.com is © Bible Society of Mauritius 2009 — confirm usage terms "
            "(DBL entry 616296e8e170ebc1) before bulk use."
        )
    )
    ap.add_argument("--audio-dir", type=Path, default=None, help="folder with the chapter mp3s (stems derived from filenames); use this on the GPU box")
    ap.add_argument("--stems", default=None, help="comma-separated list of stems to fetch instead of scanning --audio-dir")
    ap.add_argument("--version", type=int, default=344, help="bible.com version id (344 = NTKM2009)")
    ap.add_argument("--out", type=Path, default=Path("data/verses.json"))
    ap.add_argument("--rate", type=float, default=0.5, help="seconds between requests (be polite)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only fetch first N chapters")
    args = ap.parse_args()

    chapters = {}  # stem -> (book_no, chapter_no)
    if args.audio_dir:
        files = sorted(args.audio_dir.glob("*.mp3")) or sorted(args.audio_dir.glob("*.wav"))
        if not files:
            raise SystemExit(f"no audio files in {args.audio_dir}")
        for f in files:
            m = re.match(r"B(\d+)_+(\d+)_", f.stem)
            if not m:
                print(f"skip (unparseable stem): {f.name}")
                continue
            chapters[f.stem] = (int(m.group(1)), int(m.group(2)))
    elif args.stems:
        for stem in args.stems.split(","):
            stem = stem.strip()
            m = re.match(r"B(\d+)_+(\d+)_", stem)
            if not m:
                raise SystemExit(f"unparseable stem: {stem!r}")
            chapters[stem] = (int(m.group(1)), int(m.group(2)))
    else:
        raise SystemExit("need either --audio-dir or --stems")

    if args.limit:
        chapters = dict(list(chapters.items())[: args.limit])

    verses, failed = {}, []
    for i, (stem, (book, chap)) in enumerate(chapters.items(), 1):
        if not (1 <= book <= 27):
            failed.append(stem)
            print(f"skip (bad book {book}): {stem}")
            continue
        abbr = BOOK_ABBR[book - 1]
        for attempt in range(3):
            try:
                v = fetch_chapter(args.version, abbr, chap)
                if v:
                    verses[stem] = v
                    break
            except Exception as e:
                if attempt == 2:
                    failed.append(stem)
                    print(f"FAIL {stem} ({abbr} {chap}): {e}")
                else:
                    time.sleep(2 * (attempt + 1))
        else:
            continue
        print(f"[{i}/{len(chapters)}] {stem}: {len(verses[stem])} verses")
        time.sleep(args.rate)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in verses.values())
    print(f"\nwrote {len(verses)} chapters, {total} verses -> {args.out}")
    if failed:
        print(f"FAILED chapters ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()