"""Check local Markdown links and heading anchors without external services."""
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^\s)]+)\)")


def anchors(path):
    result = set()
    repeated = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^#{1,6}\s", line):
            continue
        heading = re.sub(r"^#{1,6}\s+", "", line).strip().lower()
        heading = re.sub(r"[^\w\-\s]", "", heading).replace(" ", "-")
        count = repeated.get(heading, 0)
        repeated[heading] = count + 1
        result.add(f"{heading}-{count}" if count else heading)
    return result


def main():
    failures = []
    paths = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "THIRD_PARTY_NOTICES.md",
             *sorted((ROOT / "docs").rglob("*.md"))]
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for raw in LINK.findall(line):
                target = urlsplit(raw)
                if target.scheme or target.netloc:
                    continue
                destination = (path.parent / unquote(target.path)).resolve() if target.path else path
                if not destination.is_relative_to(ROOT) or not destination.exists():
                    failures.append(f"{path.relative_to(ROOT)}:{number}: invalid local target {raw}")
                elif target.fragment and destination.suffix == ".md" and unquote(target.fragment) not in anchors(destination):
                    failures.append(f"{path.relative_to(ROOT)}:{number}: missing heading {raw}")
    for failure in failures:
        print(failure)
    print({"markdown_files": len(paths), "invalid_links": len(failures)})
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
