"""Generate a vocabulary-learning research brief from the local text files.

Run with: python research_lab.py [--json]
"""

import argparse
import json
import re
from pathlib import Path


def normalize_entry(entry):
    """Normalize case, whitespace, and punctuation for exclusion matching."""
    return " ".join(re.sub(r"[^\w]+", " ", entry.casefold()).split())


def read_entries(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in: {directory}")

    entries_by_file = {}
    for file_path in files:
        with file_path.open(encoding="utf-8") as file:
            entries_by_file[file_path.name] = [
                line.strip() for line in file if line.strip()
            ]
    return entries_by_file


def build_brief(docs_dir, exclusions_dir):
    """Build a research brief from .txt vocabulary lists and exact exclusions."""
    source_entries = read_entries(docs_dir)
    exclusion_files = sorted(exclusions_dir.glob("*.txt"))
    if not exclusions_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {exclusions_dir}")

    exclusions_by_key = {}
    for file_path in exclusion_files:
        with file_path.open(encoding="utf-8") as file:
            for line in file:
                entry = line.strip()
                if entry:
                    exclusions_by_key.setdefault(normalize_entry(entry), entry)

    included_by_file = {}
    excluded_entries = []
    included_entries = []
    matched_exclusions = set()

    for filename, entries in source_entries.items():
        included_by_file[filename] = []
        for entry in entries:
            key = normalize_entry(entry)
            if key in exclusions_by_key:
                excluded_entries.append(
                    {"entry": entry, "excluded_by": exclusions_by_key[key], "file": filename}
                )
                matched_exclusions.add(key)
            else:
                included_by_file[filename].append(entry)
                included_entries.append({"entry": entry, "file": filename})

    source_count = sum(map(len, source_entries.values()))
    distinct_exclusions = sorted(exclusions_by_key.values(), key=str.casefold)
    unmatched_exclusions = [
        entry for entry in distinct_exclusions
        if normalize_entry(entry) not in matched_exclusions
    ]
    document_groups = [
        {
            "file": filename,
            "included_count": len(entries),
            "examples": entries[:5],
        }
        for filename, entries in included_by_file.items()
    ]

    return {
        "title": "Vocabulary practice research brief",
        "status": "exploratory proposal generated from local vocabulary lists",
        "question": (
            "Does mixing eligible vocabulary items from different source files "
            "improve delayed recall compared with practicing each file separately?"
        ),
        "sources": {
            "directory": docs_dir.name,
            "files": list(source_entries),
            "entries_read": source_count,
        },
        "exclusions": {
            "directory": exclusions_dir.name,
            "terms_read": len(exclusions_by_key),
            "entries_removed": len(excluded_entries),
            "matched_terms": [
                exclusions_by_key[key]
                for key in sorted(matched_exclusions, key=str.casefold)
            ],
            "unmatched_terms": unmatched_exclusions,
            "removed_entries": excluded_entries,
        },
        "corpus": {
            "eligible_entries": len(included_entries),
            "document_groups": document_groups,
            "eligible_items": included_entries,
        },
        "hypothesis": (
            "For the eligible items in these files, mixed practice will produce "
            "higher delayed recall than file-by-file practice when study time "
            "and item exposure are matched."
        ),
        "experiment": [
            "Use the remaining entries in docs/*.txt as the study vocabulary; do not include exact matches from docs_not/*.txt.",
            "Randomly assign learners to mixed-item practice or source-file-by-file practice.",
            "Match study time, prompts, and exposure per item, then administer an immediate and a delayed recall test.",
            "Repeat with pre-registered random seeds or participant groups and report uncertainty across runs.",
        ],
        "metrics": [
            "immediate and delayed recall accuracy",
            "recall accuracy by source file",
            "study time per correctly recalled item",
            "excluded and eligible item counts",
        ],
        "falsification": (
            "Do not support the hypothesis if mixed practice does not improve "
            "delayed recall under matched exposure, or if the result is not "
            "reproducible across groups."
        ),
        "limitations": [
            "The files have generic names and do not provide validated semantic categories.",
            "Exact normalized line matching is used for exclusions; partial phrase matches are not removed.",
            "The brief proposes an experiment; it does not claim a measured learning benefit.",
        ],
    }


def print_text(brief):
    print(brief["title"])
    print("=" * len(brief["title"]))
    print("Status:", brief["status"])
    print("\nQuestion\n--------\n" + brief["question"])
    print("\nCorpus\n------")
    for group in brief["corpus"]["document_groups"]:
        examples = ", ".join(group["examples"])
        print(f"- {group['file']}: {group['included_count']} eligible entries")
        if examples:
            print(f"  Examples: {examples}")
    print(
        f"\nRead {brief['sources']['entries_read']} entries; "
        f"kept {brief['corpus']['eligible_entries']} and removed "
        f"{brief['exclusions']['entries_removed']} using "
        f"{brief['exclusions']['terms_read']} exclusion terms."
    )
    print("\nHypothesis\n----------\n" + brief["hypothesis"])
    for key, heading in (("experiment", "Experiment"), ("metrics", "Metrics")):
        print("\n" + heading + "\n" + "-" * len(heading))
        for item in brief[key]:
            print("- " + item)
    print("\nFalsification\n-------------\n" + brief["falsification"])


def main():
    parser = argparse.ArgumentParser(
        description="Generate a vocabulary research brief from docs and docs_not."
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    root = Path(__file__).resolve().parent
    parser.add_argument("--docs-dir", type=Path, default=root / "docs")
    parser.add_argument("--exclusions-dir", type=Path, default=root / "docs_not")
    args = parser.parse_args()

    try:
        brief = build_brief(args.docs_dir, args.exclusions_dir)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(brief, ensure_ascii=False, indent=2))
    else:
        print_text(brief)


if __name__ == "__main__":
    main()
