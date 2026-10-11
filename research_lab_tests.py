import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import research_lab


class ResearchLabTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.docs_dir = self.root / "docs"
        self.exclusions_dir = self.root / "docs_not"
        self.docs_dir.mkdir()
        self.exclusions_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_brief_uses_doc_entries_and_removes_excluded_exact_matches(self):
        (self.docs_dir / "vocabulary.txt").write_text(
            "Headache\nV-neck\nAirport terminal\n", encoding="utf-8"
        )
        (self.docs_dir / "notes.md").write_text(
            "This markdown file is descriptive, not a vocabulary list.",
            encoding="utf-8",
        )
        (self.exclusions_dir / "excluded.txt").write_text(
            " headache \n", encoding="utf-8"
        )

        brief = research_lab.build_brief(self.docs_dir, self.exclusions_dir)

        self.assertEqual(brief["sources"]["files"], ["vocabulary.txt"])
        self.assertEqual(brief["sources"]["entries_read"], 3)
        self.assertEqual(brief["exclusions"]["entries_removed"], 1)
        self.assertEqual(
            [item["entry"] for item in brief["corpus"]["eligible_items"]],
            ["V-neck", "Airport terminal"],
        )
        self.assertEqual(brief["exclusions"]["matched_terms"], ["headache"])

    def test_partial_phrase_does_not_remove_an_entry(self):
        (self.docs_dir / "vocabulary.txt").write_text(
            "High blood pressure\n", encoding="utf-8"
        )
        (self.exclusions_dir / "excluded.txt").write_text(
            "blood pressure\n", encoding="utf-8"
        )

        brief = research_lab.build_brief(self.docs_dir, self.exclusions_dir)

        self.assertEqual(brief["exclusions"]["entries_removed"], 0)
        self.assertEqual(brief["corpus"]["eligible_entries"], 1)

    def test_requires_text_source_files(self):
        with self.assertRaisesRegex(ValueError, "No .txt files found"):
            research_lab.build_brief(self.docs_dir, self.exclusions_dir)

    def test_json_output_is_valid_and_contains_applied_exclusions(self):
        (self.docs_dir / "vocabulary.txt").write_text("flu\n", encoding="utf-8")
        (self.exclusions_dir / "excluded.txt").write_text("flu\n", encoding="utf-8")

        output = StringIO()
        with patch(
            "sys.argv",
            [
                "research_lab.py",
                "--json",
                "--docs-dir",
                str(self.docs_dir),
                "--exclusions-dir",
                str(self.exclusions_dir),
            ],
        ), redirect_stdout(output):
            research_lab.main()

        parsed = json.loads(output.getvalue())
        self.assertEqual(parsed["exclusions"]["entries_removed"], 1)
        self.assertEqual(parsed["corpus"]["eligible_items"], [])


if __name__ == "__main__":
    unittest.main()
