import unittest

from app.chunking import ChunkingOptions, prepare_parent_child_markdown


class ChunkingTest(unittest.TestCase):
    def test_groups_parent_by_heading_across_page_breaks(self):
        markdown = """
--- Page 1 ---

# Admission Guide

## Eligibility

Applicants must meet the common requirements for all tracks.

--- Page 2 ---

Additional notes continue on the next page.

## Documents

Submit the required forms by the deadline.
"""

        result = prepare_parent_child_markdown(
            markdown,
            ChunkingOptions(child_target_chars=220, child_overlap_chars=60),
        )

        parents = [parent for parent in result.split("<<<PARENT_BREAK>>>") if parent.strip()]

        self.assertEqual(len(parents), 2)
        self.assertIn("Section: Admission Guide > Eligibility", parents[0])
        self.assertIn("--- Page 2 ---", parents[0])
        self.assertNotIn("## Documents", parents[0])
        self.assertIn("Section: Admission Guide > Documents", parents[1])

    def test_adds_child_overlap_context(self):
        markdown = """
--- Page 1 ---

# Policy

## Access

First paragraph explains the account owner and approved devices.

Second paragraph explains the review cycle and expiry.

Third paragraph explains exceptions and approval records.
"""

        result = prepare_parent_child_markdown(
            markdown,
            ChunkingOptions(child_target_chars=120, child_overlap_chars=80),
        )

        self.assertGreater(result.count("<<<CHILD_BREAK>>>"), 1)
        self.assertIn("Previous context:", result)
        self.assertNotIn("Next context:", result)
        self.assertIn("Second paragraph explains", result)

    def test_keeps_markdown_table_in_one_child(self):
        markdown = """
--- Page 3 ---

# Scholarship

## Points

Intro text.

| Item | Points |
| --- | --- |
| Attendance | 10 |
| Certificate | 20 |

Closing text.
"""

        result = prepare_parent_child_markdown(
            markdown,
            ChunkingOptions(child_target_chars=80, child_overlap_chars=20),
        )

        table_start = result.index("| Item | Points |")
        table_end = result.index("| Certificate | 20 |") + len("| Certificate | 20 |")
        table_region = result[table_start:table_end]

        self.assertNotIn("<<<CHILD_BREAK>>>", table_region)


if __name__ == "__main__":
    unittest.main()
