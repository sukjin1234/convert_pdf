import unittest

from app.markdown import render_document_pages_to_markdown, render_document_to_markdown


class MarkdownRendererTest(unittest.TestCase):
    def test_renders_headings_body_tables_and_images(self):
        doc = {
            "kids": [
                {"type": "heading", "heading level": 1, "content": "Security Policy"},
                {"type": "heading", "heading level": 2, "content": "Access Scope"},
                {"type": "paragraph", "content": "Employees must use approved devices."},
                {
                    "type": "table",
                    "rows": [
                        {
                            "type": "table row",
                            "cells": [
                                {"column number": 1, "kids": [{"type": "paragraph", "content": "Role"}]},
                                {"column number": 2, "kids": [{"type": "paragraph", "content": "Permission"}]},
                            ],
                        },
                        {
                            "type": "table row",
                            "cells": [
                                {"column number": 1, "kids": [{"type": "paragraph", "content": "Admin"}]},
                                {"column number": 2, "kids": [{"type": "paragraph", "content": "Approve | revoke"}]},
                            ],
                        },
                    ],
                },
                {
                    "type": "picture",
                    "source": "images/chart.png",
                    "description": "A chart showing quarterly access request volume.",
                },
            ]
        }

        markdown = render_document_to_markdown(doc)

        self.assertIn("# Security Policy", markdown)
        self.assertIn("## Access Scope", markdown)
        self.assertIn("Employees must use approved devices.", markdown)
        self.assertIn("| Role | Permission |", markdown)
        self.assertIn("| Admin | Approve \\| revoke |", markdown)
        self.assertNotIn("](images/chart.png)", markdown)
        self.assertIn("**Image summary:** A chart showing quarterly access request volume.", markdown)

    def test_falls_back_to_markdown_when_json_has_no_content(self):
        markdown = render_document_to_markdown({"kids": []}, "# Fallback")

        self.assertEqual(markdown, "# Fallback")

    def test_renders_metric_grid_as_key_value_table(self):
        doc = {
            "kids": [
                {"type": "paragraph", "page number": 1, "bounding box": [100, 700, 200, 720], "font size": 18, "content": "Campus Facts"},
                {"type": "paragraph", "page number": 1, "bounding box": [100, 670, 240, 690], "font size": 10, "content": "Current summary."},
                {"type": "paragraph", "page number": 1, "bounding box": [100, 600, 150, 620], "font size": 8, "content": "Founded"},
                {"type": "paragraph", "page number": 1, "bounding box": [250, 600, 300, 620], "font size": 8, "content": "Students"},
                {"type": "heading", "page number": 1, "bounding box": [100, 550, 170, 590], "font size": 30, "content": "1958"},
                {"type": "heading", "page number": 1, "bounding box": [250, 550, 330, 590], "font size": 30, "content": "7000"},
                {"type": "paragraph", "page number": 1, "bounding box": [100, 500, 150, 520], "font size": 8, "content": "Faculty"},
                {"type": "paragraph", "page number": 1, "bounding box": [250, 500, 300, 520], "font size": 8, "content": "Dorm"},
                {"type": "heading", "page number": 1, "bounding box": [100, 450, 170, 490], "font size": 30, "content": "330"},
                {"type": "heading", "page number": 1, "bounding box": [250, 450, 330, 490], "font size": 30, "content": "569"},
            ]
        }

        markdown = render_document_pages_to_markdown(doc)

        self.assertIn("| \ud56d\ubaa9 | \ub0b4\uc6a9 |", markdown)
        self.assertIn("| Founded | 1958 |", markdown)
        self.assertIn("| Students | 7000 |", markdown)

    def test_renders_timeline_by_pairing_years_and_events(self):
        doc = {
            "kids": [
                {"type": "paragraph", "page number": 1, "bounding box": [100, 700, 230, 720], "font size": 18, "content": "History"},
                {"type": "heading", "page number": 1, "bounding box": [100, 600, 150, 640], "font size": 24, "content": "1958"},
                {"type": "paragraph", "page number": 1, "bounding box": [100, 560, 240, 590], "font size": 8, "content": "- School opened"},
                {"type": "heading", "page number": 1, "bounding box": [260, 600, 310, 640], "font size": 24, "content": "1970"},
                {"type": "paragraph", "page number": 1, "bounding box": [260, 560, 400, 590], "font size": 8, "content": "- College reorganized"},
                {"type": "heading", "page number": 1, "bounding box": [100, 480, 150, 520], "font size": 24, "content": "2023"},
                {"type": "paragraph", "page number": 1, "bounding box": [100, 440, 240, 470], "font size": 8, "content": "- New departments"},
                {"type": "heading", "page number": 1, "bounding box": [260, 480, 310, 520], "font size": 24, "content": "2024"},
                {"type": "paragraph", "page number": 1, "bounding box": [260, 440, 400, 470], "font size": 8, "content": "- Free major added"},
            ]
        }

        markdown = render_document_pages_to_markdown(doc)

        self.assertIn("- 1958: School opened", markdown)
        self.assertIn("- 2024: Free major added", markdown)


if __name__ == "__main__":
    unittest.main()
