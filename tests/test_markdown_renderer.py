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

    def test_renders_page_image_description_in_flow(self):
        doc = {
            "kids": [
                {"type": "heading", "page number": 1, "heading level": 1, "content": "Visual Notice"},
                {
                    "type": "picture",
                    "page number": 1,
                    "description": "The poster states that applications close on March 8.",
                },
                {"type": "paragraph", "page number": 1, "content": "Contact the admissions office for details."},
            ]
        }

        markdown = render_document_pages_to_markdown(doc)

        self.assertIn("# Visual Notice", markdown)
        self.assertIn("**Image summary:** The poster states that applications close on March 8.", markdown)
        self.assertLess(markdown.index("# Visual Notice"), markdown.index("**Image summary:**"))
        self.assertLess(markdown.index("**Image summary:**"), markdown.index("Contact the admissions office"))

    def test_renders_wide_multi_header_table_as_records(self):
        doc = {
            "kids": [
                {
                    "type": "table",
                    "rows": [
                        {
                            "type": "table row",
                            "cells": [
                                {"column number": 1, "content": "Department"},
                                {"column number": 2, "content": "Admission"},
                                {"column number": 8, "content": "Regular"},
                            ],
                        },
                        {
                            "type": "table row",
                            "cells": [
                                {"column number": 2, "content": "Early"},
                                {"column number": 5, "content": "Special"},
                                {"column number": 8, "content": "General"},
                            ],
                        },
                        {
                            "type": "table row",
                            "cells": [
                                {"column number": 1, "content": "Engineering"},
                                {"column number": 2, "content": "Mechanical"},
                                {"column number": 3, "content": "Day"},
                                {"column number": 4, "content": "2"},
                                {"column number": 5, "content": "100"},
                                {"column number": 6, "content": "49"},
                                {"column number": 7, "content": "21"},
                                {"column number": 8, "content": "5"},
                                {"column number": 9, "content": "1"},
                                {"column number": 10, "content": "1"},
                                {"column number": 11, "content": "◈"},
                                {"column number": 12, "content": "2"},
                                {"column number": 13, "content": "◈"},
                            ],
                        },
                    ],
                }
            ]
        }

        markdown = render_document_to_markdown(doc)

        self.assertIn("**Table records:**", markdown)
        self.assertNotIn("| Department |", markdown)
        self.assertIn("Department: Engineering", markdown)
        self.assertIn("Admission > Early", markdown)
        self.assertIn("Regular > General", markdown)
        self.assertIn("Regular > General: 5", markdown)
        self.assertNotIn("Regular > General 2", markdown)
        self.assertIn("Mechanical", markdown)

    def test_renders_complex_medium_table_as_records(self):
        doc = {
            "kids": [
                {
                    "type": "table",
                    "rows": [
                        {
                            "type": "table row",
                            "cells": [
                                {"column number": 1, "content": "Area"},
                                {"column number": 2, "content": "Program"},
                                {"column number": 3, "content": "2025"},
                                {"column number": 6, "content": "2026"},
                            ],
                        },
                        {
                            "type": "table row",
                            "cells": [
                                {"column number": 3, "content": "Spring"},
                                {"column number": 4, "content": "Fall"},
                                {"column number": 6, "content": "Spring"},
                                {"column number": 7, "content": "Fall"},
                                {"column number": 8, "content": "Notes"},
                            ],
                        },
                        {
                            "type": "table row",
                            "cells": [
                                {"column number": 1, "content": "Engineering"},
                                {"column number": 2, "content": "Mechanical"},
                                {"column number": 3, "content": "10"},
                                {"column number": 4, "content": "12"},
                                {"column number": 6, "content": "14"},
                                {"column number": 7, "content": "16"},
                                {"column number": 8, "content": "Open"},
                            ],
                        },
                        {
                            "type": "table row",
                            "cells": [
                                {"column number": 2, "content": "Electrical"},
                                {"column number": 3, "content": "8"},
                                {"column number": 4, "content": "9"},
                                {"column number": 6, "content": "10"},
                                {"column number": 7, "content": "11"},
                                {"column number": 8, "content": "Open"},
                            ],
                        },
                    ],
                }
            ]
        }

        markdown = render_document_to_markdown(doc)

        self.assertIn("**Table records:**", markdown)
        self.assertNotIn("| Area | Program |", markdown)
        self.assertIn("Area: Engineering; Program: Mechanical", markdown)
        self.assertIn("2025 > Spring: 10", markdown)
        self.assertIn("Area: Engineering; Program: Electrical", markdown)

    def test_keeps_simple_medium_table_as_markdown(self):
        doc = {
            "kids": [
                {
                    "type": "table",
                    "rows": [
                        {
                            "type": "table row",
                            "cells": [
                                {"column number": index, "content": f"H{index}"}
                                for index in range(1, 9)
                            ],
                        },
                        {
                            "type": "table row",
                            "cells": [
                                {"column number": index, "content": f"V{index}"}
                                for index in range(1, 9)
                            ],
                        },
                    ],
                }
            ]
        }

        markdown = render_document_to_markdown(doc)

        self.assertIn("| H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 |", markdown)
        self.assertNotIn("**Table records:**", markdown)


if __name__ == "__main__":
    unittest.main()
