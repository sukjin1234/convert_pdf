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

    def test_renders_visual_title_body_grid_as_pairs(self):
        doc = {
            "kids": [
                {"type": "heading", "page number": 1, "bounding box": [70, 700, 210, 735], "font size": 26, "content": "국토교통부"},
                {"type": "heading", "page number": 1, "bounding box": [305, 700, 500, 735], "font size": 26, "content": "산업통상자원부"},
                {"type": "heading", "page number": 1, "bounding box": [560, 700, 690, 735], "font size": 26, "content": "교육부"},
                {"type": "paragraph", "page number": 1, "bounding box": [50, 610, 230, 650], "font size": 15, "content": "공간정보 특성화 전문대학"},
                {"type": "paragraph", "page number": 1, "bounding box": [320, 590, 500, 655], "font size": 15, "content": "창의융합형 공학인재\n양성지원사업"},
                {"type": "paragraph", "page number": 1, "bounding box": [570, 590, 700, 655], "font size": 15, "content": "고교 장애학생\n대학체험지원"},
            ]
        }

        markdown = render_document_pages_to_markdown(doc)

        self.assertIn("| 제목 | 내용 |", markdown)
        self.assertIn("| 국토교통부 | 공간정보 특성화 전문대학 |", markdown)
        self.assertIn("| 산업통상자원부 | 창의융합형 공학인재<br>양성지원사업 |", markdown)
        self.assertIn("| 교육부 | 고교 장애학생<br>대학체험지원 |", markdown)
        self.assertLess(markdown.index("공간정보 특성화 전문대학"), markdown.index("산업통상자원부"))

    def test_renders_visual_pair_grid_with_multiple_body_lines(self):
        doc = {
            "kids": [
                {"type": "paragraph", "page number": 1, "bounding box": [40, 720, 230, 750], "font size": 20, "content": "연계 편입학 협약대학"},
                {"type": "heading", "page number": 1, "bounding box": [40, 620, 190, 650], "font size": 22, "content": "글로컬캠퍼스"},
                {"type": "heading", "page number": 1, "bounding box": [430, 620, 560, 650], "font size": 22, "content": "세종캠퍼스"},
                {"type": "paragraph", "page number": 1, "bounding box": [40, 580, 190, 600], "font size": 13, "content": "- 메카트로닉스공학과"},
                {"type": "paragraph", "page number": 1, "bounding box": [40, 550, 190, 570], "font size": 13, "content": "- 컴퓨터공학과"},
                {"type": "paragraph", "page number": 1, "bounding box": [430, 580, 590, 600], "font size": 13, "content": "- 컴퓨터융합소프트웨어학과"},
                {"type": "paragraph", "page number": 1, "bounding box": [430, 550, 590, 570], "font size": 13, "content": "- 신소재화학과"},
            ]
        }

        markdown = render_document_pages_to_markdown(doc)

        self.assertIn("## 연계 편입학 협약대학", markdown)
        self.assertIn("| 글로컬캠퍼스 | - 메카트로닉스공학과<br>- 컴퓨터공학과 |", markdown)
        self.assertIn("| 세종캠퍼스 | - 컴퓨터융합소프트웨어학과<br>- 신소재화학과 |", markdown)


if __name__ == "__main__":
    unittest.main()
