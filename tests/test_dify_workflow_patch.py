import copy
import unittest

from scripts.dify_workflow_patch import (
    BUILD_LOOKUP_ID,
    DIRECT_ANSWER_ID,
    DIRECT_ASSIGNER_ID,
    DIRECT_IF_ID,
    DIRECT_STATE_ID,
    console_origin,
    patch_workflow,
    validate_patched_graph,
)


def node(node_id, node_type, title, x=0, y=0, **data):
    return {
        "id": node_id,
        "type": "custom",
        "position": {"x": x, "y": y},
        "data": {"type": node_type, "title": title, "selected": False, **data},
    }


def edge(source, target, source_type, target_type):
    return {
        "id": f"{source}-source-{target}-target",
        "type": "custom",
        "source": source,
        "target": target,
        "sourceHandle": "source",
        "targetHandle": "target",
        "data": {"sourceType": source_type, "targetType": target_type, "isInLoop": False},
        "zIndex": 0,
    }


def sample_graph():
    state_code = "def main(question='', answer='', source_summary=''): return {}"
    nodes = [
        node("start", "start", "Start", variables=[]),
        node("rewrite", "llm", "Rewrite Standalone Query"),
        node("plan", "http-request", "Query Plan", body={}),
        node("parse-plan", "code", "Parse Query Plan", code="", outputs={}),
        node("lookup", "http-request", "Structured Lookup", body={}),
        node("knowledge", "knowledge-retrieval", "Knowledge", query_variable_selector=[]),
        node("build-merge", "code", "Build Merge Evidence Body", code="", outputs={}),
        node("parse-merge", "code", "Parse Merge Evidence", code="", outputs={}),
        node("final", "llm", "Final Answer"),
        node(
            "state",
            "code",
            "Build Conversation State",
            code=state_code,
            code_language="python3",
            variables=[],
            outputs={"last_question": {"type": "string", "children": None}},
        ),
        node(
            "assign",
            "assigner",
            "Assigner",
            items=[{"variable_selector": ["conversation", "last_question"], "value": ["state", "last_question"]}],
        ),
        node("if", "if-else", "IF", cases=[]),
        node("answer", "answer", "Answer", answer="{{#final.text#}}", variables=[]),
    ]
    return {
        "nodes": nodes,
        "edges": [
            edge("parse-plan", "lookup", "code", "http-request"),
            edge("lookup", "knowledge", "http-request", "knowledge-retrieval"),
            edge("knowledge", "build-merge", "knowledge-retrieval", "code"),
            edge("parse-merge", "final", "code", "llm"),
        ],
    }


class DifyWorkflowPatchTests(unittest.TestCase):
    def test_patch_adds_generic_query_and_direct_answer_path(self):
        graph = sample_graph()

        changes = patch_workflow(graph)
        validate_patched_graph(graph)

        self.assertTrue(changes)
        ids = {item["id"] for item in graph["nodes"]}
        self.assertTrue({BUILD_LOOKUP_ID, DIRECT_IF_ID, DIRECT_STATE_ID, DIRECT_ASSIGNER_ID, DIRECT_ANSWER_ID} <= ids)
        start = next(item for item in graph["nodes"] if item["id"] == "start")
        self.assertEqual(start["data"]["variables"][0]["variable"], "document_id")
        plan = next(item for item in graph["nodes"] if item["id"] == "plan")
        plan_body = plan["data"]["body"]["data"][0]["value"]
        self.assertIn("{{#sys.query#}}", plan_body)
        self.assertIn('"retrieval_query"', plan_body)
        direct_if = next(item for item in graph["nodes"] if item["id"] == DIRECT_IF_ID)
        self.assertEqual(direct_if["data"]["cases"][0]["conditions"][0]["variable_selector"], ["parse-merge", "has_direct_answer"])
        direct_answer = next(item for item in graph["nodes"] if item["id"] == DIRECT_ANSWER_ID)
        self.assertEqual(direct_answer["data"]["answer"], "{{#parse-merge.deterministic_answer#}}")
        edge_pairs = {(item["source"], item["target"]) for item in graph["edges"]}
        self.assertNotIn(("lookup", "knowledge"), edge_pairs)
        self.assertIn(("parse-plan", "knowledge"), edge_pairs)
        self.assertIn(("lookup", "build-merge"), edge_pairs)

    def test_patch_is_idempotent(self):
        graph = sample_graph()
        patch_workflow(graph)
        once = copy.deepcopy(graph)

        patch_workflow(graph)

        self.assertEqual(graph, once)

    def test_console_origin_removes_api_path(self):
        self.assertEqual(console_origin("https://dify.example/v1"), "https://dify.example")


if __name__ == "__main__":
    unittest.main()
