import ast
from pathlib import Path


def test_worker_uses_frozen_evaluator_class_count():
    here = Path(__file__).resolve().parent
    eval_path = next(path for path in (here / "eval.py", here.parent / "eval.py") if path.exists())
    source = eval_path.read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "func_per_iteration"
    )
    method_source = ast.get_source_segment(source, method)

    assert "hist_info(self.class_num, pred, label)" in method_source
    assert "label < self.class_num" in method_source
    assert "hist_info(config.num_classes" not in method_source
