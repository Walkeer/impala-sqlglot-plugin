"""
Round-trip tests: ParsesOk cases for phases 3+ must survive
transpile(sql, read="impala", write="impala") and produce equivalent SQL.

Enabled starting Phase 3.
Run:
    uv run pytest tests/test_roundtrip.py -m phase3 -v
"""
from __future__ import annotations

import pytest
import sqlglot

import impala_sqlglot  # Register the plugin before selecting the dialect.
from conftest import ParserCase, case_id, load_fixtures


def _hint_comments(tree):
    """Keep attachment and text of comment-form hints, unlike ordinary comments.

    SQLGlot's expression equality ignores comment metadata. A leading '+' in
    a comment is significant in Impala, so compare its position in the tree
    and its text separately. Structured hint expressions remain part of normal
    AST equality. Only surrounding comment whitespace is ignored.
    """
    hints = []
    for node in tree.walk():
        path = []
        current = node
        while current.parent is not None:
            path.append((current.arg_key, current.index))
            current = current.parent
        for comment in node.comments or ():
            text = comment.strip()
            if text.startswith("+"):
                hints.append((tuple(reversed(path)), text))
    return hints


def assert_roundtrip(case: ParserCase) -> None:
    if case.known_limitation:
        pytest.xfail(f"Known limitation: {case.known_limitation}")
    sql = case.sql
    trees_in = sqlglot.parse(sql, read="impala")
    assert trees_in and all(tree is not None for tree in trees_in), (
        f"Round-trip input contains an empty statement: {sql!r}"
    )
    outputs = sqlglot.transpile(sql, read="impala", write="impala")
    assert len(outputs) == len(trees_in), (
        f"Round-trip changed the number of statements: {sql!r}; output: {outputs!r}"
    )
    for tree_in, out in zip(trees_in, outputs):
        assert isinstance(out, str) and out.strip(), (
            f"Round-trip produced empty output for: {sql!r}"
        )
        trees_out = sqlglot.parse(out, read="impala")
        assert len(trees_out) == 1 and trees_out[0] is not None, (
            f"Round-trip output must contain one statement: {out!r}"
        )
        tree_out = trees_out[0]
        assert tree_in == tree_out, (
            f"Round-trip AST differs:\n  IN:  {sql!r}\n  OUT: {out!r}"
            f"\n  IN AST:  {tree_in!r}\n  OUT AST: {tree_out!r}"
        )
        assert _hint_comments(tree_in) == _hint_comments(tree_out), (
            f"Round-trip hint comments differ:\n  IN:  {sql!r}\n  OUT: {out!r}"
            f"\n  IN hints:  {_hint_comments(tree_in)!r}"
            f"\n  OUT hints: {_hint_comments(tree_out)!r}"
        )


def _ok_cases(phase: int) -> list[ParserCase]:
    return [c for c in load_fixtures(phase=phase) if c.expect == "ok"]


# Phase 3
@pytest.mark.phase3
@pytest.mark.parametrize(
    "case",
    _ok_cases(3),
    ids=[case_id(c) for c in _ok_cases(3)],
)
def test_roundtrip_phase3(case: ParserCase):
    assert_roundtrip(case)


# Phase 4
@pytest.mark.phase4
@pytest.mark.parametrize(
    "case",
    _ok_cases(4),
    ids=[case_id(c) for c in _ok_cases(4)],
)
def test_roundtrip_phase4(case: ParserCase):
    assert_roundtrip(case)


# Phase 5
@pytest.mark.phase5
@pytest.mark.parametrize(
    "case",
    _ok_cases(5),
    ids=[case_id(c) for c in _ok_cases(5)],
)
def test_roundtrip_phase5(case: ParserCase):
    assert_roundtrip(case)
