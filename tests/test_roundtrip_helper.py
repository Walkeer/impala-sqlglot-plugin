"""Controls proving the round-trip assertion detects changes and errors."""
import pytest
import sqlglot
from sqlglot.errors import ParseError

from conftest import ParserCase
from test_roundtrip import assert_roundtrip


def _case(sql="SELECT key_col FROM demo_input"):
    return ParserCase(sql=sql, expect="ok", source="synthetic", phase=3)


def _generated(monkeypatch, *outputs):
    monkeypatch.setattr(sqlglot, "transpile", lambda *args, **kwargs: list(outputs))


def test_equivalent_formatting_passes(monkeypatch):
    _generated(monkeypatch, "select key_col\nfrom demo_input;")
    assert_roundtrip(_case())


def test_unrelated_query_fails(monkeypatch):
    _generated(monkeypatch, "SELECT 999 FROM demo_other")
    with pytest.raises(AssertionError, match="AST differs"):
        assert_roundtrip(_case())


def test_unparseable_output_fails(monkeypatch):
    _generated(monkeypatch, "SELECT FROM")
    with pytest.raises(ParseError):
        assert_roundtrip(_case())


def test_generator_exception_propagates(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic generator failure")

    monkeypatch.setattr(sqlglot, "transpile", fail)
    with pytest.raises(RuntimeError, match="synthetic generator failure"):
        assert_roundtrip(_case())


@pytest.mark.parametrize("outputs", [(), ("",), ("  ",), (None,)])
def test_empty_output_fails(monkeypatch, outputs):
    _generated(monkeypatch, *outputs)
    with pytest.raises(AssertionError):
        assert_roundtrip(_case())


def test_extra_output_statement_fails(monkeypatch):
    _generated(monkeypatch, "SELECT key_col FROM demo_input; SELECT 999")
    with pytest.raises(AssertionError, match="one statement"):
        assert_roundtrip(_case())


def test_extra_transpiled_statement_fails(monkeypatch):
    _generated(monkeypatch, "SELECT key_col FROM demo_input", "SELECT 999")
    with pytest.raises(AssertionError, match="number of statements"):
        assert_roundtrip(_case())


def test_semantically_different_literal_fails(monkeypatch):
    _generated(monkeypatch, "SELECT key_col FROM demo_input WHERE key_col = 2")
    with pytest.raises(AssertionError, match="AST differs"):
        assert_roundtrip(_case("SELECT key_col FROM demo_input WHERE key_col = 1"))


def test_ordinary_comment_may_differ(monkeypatch):
    _generated(monkeypatch, "SELECT key_col FROM demo_input")
    assert_roundtrip(_case("SELECT key_col /* ordinary comment */ FROM demo_input"))


@pytest.mark.parametrize(
    "out",
    [
        "SELECT a FROM l JOIN r ON l.a = r.a",
        "SELECT a FROM l JOIN /* +SHUFFLE */ r ON l.a = r.a",
    ],
    ids=["removed", "changed"],
)
def test_comment_plan_hint_changes_fail(monkeypatch, out):
    _generated(monkeypatch, out)
    sql = "SELECT a FROM l JOIN /* +BROADCAST */ r ON l.a = r.a"
    with pytest.raises(AssertionError, match="AST differs"):
        assert_roundtrip(_case(sql))


def test_comment_plan_hint_moved_to_another_table_fails(monkeypatch):
    _generated(monkeypatch, "SELECT a FROM l /* +BROADCAST */ JOIN r ON l.a = r.a")
    sql = "SELECT a FROM l JOIN /* +BROADCAST */ r ON l.a = r.a"
    with pytest.raises(AssertionError, match="AST differs"):
        assert_roundtrip(_case(sql))


def test_comment_plan_hint_preserved_passes(monkeypatch):
    _generated(monkeypatch, "SELECT a FROM l JOIN /* +BROADCAST */ r ON l.a = r.a")
    sql = "SELECT a FROM l JOIN -- +BROADCAST\n r ON l.a = r.a"
    assert_roundtrip(_case(sql))
