"""Fixture checks must not mistake parser crashes for rejected syntax."""
import pytest
import sqlglot
from sqlglot.errors import ParseError, TokenError

from conftest import ParserCase
from test_parser_coverage import _assert_case
import test_hive_blocked as blocked


def test_expected_error_propagates_runtime_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic parser crash")

    monkeypatch.setattr(sqlglot, "parse", fail)
    case = ParserCase("SELECT FROM", "error", "Synthetic.control", 2)
    with pytest.raises(RuntimeError, match="synthetic parser crash"):
        _assert_case(case)


@pytest.mark.parametrize("exception", [ParseError, TokenError])
def test_expected_error_accepts_syntax_error(monkeypatch, exception):
    def fail(*args, **kwargs):
        raise exception("synthetic syntax error")

    monkeypatch.setattr(sqlglot, "parse", fail)
    _assert_case(ParserCase("SELECT FROM", "error", "Synthetic.control", 2))


def test_positive_empty_statement_is_not_success(monkeypatch):
    monkeypatch.setattr(sqlglot, "parse", lambda *args, **kwargs: [None])
    with pytest.raises(pytest.fail.Exception, match="empty result"):
        _assert_case(ParserCase("SELECT 1", "ok", "Synthetic.control", 2))


def test_hive_block_control_propagates_runtime_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic parser crash")

    monkeypatch.setattr(sqlglot, "parse_one", fail)
    with pytest.raises(RuntimeError, match="synthetic parser crash"):
        blocked.TestTransformBlocked().test_transform_using()
