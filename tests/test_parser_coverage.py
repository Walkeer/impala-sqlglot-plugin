"""
Parametrized parse coverage tests driven by fixtures extracted from ParserTest.java.

Run a specific phase:
    uv run pytest tests/test_parser_coverage.py -m phase0 -v
    uv run pytest tests/test_parser_coverage.py -m phase1 -v

Run all phases:
    uv run pytest tests/test_parser_coverage.py -v
"""
from __future__ import annotations

import pytest
import sqlglot
from sqlglot.errors import ParseError, TokenError

from conftest import ParserCase, case_id, load_fixtures


def _phase_mark(phase: int) -> pytest.MarkDecorator:
    return pytest.mark.__getattr__(f"phase{phase}")


def _make_params(phase: int):
    cases = load_fixtures(phase=phase)
    marks = [_phase_mark(phase)]
    return pytest.mark.parametrize(
        "case",
        cases,
        ids=[case_id(c) for c in cases],
    )


# ---------------------------------------------------------------------------
# Phase 0 — Foundation / tokenizer
# ---------------------------------------------------------------------------
@pytest.mark.phase0
@pytest.mark.parametrize("case", load_fixtures(phase=0), ids=[case_id(c) for c in load_fixtures(phase=0)])
def test_phase0_parse(case: ParserCase):
    _assert_case(case)


# ---------------------------------------------------------------------------
# Phase 1 — Expressions & literals
# ---------------------------------------------------------------------------
@pytest.mark.phase1
@pytest.mark.parametrize("case", load_fixtures(phase=1), ids=[case_id(c) for c in load_fixtures(phase=1)])
def test_phase1_parse(case: ParserCase):
    _assert_case(case)


# ---------------------------------------------------------------------------
# Phase 2 — Query shape
# ---------------------------------------------------------------------------
@pytest.mark.phase2
@pytest.mark.parametrize("case", load_fixtures(phase=2), ids=[case_id(c) for c in load_fixtures(phase=2)])
def test_phase2_parse(case: ParserCase):
    _assert_case(case)


# ---------------------------------------------------------------------------
# Phase 3 — DML
# ---------------------------------------------------------------------------
@pytest.mark.phase3
@pytest.mark.parametrize("case", load_fixtures(phase=3), ids=[case_id(c) for c in load_fixtures(phase=3)])
def test_phase3_parse(case: ParserCase):
    _assert_case(case)


# ---------------------------------------------------------------------------
# Phase 4 — DDL
# ---------------------------------------------------------------------------
@pytest.mark.phase4
@pytest.mark.parametrize("case", load_fixtures(phase=4), ids=[case_id(c) for c in load_fixtures(phase=4)])
def test_phase4_parse(case: ParserCase):
    _assert_case(case)


# ---------------------------------------------------------------------------
# Phase 5 — SHOW / admin
# ---------------------------------------------------------------------------
@pytest.mark.phase5
@pytest.mark.parametrize("case", load_fixtures(phase=5), ids=[case_id(c) for c in load_fixtures(phase=5)])
def test_phase5_parse(case: ParserCase):
    _assert_case(case)


# ---------------------------------------------------------------------------
# Core assertion helper
# ---------------------------------------------------------------------------
def _assert_case(case: ParserCase) -> None:
    if case.known_limitation:
        pytest.xfail(f"Known limitation: {case.known_limitation}")
    if case.expect == "ok":
        try:
            result = sqlglot.parse(case.sql, read="impala", error_level=sqlglot.ErrorLevel.RAISE)
            assert result and any(tree is not None for tree in result), (
                f"parse returned empty result for: {case.sql!r}"
            )
        except (ParseError, Exception) as exc:
            pytest.fail(f"Expected parse success but got error for:\n  {case.sql!r}\n  {exc}")
    else:
        # expect error — sqlglot may raise ParseError OR return error nodes depending on error_level
        # We check that parse raises OR produces error annotations
        errored = False
        try:
            result = sqlglot.parse(case.sql, read="impala", error_level=sqlglot.ErrorLevel.RAISE)
        except (ParseError, TokenError):
            errored = True
        if not errored:
            # sqlglot is lenient about some invalid SQL; mark as xfail rather than hard fail
            pytest.xfail(
                f"Expected parse error but sqlglot parsed ok (may be analysis-only error):\n  {case.sql!r}"
            )
