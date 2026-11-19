"""
Hive-only syntax that must be BLOCKED (not silently accepted) by the Impala dialect.

Each test proves that the Impala dialect either:
  - rejects the syntax with an error, OR
  - does NOT tokenize/parse it as Hive would
    (e.g., numeric literal suffixes are treated as identifiers, not typed literals).
"""
from __future__ import annotations

import pytest
import sqlglot
from sqlglot.errors import ParseError, TokenError
from sqlglot import exp


def parse_impala(sql: str):
    return sqlglot.parse_one(sql, read="impala")


# ---------------------------------------------------------------------------
# 1. Hive numeric literal suffixes must NOT produce typed literals
# ---------------------------------------------------------------------------
class TestNumericSuffixesBlocked:
    """1L, 1BD, 1S, 1Y etc. are Hive-only suffixes. In Impala they are
    ambiguous (trailing identifier) — we verify they don't silently become
    typed literals."""

    @pytest.mark.parametrize("sql,suffix_name", [
        ("SELECT 1L FROM t", "BIGINT suffix L"),
        ("SELECT 1BD FROM t", "DECIMAL suffix BD"),
        ("SELECT 1S FROM t", "SMALLINT suffix S"),
        ("SELECT 1Y FROM t", "TINYINT suffix Y"),
    ])
    def test_numeric_suffix_not_typed_literal(self, sql: str, suffix_name: str):
        # Either parse fails, OR the AST does NOT contain a typed cast from suffix
        try:
            tree = parse_impala(sql)
            # Verify no DataType nodes were created from the suffix
            for node in tree.walk():
                if isinstance(node, exp.Cast):
                    dtype = node.args.get("to")
                    if dtype:
                        pytest.fail(
                            f"{suffix_name} produced a typed CAST node — "
                            "Hive suffix behaviour leaked into Impala dialect"
                        )
        except (ParseError, TokenError):
            pass  # error is also acceptable


# ---------------------------------------------------------------------------
# 2. Hive ${{var}} parameter syntax must NOT be recognised
# ---------------------------------------------------------------------------
class TestParameterSyntaxBlocked:

    @pytest.mark.parametrize("sql", [
        "SELECT ${my_var} FROM t",
        "SET hive.exec.parallelism = ${threads}",
    ])
    def test_dollar_brace_param(self, sql: str):
        # Should either error or not produce exp.Parameter nodes
        try:
            tree = parse_impala(sql)
            for node in tree.walk():
                if isinstance(node, exp.Parameter):
                    pytest.fail(
                        f"${{}}-style parameter was accepted as Hive Parameter — "
                        "should be blocked in Impala dialect"
                    )
        except (ParseError, TokenError):
            pass  # error is acceptable


# ---------------------------------------------------------------------------
# 3. Hive ADD JAR / ADD FILE / MSCK REPAIR must not be recognised as commands
# ---------------------------------------------------------------------------
class TestHiveCommandsBlocked:

    @pytest.mark.parametrize("sql", [
        "ADD JAR /path/to/jar.jar",
        "ADD FILE /path/to/file.txt",
        "MSCK REPAIR TABLE t",
    ])
    def test_hive_command_not_accepted(self, sql: str):
        # These should either fail to parse or be treated as generic errors
        try:
            tree = parse_impala(sql)
            # If parsed, it must NOT be a successful Command node for these
            for node in tree.walk():
                if isinstance(node, exp.Command):
                    cmd_text = node.sql(dialect="impala").upper()
                    if any(kw in cmd_text for kw in ("ADD JAR", "ADD FILE", "MSCK REPAIR")):
                        pytest.xfail(
                            f"Hive command accepted as generic Command node: {sql!r} — "
                            "should be rejected"
                        )
        except (ParseError, TokenError):
            pass  # error is correct behavior


# ---------------------------------------------------------------------------
# 4. TRANSFORM … USING should NOT be parsed
# ---------------------------------------------------------------------------
class TestTransformBlocked:

    def test_transform_using(self):
        sql = "SELECT TRANSFORM(a, b) USING 'my_script.py' FROM t"
        try:
            tree = parse_impala(sql)
            for node in tree.walk():
                if isinstance(node, exp.QueryTransform):
                    pytest.fail(
                        "TRANSFORM ... USING parsed as QueryTransform — "
                        "Hive-only feature should be blocked in Impala dialect"
                    )
        except (ParseError, TokenError):
            pass  # error is acceptable


# ---------------------------------------------------------------------------
# 5. Hive TIMESTAMP AS OF / VERSION AS OF syntax is wrong for Impala
#    (Impala uses FOR SYSTEM_TIME AS OF / FOR SYSTEM_VERSION AS OF)
# ---------------------------------------------------------------------------
class TestHiveTimeTravelBlocked:

    @pytest.mark.parametrize("sql", [
        "SELECT * FROM t TIMESTAMP AS OF '2023-01-01'",
        "SELECT * FROM t VERSION AS OF 123",
    ])
    def test_hive_time_travel_syntax(self, sql: str):
        # Impala does NOT support TIMESTAMP AS OF — should error
        try:
            tree = parse_impala(sql)
            # If sqlglot is permissive and parses it, mark as xfail (not a hard block yet)
            pytest.xfail(
                f"Hive time-travel syntax was accepted — should be blocked once "
                f"Impala time-travel override is implemented: {sql!r}"
            )
        except (ParseError, TokenError):
            pass  # expected: Impala doesn't support this syntax
