"""Synthetic postfix-factorial checks with explicit semantic expectations."""
from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from impala_sqlglot import Impala


@pytest.mark.parametrize(
    ("sql", "expected", "generated"),
    [
        (
            "SELECT 4!",
            exp.Factorial(this=exp.Literal.number(4)),
            "SELECT FACTORIAL(4)",
        ),
        (
            "SELECT 4 + 4!",
            exp.Add(
                this=exp.Literal.number(4),
                expression=exp.Factorial(this=exp.Literal.number(4)),
            ),
            "SELECT 4 + FACTORIAL(4)",
        ),
        (
            "SELECT 4! = 24",
            exp.EQ(
                this=exp.Factorial(this=exp.Literal.number(4)),
                expression=exp.Literal.number(24),
            ),
            "SELECT FACTORIAL(4) = 24",
        ),
        (
            "SELECT -4!",
            exp.Neg(this=exp.Factorial(this=exp.Literal.number(4))),
            "SELECT -FACTORIAL(4)",
        ),
        (
            "SELECT (4 + 1)!",
            exp.Factorial(
                this=exp.Paren(
                    this=exp.Add(
                        this=exp.Literal.number(4),
                        expression=exp.Literal.number(1),
                    )
                )
            ),
            "SELECT FACTORIAL((4 + 1))",
        ),
        (
            "SELECT 4!!",
            exp.Factorial(this=exp.Factorial(this=exp.Literal.number(4))),
            "SELECT FACTORIAL(FACTORIAL(4))",
        ),
        (
            "SELECT FACTORIAL(4)",
            exp.Factorial(this=exp.Literal.number(4)),
            "SELECT FACTORIAL(4)",
        ),
    ],
)
def test_factorial_structure_generation_and_reparse(sql, expected, generated):
    assert sqlglot.Dialect.get("impala") is Impala
    tree = sqlglot.parse_one(sql, read="impala")
    assert tree.expressions == [expected]
    assert tree.sql(dialect="impala") == generated
    assert sqlglot.parse_one(generated, read="impala") == tree


def test_factorial_keeps_column_and_table():
    tree = sqlglot.parse_one(
        "SELECT demo_count! AS demo_result FROM demo_input",
        read="impala",
    )
    factorial = tree.expressions[0].this
    assert isinstance(factorial, exp.Factorial)
    assert isinstance(factorial.this, exp.Column)
    assert factorial.this.name == "demo_count"
    assert tree.expressions[0].alias == "demo_result"
    assert tree.find(exp.Table).name == "demo_input"
    generated = "SELECT FACTORIAL(demo_count) AS demo_result FROM demo_input"
    assert tree.sql(dialect="impala") == generated
    assert sqlglot.parse_one(generated, read="impala") == tree


@pytest.mark.parametrize("sql", ["SELECT !TRUE", "SELECT NOT TRUE"])
def test_prefix_not_remains_logical(sql):
    tree = sqlglot.parse_one(sql, read="impala")
    assert tree.expressions == [exp.Not(this=exp.Boolean(this=True))]
    assert tree.sql(dialect="impala") == "SELECT NOT TRUE"
    assert tree.find(exp.Factorial) is None


def test_not_equal_remains_comparison():
    tree = sqlglot.parse_one("SELECT 4 != 5", read="impala")
    assert tree.expressions == [
        exp.NEQ(this=exp.Literal.number(4), expression=exp.Literal.number(5))
    ]
    assert tree.sql(dialect="impala") == "SELECT 4 <> 5"


@pytest.mark.parametrize("sql", ["SELECT !", "SELECT 4! +", "SELECT (4 +)!"])
def test_factorial_rejects_missing_operand(sql):
    with pytest.raises(ParseError):
        sqlglot.parse_one(sql, read="impala")
