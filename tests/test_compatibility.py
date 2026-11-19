"""Synthetic checks for SQLGlot API compatibility."""
from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from impala_sqlglot import Impala
from impala_sqlglot.generator import ImpalaGenerator
from impala_sqlglot.parser import ImpalaParser
from impala_sqlglot.tokenizer import ImpalaTokenizer


def test_registered_plugin_components():
    assert sqlglot.Dialect.get("impala") is Impala
    dialect = sqlglot.Dialect.get_or_raise("impala")
    assert type(dialect.tokenizer()) is ImpalaTokenizer
    assert type(dialect.parser()) is ImpalaParser
    assert type(dialect.generator()) is ImpalaGenerator


def test_basic_query_generation_and_reparse():
    sql = "SELECT demo_key FROM demo_input WHERE demo_key = 1"
    tree = sqlglot.parse_one(sql, read="impala")
    assert isinstance(tree, exp.Select)
    assert tree.find(exp.Table).name == "demo_input"
    generated = tree.sql(dialect="impala")
    assert generated == sql
    assert sqlglot.parse_one(generated, read="impala") == tree


@pytest.mark.parametrize(
    ("phrase", "value", "kind"),
    [
        ("SYSTEM_TIME", "'2024-01-01 00:00:00'", "TIMESTAMP"),
        ("SYSTEM_VERSION", "7", "VERSION"),
    ],
)
def test_time_travel_structure(phrase, value, kind):
    tree = sqlglot.parse_one(
        f"SELECT demo_key FROM demo_input FOR {phrase} AS OF {value} AS demo_alias",
        read="impala",
    )
    table = tree.find(exp.Table)
    assert table.name == "demo_input"
    assert table.alias == "demo_alias"
    version = table.args["version"]
    assert isinstance(version, exp.Version)
    assert version.this == kind
    assert version.args["kind"] == "AS OF"
    assert version.expression == sqlglot.parse_one(value, read="impala")


@pytest.mark.parametrize("phrase", ["SYSTEM_TIME", "SYSTEM_VERSION"])
def test_time_travel_requires_value(phrase):
    with pytest.raises(ParseError):
        sqlglot.parse_one(
            f"SELECT demo_key FROM demo_input FOR {phrase} AS OF",
            read="impala",
        )
