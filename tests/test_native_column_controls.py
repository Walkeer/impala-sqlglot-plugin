"""Synthetic controls separating native nullability from table-key flags."""
import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

import impala_sqlglot  # noqa: F401


@pytest.mark.parametrize("nullability", ["NOT NULL", "NULL"])
@pytest.mark.parametrize("flags", [
    "DISABLE NOVALIDATE RELY",
    "ENABLE NOVALIDATE RELY",
    "DISABLE",
    "NOVALIDATE",
    "RELY",
])
def test_table_key_flags_are_not_column_nullability_options(nullability, flags):
    sql = (
        f"CREATE TABLE demo_table (demo_key BIGINT {nullability} {flags}) "
        "STORED AS PARQUET"
    )
    with pytest.raises((ParseError, TokenError)):
        sqlglot.parse_one(sql, read="impala")


@pytest.mark.parametrize("nullability", ["NOT NULL", "NULL"])
def test_native_nullability_remains_structured(nullability):
    sql = (
        f"CREATE TABLE demo_table (demo_key BIGINT {nullability} COMMENT 'invented') "
        "STORED AS PARQUET"
    )
    tree = sqlglot.parse_one(sql, read="impala")
    assert isinstance(tree, exp.Create)
    assert isinstance(tree.find(exp.ColumnDef), exp.ColumnDef)
    assert sqlglot.parse_one(tree.sql(dialect="impala"), read="impala") == tree
