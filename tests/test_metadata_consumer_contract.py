"""Synthetic consumer checks for the structured statistics partition API."""
import pytest
import sqlglot
from sqlglot import exp

import impala_sqlglot  # noqa: F401
from impala_sqlglot.metadata import ImpalaComputeStats, ImpalaDropStats


@pytest.mark.parametrize("operation, cls", [
    ("COMPUTE INCREMENTAL STATS", ImpalaComputeStats),
    ("DROP INCREMENTAL STATS", ImpalaDropStats),
])
@pytest.mark.parametrize("predicate, expected_types", [
    ("part_col > 5", [exp.GT]),
    ("part_col >= 5 AND part_col < 10", [exp.And]),
    ("part_col = 5, region_col IN ('east', 'west')", [exp.EQ, exp.In]),
])
def test_consumers_read_partition_expression_children(operation, cls, predicate, expected_types):
    tree = sqlglot.parse_one(
        f"{operation} demo_table PARTITION ({predicate})",
        read="impala", error_level=sqlglot.ErrorLevel.RAISE,
    )
    assert isinstance(tree, cls)
    partition = tree.args["partition"]
    assert isinstance(partition, exp.Partition)
    # Partition is an AST node, not the old providers' list of key/value pairs.
    predicates = partition.expressions
    assert [type(item) for item in predicates] == expected_types
    assert all(item.parent is partition for item in predicates)
    lowered = [item.sql(dialect="impala") for item in predicates]
    assert ", ".join(lowered) == predicate
    assert sqlglot.parse_one(tree.sql(dialect="impala"), read="impala") == tree


def test_absent_statistics_partition_remains_absent():
    tree = sqlglot.parse_one("COMPUTE STATS demo_table", read="impala")
    assert tree.args.get("partition") is None
