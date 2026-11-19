"""Synthetic tests for structured Impala statistics and metadata statements."""
from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from impala_sqlglot.metadata import (
    ImpalaComputeStats,
    ImpalaDropStats,
    ImpalaInvalidateMetadata,
    ImpalaRefresh,
)


def identifier(name: str, quoted: bool = False) -> exp.Identifier:
    return exp.Identifier(this=name, quoted=quoted)


def table(name: str, db: str | None = None, quoted: bool = False) -> exp.Table:
    args = {"this": identifier(name, quoted)}
    if db is not None:
        args["db"] = identifier(db, quoted)
    return exp.Table(**args)


def comparison(name: str, operator: type[exp.Expression], value: exp.Expression):
    return operator(this=exp.Column(this=identifier(name)), expression=value)


def static_pair(name: str, value: exp.Expression) -> exp.EQ:
    return exp.EQ(this=identifier(name), expression=value)


def partition(*items: exp.Expression) -> exp.Partition:
    return exp.Partition(expressions=list(items))


@pytest.mark.parametrize(
    ("source", "expected", "generated"),
    [
        (
            "compute stats raw_events",
            ImpalaComputeStats(this=table("raw_events"), incremental=False),
            "COMPUTE STATS raw_events",
        ),
        (
            "compute stats `demo-db`.`events` (`group`, session_key) "
            "tablesample system(23) repeatable(4)",
            ImpalaComputeStats(
                this=table("events", "demo-db", quoted=True),
                incremental=False,
                columns=[identifier("group", True), identifier("session_key")],
                sample=exp.TableSample(
                    method=exp.Var(this="SYSTEM"),
                    percent=exp.Literal.number(23),
                    seed=exp.Literal.number(4),
                ),
            ),
            "COMPUTE STATS `demo-db`.`events` (`group`, session_key) "
            "TABLESAMPLE SYSTEM(23) REPEATABLE(4)",
        ),
        (
            "compute stats demo_db.events () tablesample system(13)",
            ImpalaComputeStats(
                this=table("events", "demo_db"),
                incremental=False,
                columns=[],
                sample=exp.TableSample(
                    method=exp.Var(this="SYSTEM"), percent=exp.Literal.number(13)
                ),
            ),
            "COMPUTE STATS demo_db.events () TABLESAMPLE SYSTEM(13)",
        ),
        (
            "compute stats ` demo-db `.` events ` (` account_key `) "
            "tablesample ` system `(17)",
            ImpalaComputeStats(
                this=table("events", "demo-db", quoted=True),
                incremental=False,
                columns=[identifier("account_key", True)],
                sample=exp.TableSample(
                    method=exp.Var(this="SYSTEM"), percent=exp.Literal.number(17)
                ),
            ),
            "COMPUTE STATS `demo-db`.`events` (`account_key`) TABLESAMPLE SYSTEM(17)",
        ),
        (
            "compute incremental stats demo_db.events",
            ImpalaComputeStats(this=table("events", "demo_db"), incremental=True),
            "COMPUTE INCREMENTAL STATS demo_db.events",
        ),
        (
            "compute incremental stats demo_db.events ()",
            ImpalaComputeStats(
                this=table("events", "demo_db"), incremental=True, columns=[]
            ),
            "COMPUTE INCREMENTAL STATS demo_db.events ()",
        ),
        (
            "compute incremental stats demo_db.events "
            "partition (region = 'north', fiscal_year >= 2024) "
            "(customer_key, net_amt)",
            ImpalaComputeStats(
                this=table("events", "demo_db"),
                incremental=True,
                partition=partition(
                    comparison("region", exp.EQ, exp.Literal.string("north")),
                    comparison("fiscal_year", exp.GTE, exp.Literal.number(2024)),
                ),
                columns=[identifier("customer_key"), identifier("net_amt")],
            ),
            "COMPUTE INCREMENTAL STATS demo_db.events "
            "PARTITION (region = 'north', fiscal_year >= 2024) "
            "(customer_key, net_amt)",
        ),
        (
            "drop stats demo_db.events",
            ImpalaDropStats(this=table("events", "demo_db"), incremental=False),
            "DROP STATS demo_db.events",
        ),
        (
            "drop incremental stats demo_db.events "
            "partition (fiscal_year >= 2022, region = 'west')",
            ImpalaDropStats(
                this=table("events", "demo_db"),
                incremental=True,
                partition=partition(
                    comparison("fiscal_year", exp.GTE, exp.Literal.number(2022)),
                    comparison("region", exp.EQ, exp.Literal.string("west")),
                ),
            ),
            "DROP INCREMENTAL STATS demo_db.events "
            "PARTITION (fiscal_year >= 2022, region = 'west')",
        ),
        (
            "refresh demo_db.events",
            ImpalaRefresh(this=table("events", "demo_db"), kind="TABLE", partitions=[]),
            "REFRESH demo_db.events",
        ),
        (
            "refresh demo_db.events partition (region = 'north', fiscal_year = 2024) "
            "partition (region = 'south', fiscal_year = 2025)",
            ImpalaRefresh(
                this=table("events", "demo_db"),
                kind="TABLE",
                partitions=[
                    partition(
                        static_pair("region", exp.Literal.string("north")),
                        static_pair("fiscal_year", exp.Literal.number(2024)),
                    ),
                    partition(
                        static_pair("region", exp.Literal.string("south")),
                        static_pair("fiscal_year", exp.Literal.number(2025)),
                    ),
                ],
            ),
            "REFRESH demo_db.events PARTITION (region = 'north', fiscal_year = 2024) "
            "PARTITION (region = 'south', fiscal_year = 2025)",
        ),
        (
            "refresh functions `analytics-db`",
            ImpalaRefresh(this=identifier("analytics-db", True), kind="FUNCTIONS"),
            "REFRESH FUNCTIONS `analytics-db`",
        ),
        (
            "refresh authorization",
            ImpalaRefresh(kind="AUTHORIZATION"),
            "REFRESH AUTHORIZATION",
        ),
        (
            "invalidate metadata",
            ImpalaInvalidateMetadata(),
            "INVALIDATE METADATA",
        ),
        (
            "invalidate metadata `analytics-db`.`hourly`",
            ImpalaInvalidateMetadata(this=table("hourly", "analytics-db", quoted=True)),
            "INVALIDATE METADATA `analytics-db`.`hourly`",
        ),
        (
            "refresh `FUNCTIONS`",
            ImpalaRefresh(this=table("FUNCTIONS", quoted=True), kind="TABLE", partitions=[]),
            "REFRESH `FUNCTIONS`",
        ),
        (
            "refresh `AUTHORIZATION`",
            ImpalaRefresh(
                this=table("AUTHORIZATION", quoted=True), kind="TABLE", partitions=[]
            ),
            "REFRESH `AUTHORIZATION`",
        ),
    ],
)
def test_metadata_structure_generation_and_reparse(source, expected, generated):
    tree = sqlglot.parse_one(source, read="impala")
    assert type(tree) is type(expected)
    assert tree == expected
    assert not list(tree.find_all(exp.Command))
    assert tree.sql(dialect="impala") == generated
    assert sqlglot.parse_one(generated, read="impala") == tree


def test_stats_options_and_table_are_traversable():
    tree = sqlglot.parse_one(
        "COMPUTE STATS demo_db.events (account_key, event_code) "
        "TABLESAMPLE SYSTEM(27) REPEATABLE(6)",
        read="impala",
    )
    assert isinstance(tree, ImpalaComputeStats)
    assert tree.this == table("events", "demo_db")
    assert list(tree.find_all(exp.Table)) == [tree.this]
    assert tree.args["incremental"] is False
    assert tree.args["partition"] is None
    assert tree.args["columns"] == [identifier("account_key"), identifier("event_code")]
    sample = tree.args["sample"]
    assert isinstance(sample, exp.TableSample)
    assert sample.args["method"] == exp.Var(this="SYSTEM")
    assert sample.args["percent"] == exp.Literal.number(27)
    assert sample.args["seed"] == exp.Literal.number(6)


def test_empty_column_list_is_distinct_from_absent_list():
    absent = sqlglot.parse_one("COMPUTE STATS demo_events", read="impala")
    empty = sqlglot.parse_one("COMPUTE STATS demo_events ()", read="impala")
    assert absent.args["columns"] is None
    assert empty.args["columns"] == []
    assert absent.sql(dialect="impala") == "COMPUTE STATS demo_events"
    assert empty.sql(dialect="impala") == "COMPUTE STATS demo_events ()"


def test_quoted_identifier_trimming_matches_native_scanner():
    source = "COMPUTE STATS `\u00a0demo_events\u00a0`"
    tree = sqlglot.parse_one(source, read="impala")
    assert tree.this.name == "\u00a0demo_events\u00a0"
    assert tree.sql(dialect="impala") == source
    assert sqlglot.parse_one(source, read="impala") == tree


def test_refresh_and_incremental_partition_shapes_are_distinct():
    refresh = sqlglot.parse_one(
        "REFRESH demo_events PARTITION (region = 'east')", read="impala"
    )
    compute = sqlglot.parse_one(
        "COMPUTE INCREMENTAL STATS demo_events PARTITION (region = 'east')",
        read="impala",
    )
    refresh_partition = refresh.args["partitions"][0]
    compute_partition = compute.args["partition"]
    assert isinstance(refresh_partition, exp.Partition)
    assert isinstance(compute_partition, exp.Partition)
    assert refresh_partition.expressions[0] == static_pair(
        "region", exp.Literal.string("east")
    )
    assert compute_partition.expressions[0] == comparison(
        "region", exp.EQ, exp.Literal.string("east")
    )


@pytest.mark.parametrize(
    "source",
    [
        "COMPUTE",
        "COMPUTE STATS",
        "COMPUTE STATS TABLESAMPLE SYSTEM(17)",
        "COMPUTE STATS (account_key) TABLESAMPLE SYSTEM(17)",
        "COMPUTE INCREMENTAL STATS",
        "COMPUTE INCREMENTAL STATS PARTITION (region = 'east')",
        "COMPUTE STATS ON demo_db.events",
        "COMPUTE STATS 'demo_events'",
        "COMPUTE STATS `   `",
        "COMPUTE STATS 123",
        "COMPUTE STATS TRUE",
        "COMPUTE STATS ALL",
        "COMPUTE STATS TABLE",
        "COMPUTE STATS BLOCK_SIZE",
        "COMPUTE STATS USER_DEFINED_FN",
        "COMPUTE STATS MANAGEDLOCATION",
        "COMPUTE STATS demo_db.",
        "COMPUTE STATS a.b.c",
        "COMPUTE STATS demo_events extra",
        "COMPUTE STATS demo_events PARTITION (region = 'east')",
        "COMPUTE STATS demo_events (account_key,)",
        "COMPUTE STATS demo_events (, account_key)",
        "COMPUTE STATS demo_events (account_key event_code)",
        "COMPUTE STATS demo_events (demo_db.account_key)",
        "COMPUTE STATS demo_events REPEATABLE(2)",
        "COMPUTE STATS demo_events TABLESAMPLE BERNOULLI(12)",
        "COMPUTE STATS demo_events TABLESAMPLE `\u00a0system\u00a0`(17)",
        "COMPUTE STATS demo_events TABLESAMPLE SYSTEM()",
        "COMPUTE STATS demo_events TABLESAMPLE SYSTEM(-2)",
        "COMPUTE STATS demo_events TABLESAMPLE SYSTEM(1.5)",
        "COMPUTE STATS demo_events TABLESAMPLE SYSTEM('4')",
        "COMPUTE STATS demo_events TABLESAMPLE SYSTEM(4) REPEATABLE()",
        "COMPUTE STATS demo_events TABLESAMPLE SYSTEM(4) REPEATABLE(-3)",
        "COMPUTE INCREMENTAL STATS demo_events TABLESAMPLE SYSTEM(4)",
        "COMPUTE INCREMENTAL STATS demo_events PARTITION (region = 'east') "
        "TABLESAMPLE SYSTEM(4)",
        "COMPUTE INCREMENTAL STATS demo_events (account_key) PARTITION (region = 'east')",
        "COMPUTE INCREMENTAL STATS demo_events PARTITION ()",
        "COMPUTE INCREMENTAL STATS demo_events PARTITION (region = )",
        "COMPUTE INCREMENTAL STATS demo_events PARTITION (region = 'east',)",
        "DROP STATS",
        "DROP STATS CASE",
        "DROP STATS demo_events PARTITION (region = 'east')",
        "DROP INCREMENTAL STATS",
        "DROP INCREMENTAL STATS demo_events",
        "DROP INCREMENTAL STATS demo_events PARTITION ()",
        "DROP INCREMENTAL STATS demo_events PARTITION (region = 'east') (account_key)",
        "REFRESH",
        "REFRESH 'demo_events'",
        "REFRESH a.b.c",
        "REFRESH demo_events PARTITION ()",
        "REFRESH demo_events PARTITION (region)",
        "REFRESH demo_events PARTITION (region = )",
        "REFRESH demo_events PARTITION (region >= 2)",
        "REFRESH demo_events PARTITION (FALSE = 1)",
        "REFRESH demo_events PARTITION (region = 'east',)",
        "REFRESH demo_events PARTITION (region = 'east'), PARTITION (region = 'west')",
        "REFRESH FUNCTIONS",
        "REFRESH FUNCTIONS COMMENT",
        "REFRESH FUNCTIONS demo_db.extra",
        "REFRESH FUNCTIONS demo_db PARTITION (region = 'east')",
        "REFRESH AUTHORIZATION demo_db",
        "INVALIDATE",
        "INVALIDATE METADATA 'demo_events'",
        "INVALIDATE METADATA DELETE",
        "INVALIDATE METADATA a.b.c",
        "INVALIDATE METADATA PARTITION (region = 'east')",
        "INVALIDATE METADATA demo_events PARTITION (region = 'east')",
    ],
)
def test_invalid_metadata_statements_are_not_opaque_commands(source):
    with pytest.raises(ParseError):
        sqlglot.parse_one(source, read="impala")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "COMPUTE STATS `PARTITION` (`TABLESAMPLE`)",
            "COMPUTE STATS `PARTITION` (`TABLESAMPLE`)",
        ),
        (
            "REFRESH FUNCTIONS `AUTHORIZATION`",
            "REFRESH FUNCTIONS `AUTHORIZATION`",
        ),
        (
            "INVALIDATE METADATA `PARTITION`",
            "INVALIDATE METADATA `PARTITION`",
        ),
        (
            "COMPUTE INCREMENTAL STATS demo_events "
            "PARTITION (region IN ('east', 'west'))",
            "COMPUTE INCREMENTAL STATS demo_events "
            "PARTITION (region IN ('east', 'west'))",
        ),
    ],
)
def test_adjacent_valid_metadata_forms(source, expected):
    tree = sqlglot.parse_one(source, read="impala")
    assert not list(tree.find_all(exp.Command))
    assert tree.sql(dialect="impala") == expected
    assert sqlglot.parse_one(expected, read="impala") == tree


@pytest.mark.parametrize(
    "name", ["default", "kill", "query", "uuid", "variant", "geometry", "year", "system"]
)
@pytest.mark.parametrize(
    "template",
    [
        "COMPUTE STATS {}",
        "REFRESH FUNCTIONS {}",
        "REFRESH demo_events PARTITION ({} = 1)",
    ],
)
def test_unreserved_identifier_controls(name, template):
    source = template.format(name)
    tree = sqlglot.parse_one(source, read="impala")
    assert tree.sql(dialect="impala") == source
    assert sqlglot.parse_one(source, read="impala") == tree


@pytest.mark.parametrize(
    "name", ["blocksize", "udf", "managed_location", "storage_handler_uri", "logical_or"]
)
def test_scanner_symbol_names_are_not_mistaken_for_keyword_spellings(name):
    source = f"COMPUTE STATS {name}"
    tree = sqlglot.parse_one(source, read="impala")
    assert tree.sql(dialect="impala") == source
    assert sqlglot.parse_one(source, read="impala") == tree


@pytest.mark.parametrize(
    ("source", "expected_type"),
    [
        ("SELECT 1 AS example_value", exp.Select),
        ("DROP TABLE IF EXISTS demo_events", exp.Drop),
        ("COPY TESTCASE FROM '/synthetic/location'", exp.Command),
    ],
)
def test_unrelated_statement_paths_remain_available(source, expected_type):
    tree = sqlglot.parse_one(source, read="impala")
    assert isinstance(tree, expected_type)


def test_multiple_statements_and_comments_keep_structured_results():
    trees = sqlglot.parse(
        "SELECT 1; COMPUTE /* synthetic comment */ STATS demo_events; "
        "INVALIDATE METADATA; REFRESH demo_events; DROP STATS demo_events",
        read="impala",
    )
    assert [type(tree) for tree in trees] == [
        exp.Select,
        ImpalaComputeStats,
        ImpalaInvalidateMetadata,
        ImpalaRefresh,
        ImpalaDropStats,
    ]
    assert "synthetic comment" in trees[1].sql(dialect="impala")
    assert all(not list(tree.find_all(exp.Command)) for tree in trees)


@pytest.mark.parametrize(
    ("source", "markers"),
    [
        (
            "COMPUTE STATS /* after_stats */ demo_db. /* after_dot */ events "
            "( /* after_open */ account_key, /* after_comma */ event_code) "
            "TABLESAMPLE /* after_sample */ SYSTEM(17)",
            [
                "after_stats",
                "after_dot",
                "after_open",
                "after_comma",
                "after_sample",
            ],
        ),
        (
            "COMPUTE INCREMENTAL /* after_incremental */ STATS demo_events "
            "PARTITION /* after_partition */ (region = 'east')",
            ["after_incremental", "after_partition"],
        ),
        (
            "REFRESH FUNCTIONS /* after_functions */ demo_db",
            ["after_functions"],
        ),
        (
            "REFRESH demo_events PARTITION "
            "(region /* after_key */ = /* after_equals */ 'east')",
            ["after_key", "after_equals"],
        ),
        (
            "INVALIDATE /* after_invalidate */ METADATA demo_events",
            ["after_invalidate"],
        ),
    ],
)
def test_metadata_comments_survive_generation_and_reparse(source, markers):
    tree = sqlglot.parse_one(source, read="impala")
    generated = tree.sql(dialect="impala")
    for marker in markers:
        assert generated.count(marker) == 1
    reparsed = sqlglot.parse_one(generated, read="impala")
    assert reparsed == tree
    regenerated = reparsed.sql(dialect="impala")
    for marker in markers:
        assert regenerated.count(marker) == 1


@pytest.mark.parametrize(
    "tree",
    [
        ImpalaComputeStats(
            this=table("demo_events"),
            incremental=False,
            partition=partition(comparison("region", exp.EQ, exp.Literal.string("east"))),
        ),
        ImpalaComputeStats(
            this=table("demo_events"),
            incremental=True,
            sample=exp.TableSample(
                method=exp.Var(this="SYSTEM"), percent=exp.Literal.number(12)
            ),
        ),
        ImpalaComputeStats(
            this=table("demo_events"),
            sample=exp.TableSample(method=exp.Var(this="SYSTEM")),
        ),
        ImpalaComputeStats(this=table("all")),
        ImpalaComputeStats(this=table(" padded ", quoted=True)),
        ImpalaComputeStats(
            this=table("demo_events"),
            sample=exp.TableSample(
                method=exp.Var(this="SYSTEM"),
                percent=exp.Literal.number(12),
                size=exp.Literal.number(10),
            ),
        ),
        ImpalaDropStats(this=table("demo_events"), incremental=True),
        ImpalaDropStats(
            this=table("demo_events"),
            incremental=False,
            partition=partition(comparison("region", exp.EQ, exp.Literal.string("east"))),
        ),
        ImpalaRefresh(this=table("demo_events"), kind="FUNCTIONS"),
        ImpalaRefresh(
            this=table("demo_events"),
            kind="TABLE",
            partitions=[partition(comparison("region", exp.GTE, exp.Literal.number(2)))],
        ),
        ImpalaInvalidateMetadata(this=exp.Literal.string("demo_events")),
    ],
)
def test_generator_rejects_invalid_programmatic_combinations(tree):
    with pytest.raises(ValueError):
        tree.sql(dialect="impala")
