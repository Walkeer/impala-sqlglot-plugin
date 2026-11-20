"""Synthetic tests for Impala ALTER TABLE maintenance actions."""

from __future__ import annotations

import builtins
import socket
from pathlib import Path

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError

from impala_sqlglot.ddl import ImpalaDDLParserMixin, KuduAlterRangePartition
from impala_sqlglot.impala import Impala
from impala_sqlglot.maintenance import (
    ImpalaDropPartition,
    ImpalaRecoverPartitions,
    ImpalaSetCached,
    ImpalaSetUncached,
)


DIALECT = Impala()
MAINTENANCE_ACTIONS = (
    ImpalaRecoverPartitions,
    ImpalaDropPartition,
    ImpalaSetCached,
    ImpalaSetUncached,
)


def parse(sql: str, **options) -> exp.Expression:
    return sqlglot.parse_one(sql, read=DIALECT, **options)


def round_trip(tree: exp.Expression, *, pretty: bool = False) -> str:
    before = tree.copy()
    sql = tree.sql(dialect=DIALECT, pretty=pretty)
    assert tree == before
    assert parse(sql) == tree
    return sql


def action_of(tree: exp.Expression, action_type: type) -> exp.Expression:
    assert isinstance(tree, exp.Alter)
    assert tree.args["kind"] == "TABLE"
    assert isinstance(tree.this, exp.Table)
    assert len(tree.actions) == 1
    assert isinstance(tree.actions[0], action_type)
    assert tree.find(exp.Insert) is None
    assert tree.find(exp.Command) is None
    return tree.actions[0]


def alter(action: exp.Expression, target: exp.Table | None = None) -> exp.Alter:
    return exp.Alter(
        this=target if target is not None else exp.Table(this=exp.to_identifier("demo")),
        kind="TABLE",
        actions=[action],
    )


def selector() -> exp.Partition:
    return exp.Partition(
        expressions=[exp.EQ(this=exp.column("year"), expression=exp.Literal.number(2024))]
    )


@pytest.mark.parametrize("target", ["demo", "demo_db.events", "`demo-db`.`select`"])
def test_recover_partitions_has_table_target_and_distinct_action(target):
    tree = parse(f"ALTER TABLE {target} RECOVER PARTITIONS")
    action_of(tree, ImpalaRecoverPartitions)
    assert round_trip(tree) == f"ALTER TABLE {target} RECOVER PARTITIONS"
    assert parse(round_trip(tree, pretty=True)) == tree


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("purge", [False, True])
def test_drop_partition_preserves_native_flag_order_and_predicates(exists, purge):
    exists_sql = " IF EXISTS" if exists else ""
    purge_sql = " PURGE" if purge else ""
    sql = (
        f"ALTER TABLE demo_db.events DROP{exists_sql} PARTITION "
        f"(year >= 2020, month < 7, region IN ('north', 'west')){purge_sql}"
    )
    tree = parse(sql)
    action = action_of(tree, ImpalaDropPartition)
    assert action.args["exists"] is exists
    assert action.args["purge"] is purge
    assert isinstance(action.this, exp.Partition)
    assert [type(item) for item in action.this.expressions] == [exp.GTE, exp.LT, exp.In]
    assert round_trip(tree) == sql
    assert parse(round_trip(tree, pretty=True)) == tree


def test_partition_expressions_are_not_static_assignments():
    sql = (
        "ALTER TABLE demo DROP PARTITION "
        "(year BETWEEN 2020 AND 2024, region != 'north', day IS NULL, "
        "(month = 1 OR month > 9))"
    )
    tree = parse(sql)
    action = action_of(tree, ImpalaDropPartition)
    expressions = action.this.expressions
    assert [type(item) for item in expressions] == [
        exp.Between,
        exp.NEQ,
        exp.Is,
        exp.Paren,
    ]
    assert isinstance(expressions[-1].this, exp.Or)
    assert round_trip(tree) == sql.replace("!=", "<>")


@pytest.mark.parametrize(
    ("sql", "action_type", "pool", "replication", "has_partition"),
    [
        ("ALTER TABLE demo SET CACHED IN 'pool_a'", ImpalaSetCached, "pool_a", None, False),
        (
            "ALTER TABLE demo SET CACHED IN 'pool_a' WITH REPLICATION = 3",
            ImpalaSetCached,
            "pool_a",
            "3",
            False,
        ),
        ("ALTER TABLE demo SET UNCACHED", ImpalaSetUncached, None, None, False),
        (
            "ALTER TABLE demo PARTITION (year >= 2022, region = 'north') "
            "SET CACHED IN 'pool_b' WITH REPLICATION = 2",
            ImpalaSetCached,
            "pool_b",
            "2",
            True,
        ),
        (
            "ALTER TABLE demo PARTITION (year < 2022) SET UNCACHED",
            ImpalaSetUncached,
            None,
            None,
            True,
        ),
    ],
)
def test_cache_actions_preserve_pool_replication_and_optional_selector(
    sql, action_type, pool, replication, has_partition
):
    tree = parse(sql)
    action = action_of(tree, action_type)
    partition = action.args.get("partition")
    assert isinstance(partition, exp.Partition) is has_partition
    if pool is not None:
        assert isinstance(action.this, exp.Literal) and action.this.is_string
        assert action.this.this == pool
        value = action.args.get("replication")
        assert (value.this if value is not None else None) == replication
    assert round_trip(tree) == sql
    assert parse(round_trip(tree, pretty=True)) == tree


@pytest.mark.parametrize("replication", ["0", "0003", "32768", "999999999999999999999"])
def test_replication_is_a_grammar_literal_not_a_cache_capacity_check(replication):
    sql = f"ALTER TABLE demo SET CACHED IN 'pool' WITH REPLICATION = {replication}"
    action = action_of(parse(sql), ImpalaSetCached)
    assert action.args["replication"].this == replication
    assert round_trip(alter(action.copy())) == sql


def test_escaped_and_empty_pool_names_are_preserved_without_resolving_them():
    for sql in (
        "ALTER TABLE demo SET CACHED IN ''",
        "ALTER TABLE demo SET CACHED IN 'pool\\'s name'",
    ):
        tree = parse(sql)
        action = action_of(tree, ImpalaSetCached)
        assert action.this.is_string
        assert parse(round_trip(tree)) == tree


def test_multiple_statements_and_comments_are_preserved():
    sql = (
        "ALTER /* first */ TABLE demo RECOVER /* second */ PARTITIONS;"
        "ALTER TABLE demo DROP IF EXISTS PARTITION (year > 2020) PURGE"
    )
    trees = sqlglot.parse(sql, read=DIALECT)
    assert len(trees) == 2
    action_of(trees[0], ImpalaRecoverPartitions)
    action_of(trees[1], ImpalaDropPartition)
    generated = round_trip(trees[0])
    assert "first" in generated and "second" in generated
    round_trip(trees[1])


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER /* a */ TABLE /* b */ demo /* c */ DROP /* d */ IF /* e */ EXISTS "
        "/* f */ PARTITION /* g */ (year /* h */ >= /* i */ 2020 /* j */) "
        "/* k */ PURGE /* l */",
        "ALTER /* a */ TABLE /* b */ demo /* c */ PARTITION /* d */ "
        "(year /* e */ > /* f */ 2020 /* g */) /* h */ SET /* i */ CACHED /* j */ "
        "IN /* k */ 'pool' /* l */ WITH /* m */ REPLICATION /* n */ = /* o */ 2 /* p */",
    ],
)
def test_inline_comments_are_preserved_once(sql):
    generated = round_trip(parse(sql))
    for marker in "abcdefghijklmnop":
        comment = f"/* {marker} */"
        if comment in sql:
            assert generated.count(comment) == 1


@pytest.mark.parametrize("kind", ["drop", "cached", "uncached"])
@pytest.mark.parametrize("pretty", [False, True])
def test_partition_node_comments_are_emitted_once_and_survive_reparse(kind, pretty):
    partition = selector()
    partition.add_comments(["partition node"])
    if kind == "drop":
        action = ImpalaDropPartition(this=partition)
    elif kind == "cached":
        action = ImpalaSetCached(this=exp.Literal.string("pool"), partition=partition)
    else:
        action = ImpalaSetUncached(partition=partition)

    generated = round_trip(alter(action), pretty=pretty)
    assert generated.count("/* partition node */") == 1
    regenerated = round_trip(parse(generated), pretty=pretty)
    assert regenerated.count("/* partition node */") == 1


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER TABLE RECOVER PARTITIONS",
        "ALTER TABLE db. RECOVER PARTITIONS",
        "ALTER TABLE db..demo RECOVER PARTITIONS",
        "ALTER TABLE catalog.db.demo RECOVER PARTITIONS",
        "ALTER TABLE 'demo' RECOVER PARTITIONS",
        "ALTER TABLE ` ` RECOVER PARTITIONS",
        "ALTER TABLE IF EXISTS demo RECOVER PARTITIONS",
        "ALTER TABLE ONLY demo SET UNCACHED",
        "ALTER TABLE IF EXISTS demo DROP PARTITION (year = 1)",
        "ALTER TABLE demo RECOVER",
        "ALTER TABLE demo RECOVER PARTITION",
        "ALTER TABLE demo RECOVER PARTITIONS ()",
        "ALTER TABLE demo RECOVER PARTITIONS PURGE",
        "ALTER TABLE DROP PARTITION (year = 1)",
        "ALTER TABLE demo DROP PARTITION",
        "ALTER TABLE demo DROP PARTITION ()",
        "ALTER TABLE demo DROP PARTITION (, year = 1)",
        "ALTER TABLE demo DROP PARTITION (year = 1,)",
        "ALTER TABLE demo DROP PARTITION (year = 1,, month = 2)",
        "ALTER TABLE demo DROP PARTITION (year = )",
        "ALTER TABLE demo DROP PARTITION (year = 1",
        "ALTER TABLE demo DROP IF NOT EXISTS PARTITION (year = 1)",
        "ALTER TABLE demo DROP IF PARTITION (year = 1)",
        "ALTER TABLE demo DROP IF EXISTS IF EXISTS PARTITION (year = 1)",
        "ALTER TABLE demo DROP PURGE PARTITION (year = 1)",
        "ALTER TABLE demo DROP IF EXISTS PURGE PARTITION (year = 1)",
        "ALTER TABLE demo DROP PARTITION IF EXISTS (year = 1)",
        "ALTER TABLE demo DROP PARTITION (year = 1) IF EXISTS",
        "ALTER TABLE demo DROP PARTITION (year = 1) PURGE PURGE",
        "ALTER TABLE demo DROP PARTITION (year = 1) PARTITION (month = 2)",
        "ALTER TABLE SET CACHED IN 'pool'",
        "ALTER TABLE demo SET CACHED",
        "ALTER TABLE demo CACHED IN 'pool'",
        "ALTER TABLE demo SET SET CACHED IN 'pool'",
        "ALTER TABLE demo SET CACHED IN",
        "ALTER TABLE demo SET CACHED IN pool",
        "ALTER TABLE demo SET CACHED IN 2",
        "ALTER TABLE demo SET CACHED IN 'pool' WITH REPLICATION",
        "ALTER TABLE demo SET CACHED IN 'pool' WITH REPLICATION 2",
        "ALTER TABLE demo SET CACHED IN 'pool' WITH REPLICATION =",
        "ALTER TABLE demo SET CACHED IN 'pool' WITH REPLICATION = -1",
        "ALTER TABLE demo SET CACHED IN 'pool' WITH REPLICATION = +1",
        "ALTER TABLE demo SET CACHED IN 'pool' WITH REPLICATION = 1.5",
        "ALTER TABLE demo SET CACHED IN 'pool' WITH REPLICATION = 1e2",
        "ALTER TABLE demo SET CACHED IN 'pool' WITH REPLICATION = '2'",
        "ALTER TABLE demo SET CACHED IN 'pool' WITH REPLICATION = 2 "
        "WITH REPLICATION = 3",
        "ALTER TABLE demo SET CACHED IN 'pool' PURGE",
        "ALTER TABLE demo SET CACHED IN 'pool' PARTITION (year = 1)",
        "ALTER TABLE demo SET UNCACHED IN 'pool'",
        "ALTER TABLE demo SET UNCACHED WITH REPLICATION = 2",
        "ALTER TABLE demo SET UNCACHED CACHED IN 'pool'",
        "ALTER TABLE demo PARTITION () SET UNCACHED",
        "ALTER TABLE demo PARTITION (year = 1,) SET CACHED IN 'pool'",
        "ALTER TABLE demo PARTITION (year = 1 SET UNCACHED",
        "ALTER TABLE demo PARTITION year = 1 SET CACHED IN 'pool'",
        "ALTER TABLE demo PARTITION (year = 1) PARTITION (month = 2) SET UNCACHED",
        "ALTER TABLE demo PARTITION (year = 1) SET UNCACHED PURGE",
    ],
)
@pytest.mark.parametrize("level", [ErrorLevel.IMMEDIATE, ErrorLevel.RAISE])
def test_recognized_malformed_forms_raise_in_both_error_levels(sql, level):
    with pytest.raises(ParseError):
        parse(sql, error_level=level)


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER TABLE demo ADD COLUMNS (value INT)",
        "ALTER TABLE demo DROP COLUMN value",
        "ALTER VIEW demo AS SELECT 1",
        "ALTER TABLE demo SET FILEFORMAT PARQUET",
        "ALTER TABLE demo PARTITION (year = 1) SET FILEFORMAT PARQUET",
    ],
)
def test_other_alter_forms_are_not_maintenance_actions(sql):
    tree = parse(sql)
    assert not any(isinstance(node, MAINTENANCE_ACTIONS) for node in tree.walk())


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER TABLE demo ADD COLUMNS (value INT)",
        "ALTER TABLE demo DROP COLUMN value",
        "ALTER TABLE demo DROP IF EXISTS RANGE PARTITION VALUE = 1",
        "ALTER TABLE demo SET FILEFORMAT PARQUET",
        "ALTER TABLE demo SET TBLPROPERTIES ('CACHED' = 'UNCACHED')",
        "ALTER TABLE demo PARTITION (year = 1) SET FILEFORMAT PARQUET",
        "ALTER TABLE demo PARTITION (year = 1) SET FILEFORMAT CACHED",
        "ALTER TABLE demo PARTITION (year = CACHED) SET FILEFORMAT PARQUET",
        "ALTER TABLE demo UNKNOWN RECOVER PARTITIONS",
        "ALTER TABLE demo UNKNOWN DROP PARTITION (year = 1)",
        "ALTER VIEW demo RECOVER PARTITIONS",
        "ALTER TABLE demo `RECOVER` PARTITIONS",
    ],
)
def test_unsupported_alter_is_delegated_without_consuming_tokens(sql, monkeypatch):
    calls = []

    def delegated(parser):
        calls.append(parser._curr.text)
        while parser._curr:
            parser._advance()
        return exp.Command(this="ALTER", expression=" delegated")

    monkeypatch.setattr(ImpalaDDLParserMixin, "_parse_alter", delegated, raising=False)
    assert isinstance(parse(sql), exp.Command)
    assert calls == [sql.split()[1]]


def test_kudu_drop_range_still_uses_existing_cooperative_parser():
    tree = parse("ALTER TABLE demo DROP IF EXISTS RANGE PARTITION VALUE = 1")
    assert isinstance(tree, exp.Alter)
    assert isinstance(tree.actions[0], KuduAlterRangePartition)
    assert round_trip(tree) == "ALTER TABLE demo DROP IF EXISTS RANGE PARTITION VALUE = 1"


@pytest.mark.parametrize(
    "action",
    [
        ImpalaDropPartition(this=exp.Partition(expressions=[])),
        ImpalaDropPartition(this=exp.Partition(expressions=["year = 1"])),
        ImpalaDropPartition(this=selector(), exists="yes"),
        ImpalaDropPartition(this=selector(), purge=1),
        ImpalaDropPartition(this=selector(), unexpected=False),
        ImpalaSetCached(this=exp.Var(this="pool")),
        ImpalaSetCached(this=exp.Literal.number(1)),
        ImpalaSetCached(this=exp.Literal.string("pool"), replication=exp.Literal.string("1")),
        ImpalaSetCached(this=exp.Literal.string("pool"), replication=exp.Literal.number(-1)),
        ImpalaSetCached(this=exp.Literal.string("pool"), replication=exp.Literal.number("1.5")),
        ImpalaSetCached(this=exp.Literal.string("pool"), replication=exp.Literal.number("1e2")),
        ImpalaSetCached(this=exp.Literal.string("pool"), replication=exp.Literal.number("１２")),
        ImpalaSetCached(this=exp.Literal.string("pool"), replication=1),
        ImpalaSetCached(this=exp.Literal.string("pool"), replication=True),
        ImpalaSetCached(this=exp.Literal.string("pool"), partition=exp.Partition(expressions=[])),
        ImpalaSetUncached(partition=exp.Partition(expressions=[])),
        ImpalaSetUncached(replication=exp.Literal.number(2)),
        ImpalaRecoverPartitions(unexpected=None),
    ],
)
def test_generation_rejects_invalid_or_unrenderable_action_fields(action):
    with pytest.raises(ValueError):
        alter(action).sql(dialect=DIALECT)


@pytest.mark.parametrize("kind", ["drop", "cached", "uncached"])
@pytest.mark.parametrize(
    "item",
    [
        exp.Table(this=exp.to_identifier("demo")),
        exp.Alias(this=exp.column("year"), alias=exp.to_identifier("y")),
        exp.Identifier(this="year", quoted=False),
        exp.Var(this="year"),
        exp.Where(this=exp.column("year")),
        exp.Partition(expressions=[exp.column("year")]),
        exp.Tuple(expressions=[exp.Literal.number(1), exp.Literal.number(2)]),
        exp.Paren(this=exp.Limit(expression=exp.Literal.number(1))),
        exp.EQ(this=exp.column("year"), expression=exp.Limit(expression=exp.Literal.number(1))),
        exp.Paren(this=exp.Offset(expression=exp.Literal.number(1))),
        exp.select("year"),
        exp.Subquery(this=exp.Table(this=exp.to_identifier("demo"))),
        exp.Subquery(
            this=exp.select("year"),
            alias=exp.TableAlias(this=exp.to_identifier("q")),
        ),
        exp.Paren(this=exp.Table(this=exp.to_identifier("demo"))),
        exp.EQ(this=exp.Table(this=exp.to_identifier("demo")), expression=exp.Literal.number(1)),
        exp.Paren(this=exp.Identifier(this="year")),
        exp.Paren(this=exp.Var(this="year")),
        exp.EQ(this=exp.Identifier(this="year"), expression=exp.Literal.number(1)),
        exp.GT(this=exp.Var(this="year"), expression=exp.Literal.number(1)),
        exp.EQ(
            this=exp.Alias(this=exp.column("year"), alias=exp.to_identifier("y")),
            expression=exp.Literal.number(1),
        ),
    ],
)
def test_partition_rejects_non_value_wrappers(kind, item):
    partition = exp.Partition(expressions=[item.copy()])
    if kind == "drop":
        action = ImpalaDropPartition(this=partition)
    elif kind == "cached":
        action = ImpalaSetCached(this=exp.Literal.string("pool"), partition=partition)
    else:
        action = ImpalaSetUncached(partition=partition)
    with pytest.raises(ValueError):
        alter(action).sql(dialect=DIALECT)


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER TABLE demo DROP PARTITION (year + 1 >= 2024, LOWER(region) = 'north')",
        "ALTER TABLE demo DROP PARTITION ((SELECT 1))",
        "ALTER TABLE demo DROP PARTITION (EXISTS (SELECT year AS y FROM reference_table))",
        "ALTER TABLE demo PARTITION (year IN (SELECT year FROM reference_table)) SET UNCACHED",
        "ALTER TABLE demo PARTITION (year IN (SELECT year FROM reference_table LIMIT 1)) SET UNCACHED",
        "ALTER TABLE demo PARTITION (CAST(year AS INT) > 2) SET CACHED IN 'pool'",
        "ALTER TABLE demo DROP PARTITION (EXTRACT(YEAR FROM event_time) > 2020)",
        "ALTER TABLE demo DROP PARTITION (CAST(item AS STRUCT<a: INT>) IS NOT NULL)",
        "ALTER TABLE demo DROP PARTITION (`SAMPLE_FN`(year) > 2)",
    ],
)
def test_value_expressions_and_grammar_subqueries_still_round_trip(sql):
    round_trip(parse(sql))


@pytest.mark.parametrize(
    "target",
    [
        exp.Table(),
        exp.Table(this=exp.to_identifier("demo"), catalog=exp.to_identifier("catalog")),
        exp.Table(this=exp.to_identifier("demo"), alias=exp.TableAlias(this=exp.to_identifier("x"))),
        exp.Table(this=exp.to_identifier("demo"), unexpected=None),
        exp.Table(this=exp.Identifier(this="demo", quoted="yes")),
        exp.Table(this=exp.Identifier(this="demo", quoted=False, unexpected=False)),
        exp.Table(this=exp.Identifier(this="two words", quoted=False)),
        exp.Table(this=exp.Identifier(this="a.b", quoted=False)),
        exp.Table(this=exp.Identifier(this="123", quoted=False)),
        exp.Table(this=exp.Identifier(this="name'", quoted=False)),
        exp.Table(this=exp.Identifier(this=123, quoted=False)),
    ],
)
def test_generation_rejects_missing_or_unrenderable_targets(target):
    with pytest.raises(ValueError):
        alter(ImpalaRecoverPartitions(), target).sql(dialect=DIALECT)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("kind", "VIEW"),
        ("exists", True),
        ("only", True),
        ("cascade", True),
        ("options", [exp.Var(this="x")]),
        ("cluster", exp.Var(this="x")),
        ("unexpected", None),
        ("unexpected", False),
    ],
)
def test_generation_does_not_silently_drop_alter_fields(key, value):
    tree = alter(ImpalaRecoverPartitions())
    if value is None:
        tree.args[key] = None
    else:
        tree.set(key, value)
    with pytest.raises(ValueError):
        tree.sql(dialect=DIALECT)


def test_generation_rejects_multiple_actions_and_unknown_partition_fields():
    tree = alter(ImpalaRecoverPartitions())
    tree.set("actions", [ImpalaRecoverPartitions(), ImpalaSetUncached()])
    with pytest.raises(ValueError):
        tree.sql(dialect=DIALECT)

    tree = alter(ImpalaRecoverPartitions())
    tree.args["actions"] = (ImpalaRecoverPartitions(),)
    with pytest.raises(ValueError):
        tree.sql(dialect=DIALECT)

    tree = alter(ImpalaRecoverPartitions())
    tree.args["actions"] = ImpalaRecoverPartitions()
    with pytest.raises(ValueError):
        tree.sql(dialect=DIALECT)

    partition = selector()
    partition.args["unexpected"] = None
    with pytest.raises(ValueError):
        alter(ImpalaDropPartition(this=partition)).sql(dialect=DIALECT)

    partition = selector()
    partition.expressions[0].set("unexpected", False)
    with pytest.raises(ValueError):
        alter(ImpalaDropPartition(this=partition)).sql(dialect=DIALECT)


def test_parsing_and_generation_do_not_open_files_or_connect(monkeypatch):
    parse("ALTER TABLE demo RECOVER PARTITIONS").sql(dialect=DIALECT)

    def blocked(*args, **kwargs):
        raise AssertionError("Maintenance parsing and generation must not perform I/O")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", blocked)
        patch.setattr(Path, "open", blocked)
        patch.setattr(socket, "create_connection", blocked)
        patch.setattr(socket, "socket", blocked)
        for sql in (
            "ALTER TABLE demo RECOVER PARTITIONS",
            "ALTER TABLE demo DROP IF EXISTS PARTITION (year > 2020) PURGE",
            "ALTER TABLE demo SET CACHED IN 'pool' WITH REPLICATION = 2",
            "ALTER TABLE demo SET UNCACHED",
        ):
            round_trip(parse(sql))
