"""Synthetic tests for the bounded native Impala SHOW grammar."""
from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError

from impala_sqlglot.impala import Impala
from impala_sqlglot.show import ImpalaShow


DIALECT = Impala()


def parse(source: str) -> exp.Expression:
    return DIALECT.parse(source)[0]


def sql(tree: exp.Expression) -> str:
    return DIALECT.generate(tree)


def identifier(name: str, quoted: bool = False) -> exp.Identifier:
    return exp.Identifier(this=name, quoted=quoted)


def table(name: str, db: str | None = None, quoted: bool = False) -> exp.Table:
    return exp.Table(
        this=identifier(name, quoted),
        db=identifier(db, quoted) if db is not None else None,
    )


@pytest.mark.parametrize(
    ("source", "expected", "generated"),
    [
        ("show tables", ImpalaShow(kind="TABLES"), "SHOW TABLES"),
        (
            "SHOW TABLES IN demo_db",
            ImpalaShow(kind="TABLES", db=identifier("demo_db")),
            "SHOW TABLES IN demo_db",
        ),
        (
            "SHOW TABLES 'event*'",
            ImpalaShow(kind="TABLES", pattern=exp.Literal.string("event*")),
            "SHOW TABLES LIKE 'event*'",
        ),
        (
            "SHOW TABLES IN demo_db LIKE 'ev*|log*'",
            ImpalaShow(
                kind="TABLES", db=identifier("demo_db"),
                pattern=exp.Literal.string("ev*|log*"),
            ),
            "SHOW TABLES IN demo_db LIKE 'ev*|log*'",
        ),
        (
            "SHOW TABLES IN ` demo-db ` 'foo'",
            ImpalaShow(
                kind="TABLES", db=identifier("demo-db", True),
                pattern=exp.Literal.string("foo"),
            ),
            "SHOW TABLES IN `demo-db` LIKE 'foo'",
        ),
        ("SHOW DATABASES", ImpalaShow(kind="DATABASES"), "SHOW DATABASES"),
        ("SHOW SCHEMAS", ImpalaShow(kind="DATABASES"), "SHOW DATABASES"),
        (
            "SHOW SCHEMAS 'sample*'",
            ImpalaShow(kind="DATABASES", pattern=exp.Literal.string("sample*")),
            "SHOW DATABASES LIKE 'sample*'",
        ),
        (
            "SHOW DATABASES LIKE ''",
            ImpalaShow(kind="DATABASES", pattern=exp.Literal.string("")),
            "SHOW DATABASES LIKE ''",
        ),
        (
            "SHOW CREATE TABLE demo_events",
            ImpalaShow(kind="CREATE TABLE", this=table("demo_events")),
            "SHOW CREATE TABLE demo_events",
        ),
        (
            "SHOW CREATE TABLE ` demo-db `.` event-table `",
            ImpalaShow(kind="CREATE TABLE", this=table("event-table", "demo-db", True)),
            "SHOW CREATE TABLE `demo-db`.`event-table`",
        ),
        (
            "SHOW CREATE TABLE default.`TABLE`",
            ImpalaShow(
                kind="CREATE TABLE",
                this=exp.Table(this=identifier("TABLE", True), db=identifier("default")),
            ),
            "SHOW CREATE TABLE default.`TABLE`",
        ),
        ("SHOW TABLE STATS demo_db.demo_events", ImpalaShow(kind="TABLE STATS", this=table("demo_events", "demo_db")), "SHOW TABLE STATS demo_db.demo_events"),
        ("SHOW COLUMN STATS demo_events", ImpalaShow(kind="COLUMN STATS", this=table("demo_events")), "SHOW COLUMN STATS demo_events"),
        ("SHOW PARTITIONS demo_events", ImpalaShow(kind="PARTITIONS", this=table("demo_events")), "SHOW PARTITIONS demo_events"),
        ("SHOW RANGE PARTITIONS demo_events", ImpalaShow(kind="RANGE PARTITIONS", this=table("demo_events")), "SHOW RANGE PARTITIONS demo_events"),
        ("SHOW HASH SCHEMA demo_events", ImpalaShow(kind="HASH SCHEMA", this=table("demo_events")), "SHOW HASH SCHEMA demo_events"),
        ("SHOW FUNCTIONS", ImpalaShow(kind="FUNCTIONS"), "SHOW FUNCTIONS"),
        ("SHOW AGGREGATE FUNCTIONS IN demo_db 'f*'", ImpalaShow(kind="FUNCTIONS", function_kind="AGGREGATE", db=identifier("demo_db"), pattern=exp.Literal.string("f*")), "SHOW AGGREGATE FUNCTIONS IN demo_db LIKE 'f*'"),
        ("SHOW ANALYTIC FUNCTIONS LIKE 'x*'", ImpalaShow(kind="FUNCTIONS", function_kind="ANALYTIC", pattern=exp.Literal.string("x*")), "SHOW ANALYTIC FUNCTIONS LIKE 'x*'"),
        ("SHOW CREATE VIEW demo_view", ImpalaShow(kind="CREATE VIEW", this=table("demo_view")), "SHOW CREATE VIEW demo_view"),
        ("SHOW CREATE TABLE demo_events WITH STATS", ImpalaShow(kind="CREATE TABLE", this=table("demo_events"), with_stats=True), "SHOW CREATE TABLE demo_events WITH STATS"),
        ("SHOW CREATE VIEW demo_view WITH STATS", ImpalaShow(kind="CREATE VIEW", this=table("demo_view"), with_stats=True), "SHOW CREATE VIEW demo_view WITH STATS"),
        ("SHOW FILES IN demo_events", ImpalaShow(kind="FILES", this=table("demo_events")), "SHOW FILES IN demo_events"),
        ("SHOW PARTITIONS demo_events WHERE year >= 2020", ImpalaShow(kind="PARTITIONS", this=table("demo_events"), where=exp.GTE(this=exp.column("year"), expression=exp.Literal.number(2020))), "SHOW PARTITIONS demo_events WHERE year >= 2020"),
        ("SHOW FILES IN demo_events PARTITION (year = 2020, month > 2)", ImpalaShow(kind="FILES", this=table("demo_events"), partition=exp.Partition(expressions=[exp.EQ(this=exp.column("year"), expression=exp.Literal.number(2020)), exp.GT(this=exp.column("month"), expression=exp.Literal.number(2))])), "SHOW FILES IN demo_events PARTITION (year = 2020, month > 2)"),
        ("SHOW FILES IN demo_events FOR SYSTEM_TIME AS OF '2024-01-01'", ImpalaShow(kind="FILES", this=table("demo_events"), version=exp.Version(this="TIMESTAMP", kind="AS OF", expression=exp.Literal.string("2024-01-01"))), "SHOW FILES IN demo_events FOR SYSTEM_TIME AS OF '2024-01-01'"),
        ("SHOW FILES IN demo_events FOR SYSTEM_VERSION AS OF 7 PARTITION (year = 2020)", ImpalaShow(kind="FILES", this=table("demo_events"), version=exp.Version(this="VERSION", kind="AS OF", expression=exp.Literal.number(7)), partition=exp.Partition(expressions=[exp.EQ(this=exp.column("year"), expression=exp.Literal.number(2020))])), "SHOW FILES IN demo_events FOR SYSTEM_VERSION AS OF 7 PARTITION (year = 2020)"),
    ],
)
def test_structure_canonical_generation_and_reparse(source, expected, generated):
    tree = parse(source)
    assert tree == expected
    assert not list(tree.find_all(exp.Command))
    assert sql(tree) == generated
    assert parse(generated) == tree


def test_expression_children_are_traversable():
    tree = parse("SHOW CREATE TABLE demo_db.demo_events")
    assert list(tree.find_all(exp.Table)) == [tree.this]
    tree = parse("SHOW TABLES IN demo_db 'x*'")
    assert list(tree.find_all(exp.Identifier)) == [tree.args["db"]]
    assert list(tree.find_all(exp.Literal)) == [tree.args["pattern"]]
    tree = parse("SHOW FILES IN demo_events FOR SYSTEM_TIME AS OF '2024-01-01' PARTITION (year = 2020)")
    assert list(tree.find_all(exp.Version)) == [tree.args["version"]]
    assert list(tree.find_all(exp.Partition)) == [tree.args["partition"]]
    assert len(list(tree.find_all(exp.Literal))) == 2
    tree = parse("SHOW PARTITIONS demo_events WHERE year >= 2020")
    assert list(tree.find_all(exp.GTE)) == [tree.args["where"]]


@pytest.mark.parametrize(
    ("source", "omitted"),
    [
        ("SHOW FILES IN t", ("version", "partition")),
        ("SHOW PARTITIONS t", ("where",)),
        ("SHOW FUNCTIONS", ("function_kind", "db", "pattern")),
        ("SHOW CREATE VIEW t", ("with_stats",)),
    ],
)
def test_absent_options_remain_absent(source, omitted):
    tree = parse(source)
    assert all(tree.args.get(key) is None for key in omitted)
    assert parse(sql(tree)) == tree


@pytest.mark.parametrize(
    "source",
    [
        "SHOW",
        "SHOW CREATE",
        "SHOW TABLES IN",
        "SHOW TABLES IN 'demo_db'",
        "SHOW TABLES IN `  `",
        "SHOW TABLES IN ALL",
        "SHOW TABLES IN demo_db.other",
        "SHOW TABLES FROM demo_db",
        "SHOW TABLES LIKE",
        "SHOW TABLES LIKE 42",
        "SHOW TABLES LIKE `pattern`",
        "SHOW TABLES 'a' 'b'",
        "SHOW TABLES LIKE 'a' IN demo_db",
        "SHOW DATABASES IN demo_db",
        "SHOW SCHEMAS LIKE",
        "SHOW DATABASES 7",
        "SHOW CREATE TABLE",
        "SHOW CREATE TABLE 'demo_events'",
        "SHOW CREATE TABLE ALL",
        "SHOW CREATE TABLE demo_db.",
        "SHOW CREATE TABLE a.b.c",
        "SHOW CREATE TABLE demo_events extra",
        "SHOW CREATE TABLE demo_events WITH",
        "SHOW CREATE TABLE demo_events WITH STATS extra",
        "SHOW CREATE VIEW", "SHOW CREATE VIEW t WITH", "SHOW CREATE VIEW t WITH STATS extra",
        "SHOW TABLE", "SHOW TABLE STATS", "SHOW COLUMN STATS", "SHOW RANGE",
        "SHOW RANGE PARTITIONS", "SHOW HASH", "SHOW HASH SCHEMA", "SHOW PARTITIONS",
        "SHOW PARTITIONS t WHERE", "SHOW PARTITIONS t WHERE x =",
        "SHOW TABLE STATS t WHERE x = 1", "SHOW HASH SCHEMA t PARTITION (x = 1)",
        "SHOW AGGREGATE", "SHOW ANALYTIC FUNCTIONS IN", "SHOW FUNCTIONS LIKE",
        "SHOW FUNCTIONS IN 'db'", "SHOW FUNCTIONS 'x' IN db", "SHOW SCALAR FUNCTIONS",
        "SHOW FILES", "SHOW FILES t", "SHOW FILES IN", "SHOW FILES IN t PARTITION ()",
        "SHOW FILES IN t PARTITION (x =)", "SHOW FILES IN t PARTITION (x = 1,)",
        "SHOW FILES IN t FOR", "SHOW FILES IN t FOR SYSTEM_TIME AS OF",
        "SHOW FILES IN t FOR SYSTEM_VERSION AS OF '1'",
        "SHOW FILES IN t FOR SYSTEM_VERSION AS OF -1",
        "SHOW FILES IN t FOR SYSTEM_VERSION AS OF 1 + 2",
        "SHOW FILES IN t PARTITION (x = 1) FOR SYSTEM_VERSION AS OF 1",
    ],
)
@pytest.mark.parametrize("error_level", [ErrorLevel.RAISE, ErrorLevel.IMMEDIATE])
def test_invalid_supported_forms_raise(source, error_level):
    with pytest.raises(ParseError):
        DIALECT.parse(source, error_level=error_level)


@pytest.mark.parametrize(
    "source",
    [
        "SHOW VIEWS IN demo_db LIKE 'a*'",
        "SHOW CREATE FUNCTION demo_db.demo_fn",
        "SHOW CREATE AGGREGATE FUNCTION demo_db.demo_fn",
        "SHOW METADATA TABLES IN demo_db.demo_events",
        "SHOW DATA SOURCES",
    ],
)
def test_other_show_families_remain_commands(source):
    tree = parse(source)
    assert isinstance(tree, exp.Command)
    assert sql(tree).upper() == source.upper()
    assert isinstance(parse(sql(tree)), exp.Command)


@pytest.mark.parametrize(
    "source",
    [
        "/* before */ SHOW /* after_show */ VIEWS IN demo_db /* after_name */",
        "/* before */ SHOW /* after_show */ CREATE FUNCTION "
        "demo_fn /* after_name */",
    ],
)
def test_opaque_commands_preserve_comments(source):
    tree = parse(source)
    assert isinstance(tree, exp.Command)
    generated = sql(tree)
    for marker in ("before", "after_show"):
        assert generated.count(marker) == 1
    final_marker = "after_stats" if "after_stats" in source else "after_name"
    assert generated.count(final_marker) == 1
    assert isinstance(parse(generated), exp.Command)


@pytest.mark.parametrize("pattern", ["O'Reilly", r"foo\bar", "a\nb"])
def test_pattern_literal_generation_round_trips(pattern):
    tree = ImpalaShow(kind="TABLES", pattern=exp.Literal.string(pattern))
    assert parse(sql(tree)) == tree


@pytest.mark.parametrize(
    ("source", "markers"),
    [
        (
            "SHOW /* after_show */ TABLES /* after_tables */ IN /* after_in */ "
            "`demo-db` /* after_db */ LIKE /* after_like */ 'x*' /* after_pattern */",
            ["after_show", "after_tables", "after_in", "after_db", "after_like", "after_pattern"],
        ),
        (
            "SHOW /* after_show */ CREATE /* after_create */ TABLE /* after_table */ "
            "demo_db /* after_db */ . /* after_dot */ demo_events /* after_name */",
            ["after_show", "after_create", "after_table", "after_db", "after_dot", "after_name"],
        ),
        (
            "SHOW /* show */ FILES /* files */ IN /* in */ demo_events /* name */ "
            "FOR /* for */ SYSTEM_VERSION /* system */ AS /* as */ OF /* of */ 7 "
            "PARTITION /* partition */ (year = 2020)",
            ["show", "files", "in", "name", "for", "system", "as", "of", "partition"],
        ),
        (
            "SHOW /* a1 */ PARTITIONS /* a2 */ t /* a3 */ WHERE /* a4 */ "
            "year /* a5 */ = /* a6 */ 2020 /* a7 */",
            ["a1", "a2", "a3", "a4", "a5", "a6", "a7"],
        ),
        (
            "SHOW /* b1 */ ANALYTIC /* b2 */ FUNCTIONS /* b3 */ IN /* b4 */ "
            "demo_db /* b5 */ LIKE /* b6 */ 'x*' /* b7 */",
            ["b1", "b2", "b3", "b4", "b5", "b6", "b7"],
        ),
        (
            "SHOW /* c1 */ CREATE /* c2 */ VIEW /* c3 */ demo_view /* c4 */ "
            "WITH /* c5 */ STATS /* c6 */",
            ["c1", "c2", "c3", "c4", "c5", "c6"],
        ),
    ],
)
def test_comments_survive_generation_and_reparse(source, markers):
    tree = parse(source)
    generated = sql(tree)
    for marker in markers:
        assert generated.count(marker) == 1
    reparsed = parse(generated)
    assert reparsed == tree
    regenerated = sql(reparsed)
    for marker in markers:
        assert regenerated.count(marker) == 1


def test_multiple_statements():
    trees = DIALECT.parse(
        "SELECT 1; SHOW TABLES; SHOW FUNCTIONS; SHOW SCHEMAS 'x*'; "
        "SHOW CREATE TABLE demo_db.demo_events WITH STATS; "
        "SHOW PARTITIONS t WHERE x = 1; SHOW FILES IN t PARTITION (x = 1); "
        "SHOW CREATE FUNCTION demo_fn"
    )
    assert [type(tree) for tree in trees] == [
        exp.Select, ImpalaShow, ImpalaShow, ImpalaShow, ImpalaShow,
        ImpalaShow, ImpalaShow, exp.Command,
    ]
    assert trees[4].args["with_stats"] is True
    for tree in trees[1:-1]:
        assert parse(sql(tree)) == tree


@pytest.mark.parametrize(
    "tree",
    [
        ImpalaShow(kind="UNKNOWN"),
        ImpalaShow(kind="TABLES", this=table("demo_events")),
        ImpalaShow(kind="DATABASES", db=identifier("demo_db")),
        ImpalaShow(kind="TABLES", db=exp.Literal.string("demo_db")),
        ImpalaShow(kind="TABLES", db=identifier("all")),
        ImpalaShow(kind="TABLES", db=identifier(" padded ", True)),
        ImpalaShow(kind="TABLES", pattern=exp.Literal.number(1)),
        ImpalaShow(kind="TABLES", pattern=exp.Identifier(this="x")),
        ImpalaShow(kind="CREATE TABLE"),
        ImpalaShow(kind="CREATE TABLE", this=table("demo_events"), pattern=exp.Literal.string("x")),
        ImpalaShow(kind="CREATE TABLE", this=table("demo_events"), db=identifier("demo_db")),
        ImpalaShow(kind="CREATE TABLE", this=table("all")),
        ImpalaShow(kind="TABLES", unsupported=True),
        ImpalaShow(kind="TABLES", unsupported=None),
        ImpalaShow(kind="FUNCTIONS", function_kind="SCALAR"),
        ImpalaShow(kind="FUNCTIONS", function_kind=True),
        ImpalaShow(kind="TABLES", function_kind="ANALYTIC"),
        ImpalaShow(kind="FILES"),
        ImpalaShow(kind="TABLE STATS", this=table("demo"), where=exp.true()),
        ImpalaShow(kind="CREATE VIEW", this=table("demo"), with_stats=False),
        ImpalaShow(kind="PARTITIONS", this=table("demo"), where="x"),
        ImpalaShow(kind="PARTITIONS", this=table("demo"), where=exp.Where(this=exp.true())),
        ImpalaShow(kind="FILES", this=table("demo"), partition=exp.Partition(expressions=[])),
        ImpalaShow(kind="FILES", this=table("demo"), partition=exp.Partition(expressions=["x"])),
        ImpalaShow(kind="FILES", this=table("demo"), partition=exp.Partition(expressions=[exp.select("x")])),
        ImpalaShow(kind="FILES", this=table("demo"), version=exp.Version(this="VERSION", kind="AS OF", expression=exp.Literal.string("1"))),
        ImpalaShow(kind="FILES", this=table("demo"), version=exp.Version(this="VERSION", kind="AS OF", expression=exp.Literal.number(-1))),
        ImpalaShow(kind="FILES", this=table("demo"), version=exp.Version(this="VERSION", kind="AS OF", expression=exp.Literal.number(1), unsupported=None)),
        ImpalaShow(kind="TABLES", db=exp.Identifier(this="demo", quoted=False, unsupported=None)),
        ImpalaShow(kind="TABLES", pattern=exp.Literal(this="x", is_string=True, unsupported=None)),
        ImpalaShow(kind="TABLE STATS", this=exp.Table(this=identifier("demo"), unsupported=None)),
        ImpalaShow(kind="FILES", this=table("demo"), partition=exp.Partition(expressions=[exp.true()], unsupported=None)),
        ImpalaShow(kind="FILES", this=table("demo"), version=exp.Version(this="VERSION", kind="AS OF", expression=exp.Literal(this="nonsense", is_string=False))),
        ImpalaShow(kind="PARTITIONS", this=table("demo"), where=exp.EQ(this=exp.column("x"), expression=exp.Literal.number(1), unsupported=None)),
    ],
)
def test_generator_rejects_invalid_programmatic_nodes(tree):
    with pytest.raises(ValueError):
        sql(tree)


@pytest.mark.parametrize(
    "name",
    [
        "db.tbl", "hello world", " leading", "trailing ", "trailing\n",
        "name--comment", "name/*comment*/", "name;", "name+other",
        "name`", "'name'", "42", "",
    ],
)
@pytest.mark.parametrize("position", ["database", "table", "table_database"])
def test_generator_rejects_non_identifier_unquoted_names(name, position):
    bad = identifier(name)
    if position == "database":
        tree = ImpalaShow(kind="FUNCTIONS", db=bad)
    elif position == "table":
        tree = ImpalaShow(kind="TABLE STATS", this=exp.Table(this=bad))
    else:
        tree = ImpalaShow(kind="FILES", this=exp.Table(this=identifier("t"), db=bad))
    with pytest.raises(ValueError):
        sql(tree)


@pytest.mark.parametrize(
    ("name", "quoted"),
    [
        ("simple_name", False), ("1a", False), ("default", False),
        ("kill", False), ("query", False), ("uuid", False),
        ("variant", False), ("geometry", False),
        ("db.tbl", True), ("name with spaces", True), ("a-b", True),
        ("name--comment", True), ("a`b", True), ("42", True),
    ],
)
@pytest.mark.parametrize("position", ["database", "table", "table_database"])
def test_generator_identifier_controls_round_trip(name, quoted, position):
    value = identifier(name, quoted)
    if position == "database":
        tree = ImpalaShow(kind="FUNCTIONS", db=value)
    elif position == "table":
        tree = ImpalaShow(kind="TABLE STATS", this=exp.Table(this=value))
    else:
        tree = ImpalaShow(kind="FILES", this=exp.Table(this=identifier("t"), db=value))
    assert parse(sql(tree)) == tree


@pytest.mark.parametrize(
    "value",
    [
        exp.Identifier(this="k"),
        exp.Paren(this=exp.Limit(expression=exp.Literal.number(1))),
        exp.EQ(this=exp.column("x"), expression=exp.Limit(expression=exp.Literal.number(1))),
        exp.Paren(this=exp.Offset(expression=exp.Literal.number(1))),
        exp.alias_(exp.column("x"), "y"),
        exp.Paren(this=exp.alias_(exp.column("x"), "y")),
        exp.EQ(this=exp.column("x"), expression=exp.alias_(exp.column("y"), "z")),
        exp.EQ(this=exp.column("x"), expression=table("other")),
        exp.Anonymous(this="f", expressions=[table("other")]),
        exp.Subquery(this=exp.Subquery(this=exp.alias_(exp.column("x"), "y"))),
    ],
)
@pytest.mark.parametrize("position", ["where", "partition", "system_time"])
def test_generator_rejects_non_value_wrappers(value, position):
    if position == "where":
        tree = ImpalaShow(kind="PARTITIONS", this=table("t"), where=value)
    elif position == "partition":
        tree = ImpalaShow(kind="FILES", this=table("t"), partition=exp.Partition(expressions=[value]))
    else:
        tree = ImpalaShow(
            kind="FILES", this=table("t"),
            version=exp.Version(this="TIMESTAMP", kind="AS OF", expression=value),
        )
    with pytest.raises(ValueError):
        sql(tree)


@pytest.mark.parametrize(
    "source",
    [
        "SHOW PARTITIONS t WHERE k",
        "SHOW PARTITIONS t WHERE CAST(k AS INT)",
        "SHOW FILES IN t PARTITION (k)",
        "SHOW FILES IN t PARTITION (CAST(k AS INT))",
        "SHOW FILES IN t FOR SYSTEM_TIME AS OF k",
        "SHOW FILES IN t FOR SYSTEM_TIME AS OF CAST(k AS INT)",
    ],
)
def test_value_column_and_cast_identifiers_remain_valid(source):
    tree = parse(source)
    assert parse(sql(tree)) == tree


@pytest.mark.parametrize(
    "source",
    [
        "SHOW PARTITIONS t WHERE EXISTS (SELECT 1 AS x FROM demo AS d)",
        "SHOW PARTITIONS t WHERE x IN (SELECT y FROM demo LIMIT 1)",
        "SHOW PARTITIONS t WHERE x IN (SELECT y FROM demo AS d)",
        "SHOW PARTITIONS t WHERE x = (SELECT MAX(y) FROM demo)",
        "SHOW FILES IN t PARTITION (x IN (SELECT y FROM demo AS d))",
        "SHOW FILES IN t FOR SYSTEM_TIME AS OF (SELECT MAX(y) FROM demo)",
    ],
)
def test_value_subqueries_allow_native_query_aliases_and_tables(source):
    tree = sqlglot.parse_one(source, read="impala")
    assert isinstance(tree, ImpalaShow)
    assert sqlglot.parse_one(tree.sql(dialect="impala"), read="impala") == tree


def test_public_impala_dialect_is_wired():
    """Exercise the registered dialect, not a test-only parser adapter."""
    tree = sqlglot.parse_one("SHOW TABLES IN demo_db LIKE 'x*'", read="impala")
    assert isinstance(tree, ImpalaShow)
    assert tree.sql(dialect="impala") == "SHOW TABLES IN demo_db LIKE 'x*'"
    assert sqlglot.parse_one(tree.sql(dialect="impala"), read="impala") == tree
    assert isinstance(sqlglot.parse_one("SHOW TABLE STATS demo_events", read="impala"), ImpalaShow)


def test_programmatic_partition_comments_are_rendered():
    partition = exp.Partition(expressions=[exp.EQ(this=exp.column("x"), expression=exp.Literal.number(1))])
    partition.add_comments(["partition_note"])
    generated = sql(ImpalaShow(kind="FILES", this=table("demo"), partition=partition))
    assert generated.count("partition_note") == 1
    assert isinstance(parse(generated), ImpalaShow)


@pytest.mark.parametrize(
    "source",
    [
        "SHOW TABLE STATS db.t", "SHOW COLUMN STATS db.t",
        "SHOW PARTITIONS db.t WHERE x > 1", "SHOW RANGE PARTITIONS db.t",
        "SHOW HASH SCHEMA db.t", "SHOW ANALYTIC FUNCTIONS IN db 'x*'",
        "SHOW CREATE VIEW db.v WITH STATS",
        "SHOW FILES IN db.t FOR SYSTEM_TIME AS OF '2020-01-01' PARTITION (x = 1)",
        "SHOW FILES IN db.t FOR SYSTEM_VERSION AS OF 1",
    ],
)
def test_public_dialect_new_families(source):
    tree = sqlglot.parse_one(source, read="impala")
    assert isinstance(tree, ImpalaShow)
    assert sqlglot.parse_one(tree.sql(dialect="impala"), read="impala") == tree
