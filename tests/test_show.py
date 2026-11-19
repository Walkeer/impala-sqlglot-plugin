"""Synthetic tests for the bounded native Impala SHOW grammar."""
from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

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
    ],
)
def test_invalid_supported_forms_raise(source):
    with pytest.raises(ParseError):
        parse(source)


@pytest.mark.parametrize(
    "source",
    [
        "SHOW VIEWS IN demo_db LIKE 'a*'",
        "SHOW TABLE STATS demo_db.demo_events",
        "SHOW COLUMN STATS demo_db.demo_events",
        "SHOW PARTITIONS demo_db.demo_events",
        "SHOW FUNCTIONS IN demo_db",
        "SHOW CREATE VIEW demo_db.demo_view",
        "SHOW CREATE FUNCTION demo_db.demo_fn",
        "SHOW CREATE TABLE demo_db.demo_events WITH STATS",
        "SHOW METADATA TABLES IN demo_db.demo_events",
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
        "/* before */ SHOW /* after_show */ TABLE STATS demo_events /* after_name */",
        "/* before */ SHOW /* after_show */ CREATE /* after_create */ TABLE "
        "demo_events WITH STATS /* after_stats */",
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
        "SHOW CREATE TABLE demo_db.demo_events"
    )
    assert [type(tree) for tree in trees] == [
        exp.Select, ImpalaShow, exp.Command, ImpalaShow, ImpalaShow
    ]


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
    ],
)
def test_generator_rejects_invalid_programmatic_nodes(tree):
    with pytest.raises(ValueError):
        sql(tree)


def test_public_impala_dialect_is_wired():
    """Exercise the registered dialect, not a test-only parser adapter."""
    tree = sqlglot.parse_one("SHOW TABLES IN demo_db LIKE 'x*'", read="impala")
    assert isinstance(tree, ImpalaShow)
    assert tree.sql(dialect="impala") == "SHOW TABLES IN demo_db LIKE 'x*'"
    assert sqlglot.parse_one(tree.sql(dialect="impala"), read="impala") == tree
    assert isinstance(sqlglot.parse_one("SHOW TABLE STATS demo_events", read="impala"), exp.Command)
