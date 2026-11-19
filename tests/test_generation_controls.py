"""Synthetic controls for the narrow Impala generation compatibility fixes."""
from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp

from impala_sqlglot.compat_generation import (
    ImpalaGenerationGeneratorMixin,
    ImpalaGenerationParserMixin,
)
from impala_sqlglot.generator import ImpalaGenerator
from impala_sqlglot.impala import Impala
from impala_sqlglot.parser import ImpalaParser


if issubclass(ImpalaParser, ImpalaGenerationParserMixin):
    _Parser = ImpalaParser
else:
    class _Parser(ImpalaGenerationParserMixin, ImpalaParser):
        pass


if issubclass(ImpalaGenerator, ImpalaGenerationGeneratorMixin):
    _Generator = ImpalaGenerator
else:
    class _Generator(ImpalaGenerationGeneratorMixin, ImpalaGenerator):
        TRANSFORMS = {
            **ImpalaGenerator.TRANSFORMS,
            **ImpalaGenerationGeneratorMixin.GENERATION_TRANSFORMS,
        }


class _GenerationDialect(Impala):
    Parser = _Parser
    Generator = _Generator


DIALECT = _GenerationDialect()


def parse(sql: str) -> exp.Expression:
    return sqlglot.parse_one(sql, read=DIALECT)


def generated_and_reparsed(sql: str) -> tuple[exp.Expression, str]:
    tree = parse(sql)
    before = tree.copy()
    generated = tree.sql(dialect=DIALECT)
    assert tree == before, "generation changed its input tree"
    assert parse(generated) == tree
    return tree, generated


@pytest.mark.parametrize("target", ["Bar", "Foo.Bar"])
def test_alter_view_values_preserves_as_and_structure(target):
    tree, out = generated_and_reparsed(f"ALTER VIEW {target} AS VALUES(1, 2, 3)")
    assert isinstance(tree, exp.Alter)
    assert tree.args["kind"] == "VIEW"
    assert isinstance(tree.actions[0], exp.Values)
    assert out == f"ALTER VIEW {target} AS VALUES (1, 2, 3)"


def test_alter_view_values_pretty_roundtrip():
    tree = parse("ALTER VIEW demo AS VALUES (1, 2)")
    out = tree.sql(dialect=DIALECT, pretty=True)
    assert "AS VALUES" in out
    assert parse(out) == tree


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER VIEW demo AS SELECT 1",
        "ALTER TABLE demo ADD COLUMNS (k INT)",
    ],
)
def test_other_alter_generation_unchanged(sql):
    tree = parse(sql)
    assert tree.sql(dialect=DIALECT) == tree.sql(dialect=Impala())
    assert parse(tree.sql(dialect=DIALECT)) == tree


@pytest.mark.parametrize(
    ("declaration", "prefix"),
    [
        ("Foo() RETURNS INT", "Foo() RETURNS INT"),
        ("foo", "foo"),
        ("db.foo", "db.foo"),
        ("IF NOT EXISTS foo", "IF NOT EXISTS foo"),
        ("IF NOT EXISTS db.foo", "IF NOT EXISTS db.foo"),
    ],
)
def test_create_function_symbol_is_native(declaration, prefix):
    sql = f"CREATE FUNCTION {declaration} LOCATION 'f.jar' SYMBOL='class.Udf'"
    tree, out = generated_and_reparsed(sql)
    assert tree.args["kind"] == "FUNCTION"
    prop = next(p for p in tree.args["properties"].expressions if type(p) is exp.Property)
    assert isinstance(prop.this, exp.Var)
    assert prop.this.name == "SYMBOL"
    assert prop.args["value"].this == "class.Udf"
    assert out == f"CREATE FUNCTION {prefix} LOCATION 'f.jar' SYMBOL='class.Udf'"
    assert "TBLPROPERTIES" not in out


def test_scalar_function_native_lifecycle_options():
    sql = (
        "CREATE FUNCTION demo(INT) RETURNS INT LOCATION 'demo.so' "
        "SYMBOL='my_func' PREPARE_FN='prepare' CLOSE_FN='close'"
    )
    tree, out = generated_and_reparsed(sql)
    options = [p for p in tree.args["properties"].expressions if type(p) is exp.Property]
    assert [(p.this.name, p.args["value"].this) for p in options] == [
        ("SYMBOL", "my_func"),
        ("PREPARE_FN", "prepare"),
        ("CLOSE_FN", "close"),
    ]
    assert out == sql


def test_lowercase_native_option_names_are_not_quoted():
    sql = (
        "CREATE FUNCTION demo(INT) RETURNS INT LOCATION 'demo.so' "
        "symbol='my_func' prepare_fn='prepare' close_fn='close'"
    )
    _, out = generated_and_reparsed(sql)
    assert out == sql


def test_native_function_options_pretty_roundtrip():
    tree = parse(
        "CREATE FUNCTION demo(INT) RETURNS INT LOCATION 'demo.so' "
        "SYMBOL='my_func' PREPARE_FN='prepare' CLOSE_FN='close'"
    )
    out = tree.sql(dialect=DIALECT, pretty=True)
    assert "TBLPROPERTIES" not in out
    assert parse(out) == tree


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE demo (k INT) TBLPROPERTIES ('SYMBOL'='not a function')",
        "CREATE FUNCTION demo() RETURNS INT LOCATION 'demo.so' "
        "TBLPROPERTIES ('SYMBOL'='not a native option')",
    ],
)
def test_quoted_table_property_is_not_a_native_function_option(sql):
    tree = parse(sql)
    assert tree.sql(dialect=DIALECT) == tree.sql(dialect=Impala())


@pytest.mark.parametrize("style", ["", "FORMATTED ", "EXTENDED "])
def test_describe_database_keeps_target_kind(style):
    sql = f"DESCRIBE DATABASE {style}databasename"
    tree, out = generated_and_reparsed(sql)
    assert isinstance(tree, exp.Describe)
    assert tree.args["kind"] == "DATABASE"
    assert out == sql


@pytest.mark.parametrize(
    "sql",
    ["DESCRIBE TABLE FORMATTED demo", "DESCRIBE demo", "DESCRIBE HISTORY demo"],
)
def test_describe_non_database_delegates(sql):
    tree = parse(sql)
    assert tree.sql(dialect=DIALECT) == tree.sql(dialect=Impala())


@pytest.mark.parametrize(
    "sql",
    [
        ":foobar()",
        ":\tfoobar()",
        ": foobar()",
        ":foobar(123)",
        ":foobar(123, 456)",
        ":foobar('foo', 'bar')",
        ":foobar('foo', 'bar', 1, -1, 1234, 99, false)",
    ],
)
def test_colon_command_keeps_original_suffix(sql):
    tree, out = generated_and_reparsed(sql)
    assert isinstance(tree, exp.Command)
    assert tree.this == ":"
    assert out == sql


def test_other_command_delegates():
    tree = exp.Command(this="OTHER", expression="  demo()  ")
    assert tree.sql(dialect=DIALECT) == tree.sql(dialect=Impala())
