"""Independent synthetic controls for native table-key modifiers."""
from itertools import product

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

import impala_sqlglot  # noqa: F401
from impala_sqlglot.syntax import (
    ImpalaForeignKey, ImpalaPrimaryKey, NonUniquePrimaryKey,
    NonUniquePrimaryKeyColumnConstraint,
)


def parse(sql, error_level=sqlglot.ErrorLevel.RAISE):
    return sqlglot.parse_one(sql, read="impala", error_level=error_level)


def create(key):
    return f"CREATE TABLE demo_table (key_col INT, other_col INT, {key}) STORED AS PARQUET"


MODIFIERS = [
    (enabled, validated, rely)
    for enabled, validated, rely in product((None, True, False), repeat=3)
]


def modifier_sql(values):
    return " ".join(
        yes if value else no
        for value, yes, no in zip(
            values, ("ENABLE", "VALIDATE", "RELY"), ("DISABLE", "NOVALIDATE", "NORELY"),
        )
        if value is not None
    )


@pytest.mark.parametrize("values", MODIFIERS)
@pytest.mark.parametrize("kind", ["PRIMARY", "NON UNIQUE PRIMARY", "FOREIGN"])
def test_native_modifiers_and_omissions(values, kind):
    key_sql = f"{kind} KEY (key_col, other_col)"
    if kind == "FOREIGN":
        key_sql += " REFERENCES demo_parent (parent_key, parent_other)"
    suffix = modifier_sql(values)
    tree = parse(create(key_sql + (" " + suffix if suffix else "")))
    cls = exp.ForeignKey if kind == "FOREIGN" else exp.PrimaryKey
    key = tree.find(cls)
    assert key is not None
    assert [column.name for column in key.expressions] == ["key_col", "other_col"]
    for field, expected in zip(("enabled", "validated", "rely"), values):
        assert key.args.get(field) is expected
    if any(value is not None for value in values):
        assert isinstance(key, ImpalaForeignKey if kind == "FOREIGN" else ImpalaPrimaryKey)
    if kind == "NON UNIQUE PRIMARY":
        assert isinstance(key, NonUniquePrimaryKey)
        assert key.args["unique"] is False
    if kind == "FOREIGN":
        reference = key.args["reference"]
        assert reference.this.this.name == "demo_parent"
        assert [column.name for column in reference.this.expressions] == ["parent_key", "parent_other"]
    generated = tree.sql(dialect="impala")
    assert key_sql + (" " + suffix if suffix else "") in generated
    assert parse(generated) == tree


def test_reported_primary_key_spelling():
    tree = parse(
        "CREATE TABLE demo_table (key_col INT, "
        "PRIMARY KEY (key_col) DISABLE NOVALIDATE) STORED AS PARQUET"
    )
    key = tree.find(ImpalaPrimaryKey)
    assert key.args["enabled"] is False
    assert key.args["validated"] is False
    assert key.args.get("rely") is None
    assert key.args.get("options") == []
    assert parse(tree.sql(dialect="impala")) == tree


@pytest.mark.parametrize("modifiers", [
    "DISABLE DISABLE", "ENABLE DISABLE", "NOVALIDATE VALIDATE",
    "RELY RELY", "RELY NORELY", "NOVALIDATE DISABLE", "RELY ENABLE",
    "ENABLE RELY VALIDATE", "DISABLE NOVALIDATE RELY NORELY",
    "DISABLE NOT ENFORCED", "ENABLE bogus", "NOVALIDATE bogus",
])
@pytest.mark.parametrize("kind", ["PRIMARY", "FOREIGN"])
@pytest.mark.parametrize("level", [sqlglot.ErrorLevel.RAISE, sqlglot.ErrorLevel.IMMEDIATE])
def test_repeated_malformed_and_out_of_order_modifiers(modifiers, kind, level):
    key = f"{kind} KEY (key_col)"
    if kind == "FOREIGN":
        key += " REFERENCES demo_parent (key_col)"
    with pytest.raises(ParseError):
        parse(create(f"{key} {modifiers}"), level)


def test_not_enforced_is_separate_from_native_modifiers():
    tree = parse(create("PRIMARY KEY (key_col) NOT ENFORCED DISABLE NOVALIDATE RELY"))
    key = tree.find(ImpalaPrimaryKey)
    assert key.args["options"] == ["NOT ENFORCED"]
    assert key.args["enabled"] is False
    assert key.args["validated"] is False
    assert key.args["rely"] is True
    assert parse(tree.sql(dialect="impala")) == tree


@pytest.mark.parametrize("kind", ["PRIMARY", "FOREIGN"])
def test_comments_after_each_modifier_are_preserved_once(kind):
    key = f"{kind} KEY (key_col)"
    if kind == "FOREIGN":
        key += " REFERENCES demo_parent (key_col)"
    tree = parse(create(
        key + " /* columns */ DISABLE /* enable_flag */ "
        "NOVALIDATE /* validate_flag */ RELY /* rely_flag */"
    ))
    generated = tree.sql(dialect="impala")
    for marker in ("columns", "enable_flag", "validate_flag", "rely_flag"):
        assert generated.count(marker) == 1
    assert parse(generated) == tree


def test_primary_and_multiple_foreign_keys_keep_independent_flags():
    tree = parse(create(
        "PRIMARY KEY (key_col) DISABLE NOVALIDATE RELY, "
        "FOREIGN KEY (other_col) REFERENCES demo_parent (parent_key) ENABLE VALIDATE NORELY, "
        "FOREIGN KEY (key_col) REFERENCES demo_other (other_key) NOVALIDATE"
    ))
    primary = tree.find(ImpalaPrimaryKey)
    foreign = list(tree.find_all(ImpalaForeignKey))
    assert primary.args["rely"] is True
    assert len(foreign) == 2
    assert foreign[0].args["enabled"] is True
    assert foreign[0].args["validated"] is True
    assert foreign[0].args["rely"] is False
    assert foreign[1].args.get("enabled") is None
    assert foreign[1].args["validated"] is False
    assert foreign[1].args.get("rely") is None
    assert parse(tree.sql(dialect="impala")) == tree


@pytest.mark.parametrize("key", [
    "FOREIGN KEY (key_col)",
    "FOREIGN KEY (key_col) REFERENCES demo_parent",
    "FOREIGN KEY (key_col) REFERENCES demo_parent ()",
    "FOREIGN KEY () REFERENCES demo_parent (key_col)",
    "FOREIGN KEY (key_col) REFERENCES demo_catalog.demo_db.demo_parent (key_col)",
])
@pytest.mark.parametrize("level", [sqlglot.ErrorLevel.RAISE, sqlglot.ErrorLevel.IMMEDIATE])
def test_native_foreign_key_modifiers_require_complete_reference(key, level):
    with pytest.raises(ParseError):
        parse(create(key + " DISABLE NOVALIDATE RELY"), level)


@pytest.mark.parametrize("key", [
    "PRIMARY KEY (key_col,)",
    "PRIMARY KEY ('key_col')",
    "PRIMARY KEY (1)",
    "NON UNIQUE PRIMARY KEY (key_col) NOT ENFORCED",
    "FOREIGN KEY (key_col,) REFERENCES demo_parent (key_col)",
    "FOREIGN KEY (key_col) REFERENCES demo_parent (key_col,)",
    "FOREIGN KEY (key_col) REFERENCES demo_parent ('key_col')",
    "FOREIGN KEY (key_col) REFERENCES demo_parent (1)",
    "FOREIGN KEY (key_col) REFERENCES demo_parent (key_col) ON DELETE CASCADE",
])
@pytest.mark.parametrize("level", [sqlglot.ErrorLevel.RAISE, sqlglot.ErrorLevel.IMMEDIATE])
def test_native_modifiers_do_not_enable_invalid_key_forms(key, level):
    with pytest.raises(ParseError):
        parse(create(key + " DISABLE"), level)


def test_not_enforced_comment_is_preserved_on_both_dependency_versions():
    tree = parse(create("PRIMARY KEY (key_col) NOT /* modifier_comment */ ENFORCED DISABLE"))
    generated = tree.sql(dialect="impala")
    assert generated.count("modifier_comment") == 1
    assert parse(generated) == tree


@pytest.mark.parametrize("level", [sqlglot.ErrorLevel.RAISE, sqlglot.ErrorLevel.IMMEDIATE])
def test_ctas_does_not_accept_native_key_modifiers(level):
    with pytest.raises(ParseError):
        parse(
            "CREATE TABLE demo_table PRIMARY KEY (key_col) DISABLE "
            "STORED AS KUDU AS SELECT 1 AS key_col",
            level,
        )


def test_programmatic_ctas_with_modifiers_is_not_generated():
    tree = parse(
        "CREATE TABLE demo_table PRIMARY KEY (key_col) "
        "STORED AS KUDU AS SELECT 1 AS key_col"
    )
    old = tree.find(exp.PrimaryKey)
    old.replace(ImpalaPrimaryKey(**old.args, enabled=False))
    with pytest.raises(ValueError, match="AS SELECT"):
        tree.sql(dialect="impala")


@pytest.mark.parametrize("kind", ["PRIMARY KEY", "NON UNIQUE PRIMARY KEY"])
def test_column_key_controls_remain_unchanged(kind):
    tree = parse(f"CREATE TABLE demo_table (key_col INT {kind}, other_col INT) STORED AS KUDU")
    key = tree.find(exp.PrimaryKeyColumnConstraint)
    assert key is not None
    assert isinstance(key, NonUniquePrimaryKeyColumnConstraint) == kind.startswith("NON UNIQUE")
    assert parse(tree.sql(dialect="impala")) == tree


@pytest.mark.parametrize("cls", [ImpalaPrimaryKey, ImpalaForeignKey, NonUniquePrimaryKey])
@pytest.mark.parametrize("field", ["enabled", "validated", "rely"])
@pytest.mark.parametrize("value", ["DISABLE", 1, exp.true()])
def test_generation_rejects_non_boolean_flags(cls, field, value):
    key = cls(expressions=[exp.to_identifier("key_col")], **{field: value})
    with pytest.raises(ValueError, match="boolean or None"):
        key.sql(dialect="impala")


@pytest.mark.parametrize("options", [["RELY"], ["NORELY"], ["DISABLE"], ["MATCH FULL"]])
def test_generation_does_not_duplicate_native_flags_in_generic_options(options):
    key = ImpalaPrimaryKey(
        expressions=[exp.to_identifier("key_col")], enabled=False, options=options,
    )
    with pytest.raises(ValueError, match="modifier fields"):
        key.sql(dialect="impala")


@pytest.mark.parametrize("key", [
    ImpalaForeignKey(expressions=[exp.to_identifier("key_col")], enabled=False),
    ImpalaForeignKey(
        expressions=[exp.to_identifier("key_col")], enabled=False,
        reference=exp.Reference(this=exp.to_table("demo_parent")),
    ),
])
def test_programmatic_foreign_key_requires_a_complete_reference(key):
    with pytest.raises(ValueError, match="complete reference"):
        key.sql(dialect="impala")


def test_native_modifiers_require_explicit_impala_generation():
    key = parse(create("PRIMARY KEY (key_col) DISABLE NOVALIDATE RELY")).find(ImpalaPrimaryKey)
    assert key.sql(dialect="impala") == "PRIMARY KEY (key_col) DISABLE NOVALIDATE RELY"
    with pytest.raises(ValueError, match="Unsupported expression type"):
        key.sql()


@pytest.mark.parametrize("columns", [[], [exp.Literal.number(1)], [exp.Literal.string("key_col")]])
def test_programmatic_primary_key_requires_column_identifiers(columns):
    key = ImpalaPrimaryKey(expressions=columns, enabled=False)
    with pytest.raises(ValueError, match="column identifiers"):
        key.sql(dialect="impala")
