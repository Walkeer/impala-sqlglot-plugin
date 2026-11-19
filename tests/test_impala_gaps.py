"""Independent synthetic regressions for structured Impala syntax."""
from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from impala_sqlglot import Impala
from impala_sqlglot.syntax import (
    DropFunction, NonUniquePrimaryKey, NonUniquePrimaryKeyColumnConstraint,
    Upsert, VariadicParameter,
    ImpalaSubquery, PlanHintProperty,
)


def parse(sql):
    assert sqlglot.Dialect.get("impala") is Impala
    return sqlglot.parse_one(sql, read="impala")


def roundtrip(tree):
    generated = tree.sql(dialect="impala")
    assert parse(generated) == tree
    return generated


@pytest.mark.parametrize("operation", ["INSERT", "UPSERT"])
@pytest.mark.parametrize("source", ["SELECT demo_key FROM demo_input", "VALUES (1, 'synthetic')"])
def test_write_operation_and_source(operation, source):
    tree = parse(f"{operation} INTO demo_output (demo_key, demo_value) {source}")
    assert type(tree) is (Upsert if operation == "UPSERT" else exp.Insert)
    assert isinstance(tree.this, exp.Schema)
    assert tree.this.this.name == "demo_output"
    assert [column.name for column in tree.this.expressions] == ["demo_key", "demo_value"]
    assert isinstance(tree.expression, exp.Select if source.startswith("SELECT") else exp.Values)
    assert roundtrip(tree).startswith(f"{operation} INTO demo_output")


def test_insert_overwrite_partition_control():
    tree = parse("INSERT OVERWRITE TABLE demo_output PARTITION (demo_part=1) SELECT demo_key FROM demo_input")
    assert type(tree) is exp.Insert
    assert tree.args["overwrite"]
    assert isinstance(tree.args["partition"], exp.Partition)
    assert roundtrip(tree).startswith("INSERT OVERWRITE TABLE demo_output PARTITION")


@pytest.mark.parametrize("hint", ["[BROADCAST]", "/* +BROADCAST */", "/*+BROADCAST*/", "-- +BROADCAST\n", "[SHUFFLE]"])
def test_join_hint_attachment(hint):
    tree = parse(f"SELECT a.demo_key FROM demo_input a JOIN {hint} demo_lookup b ON a.demo_key=b.demo_key")
    join = tree.args["joins"][0]
    assert isinstance(join.args["hint"], exp.Hint)
    assert join.this.name == "demo_lookup"
    assert join.this.alias == "b"
    assert isinstance(join.args["on"], exp.EQ)
    generated = roundtrip(tree)
    assert "JOIN /*+" in generated
    assert generated.index("JOIN") < generated.index("BROADCAST" if "BROADCAST" in hint else "SHUFFLE") < generated.index("demo_lookup")


@pytest.mark.parametrize("operation", ["INSERT", "UPSERT"])
@pytest.mark.parametrize("hint", ["[SHUFFLE]", "/* +SHUFFLE */", "/*+SHUFFLE*/", "-- +SHUFFLE\n"])
@pytest.mark.parametrize("leading", [True, False])
def test_write_hints_are_not_target_indexing(operation, hint, leading):
    sql = f"{operation} {hint} INTO demo_output SELECT demo_key FROM demo_input" if leading else f"{operation} INTO demo_output {hint} SELECT demo_key FROM demo_input"
    tree = parse(sql)
    assert isinstance(tree.args["hint"], exp.Hint)
    assert isinstance(tree.this, exp.Table)
    assert tree.this.name == "demo_output"
    assert not tree.find(exp.Bracket)
    generated = roundtrip(tree)
    assert generated.startswith(f"{operation} /*+")
    assert generated.index("SHUFFLE") < generated.index("INTO demo_output")


def test_straight_join_is_query_modifier():
    tree = parse("SELECT DISTINCT STRAIGHT_JOIN a.demo_key FROM demo_input a JOIN demo_lookup b ON a.demo_key=b.demo_key")
    assert isinstance(tree.expressions[0], exp.Column)
    assert tree.expressions[0].name == "demo_key"
    assert isinstance(tree.args["distinct"], exp.Distinct)
    assert [modifier.name for modifier in tree.args["operation_modifiers"]] == ["STRAIGHT_JOIN"]
    assert len(tree.args["joins"]) == 1
    assert roundtrip(tree).startswith("SELECT DISTINCT STRAIGHT_JOIN a.demo_key")


def test_select_hint_and_modifier():
    tree = parse("SELECT DISTINCT /* +STRAIGHT_JOIN */ demo_key FROM demo_input")
    assert isinstance(tree.args["hint"], exp.Hint)
    assert tree.expressions[0].name == "demo_key"
    assert roundtrip(tree).startswith("SELECT DISTINCT /*+ STRAIGHT_JOIN */")


@pytest.mark.parametrize(("kind", "value"), [("SYSTEM_TIME", "'2024-02-01 00:00:00'"), ("SYSTEM_VERSION", "7")])
def test_time_travel_generation(kind, value):
    tree = parse(f"SELECT demo_key FROM `demo_table` FOR {kind} AS OF {value} AS demo_alias")
    table = tree.find(exp.Table)
    assert table.name == "demo_table"
    assert table.alias == "demo_alias"
    assert isinstance(table.args["version"], exp.Version)
    generated = roundtrip(tree)
    assert f"FOR {kind} AS OF {value}" in generated


@pytest.mark.parametrize("percent", [0, 1, 10, 100])
@pytest.mark.parametrize("seed", [None, 1, 9223372036854775807])
def test_sampling_preserves_percent_method_seed(percent, seed):
    suffix = "" if seed is None else f" REPEATABLE({seed})"
    tree = parse(f"SELECT demo_key FROM demo_table TABLESAMPLE SYSTEM({percent}){suffix}")
    sample = tree.find(exp.TableSample)
    assert sample.args["method"].name == "SYSTEM"
    assert sample.args["percent"] == exp.Literal.number(percent)
    assert sample.args.get("size") is None
    assert sample.args.get("seed") == (None if seed is None else exp.Literal.number(seed))
    assert f"TABLESAMPLE SYSTEM({percent}){suffix}" in roundtrip(tree)


@pytest.mark.parametrize("column", [True, False])
@pytest.mark.parametrize("non_unique", [True, False])
def test_primary_key_uniqueness(column, non_unique):
    key = "NON UNIQUE PRIMARY KEY" if non_unique else "PRIMARY KEY"
    schema = f"demo_key BIGINT {key}, demo_value STRING" if column else f"demo_key BIGINT, demo_value STRING, {key} (demo_key)"
    tree = parse(f"CREATE TABLE demo_table ({schema}) STORED AS KUDU")
    cls = NonUniquePrimaryKeyColumnConstraint if column else NonUniquePrimaryKey
    if non_unique:
        constraint = tree.find(cls)
        assert constraint is not None
        assert constraint.args["unique"] is False
    else:
        assert tree.find(NonUniquePrimaryKey) is None
        assert tree.find(NonUniquePrimaryKeyColumnConstraint) is None
    generated = roundtrip(tree)
    assert (key in generated) and (("NON UNIQUE" in generated) is non_unique)


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("aggregate", [False, True])
def test_variadic_drop_signature(exists, aggregate):
    prefix = "DROP AGGREGATE FUNCTION" if aggregate else "DROP FUNCTION"
    optional = " IF EXISTS" if exists else ""
    tree = parse(f"{prefix}{optional} demo_function(STRING, INT...)")
    assert isinstance(tree, DropFunction)
    assert tree.this.name == "demo_function"
    assert bool(tree.args["exists"]) is exists
    assert bool(tree.args["aggregate"]) is aggregate
    assert isinstance(tree.expressions[-1], VariadicParameter)
    assert len(tree.expressions) == 2
    assert roundtrip(tree) == f"{prefix}{optional} demo_function(STRING, INT...)"


@pytest.mark.parametrize("clause", ["DISTRIBUTE BY", "CLUSTER BY", "SORT BY"])
def test_hive_query_clause_rejected(clause):
    with pytest.raises(ParseError):
        parse(f"SELECT demo_key FROM demo_table {clause} demo_key")


def test_order_window_and_shared_identifier_controls():
    tree = parse("SELECT SUM(demo_key) OVER (PARTITION BY demo_group ORDER BY demo_key) AS demo_sum FROM demo_table ORDER BY demo_key")
    assert tree.find(exp.Window)
    assert tree.args["order"]
    roundtrip(tree)
    assert parse("SELECT distribute, cluster, sort FROM demo_table").expressions[0].name == "distribute"


@pytest.mark.parametrize("sql", [
    "UPSERT demo_output SELECT demo_key FROM demo_input", "UPSERT INTO demo_output",
    "UPSERT OVERWRITE TABLE demo_output SELECT demo_key FROM demo_input",
    "UPSERT INTO demo_output PARTITION (demo_part=1) VALUES (1)",
    "SELECT demo_key FROM demo_input JOIN [BROADCAST demo_lookup ON TRUE",
    "SELECT demo_key FROM demo_input JOIN [1] demo_lookup ON TRUE",
    "INSERT [SHUFFLE INTO demo_output VALUES (1)",
    "SELECT demo_key FROM demo_table FOR SYSTEM_VERSION AS OF demo_key",
    "SELECT demo_key FROM demo_table FOR SYSTEM_VERSION AS OF '7'",
    "SELECT demo_key FROM demo_table FOR SYSTEM_VERSION AS OF 1 + 1",
    "SELECT demo_key FROM demo_table TABLESAMPLE BERNOULLI(10)",
    "SELECT demo_key FROM demo_table TABLESAMPLE SYSTEM(101)",
    "SELECT demo_key FROM demo_table TABLESAMPLE SYSTEM(-1)",
    "SELECT demo_key FROM demo_table TABLESAMPLE SYSTEM(1.5)",
    "SELECT demo_key FROM demo_table TABLESAMPLE SYSTEM(10) REPEATABLE(-1)",
    "DROP FUNCTION demo_function(INT..., STRING)",
    "DROP FUNCTION demo_function(INT....)", "DROP FUNCTION demo_function(INT..)",
    "DROP FUNCTION demo_function(INT,)", "DROP FUNCTION demo_function(BOGUS...)",
])
@pytest.mark.parametrize("error_level", [sqlglot.ErrorLevel.IMMEDIATE, sqlglot.ErrorLevel.RAISE])
def test_malformed_supported_syntax_rejected(sql, error_level):
    with pytest.raises(ParseError):
        sqlglot.parse_one(sql, read="impala", error_level=error_level)


@pytest.mark.parametrize("hint", ["[STRAIGHT_JOIN]", "/*+STRAIGHT_JOIN*/", "-- +STRAIGHT_JOIN\n"])
@pytest.mark.parametrize("modifier", ["", "ALL ", "DISTINCT "])
def test_select_hint_after_native_modifier(modifier, hint):
    tree = parse(f"SELECT {modifier}{hint} demo_key FROM demo_input")
    assert tree.args["hint"].expressions[0].name == "STRAIGHT_JOIN"
    assert tree.expressions[0].name == "demo_key"
    generated = roundtrip(tree)
    assert "/*+ STRAIGHT_JOIN */" in generated


@pytest.mark.parametrize("hint", ["[SHUFFLE, CLUSTERED]", "/*+SHUFFLE, CLUSTERED*/", "-- +SHUFFLE, CLUSTERED\n"])
def test_multiple_write_hints(hint):
    tree = parse(f"UPSERT {hint} INTO demo_output VALUES (1)")
    assert [item.name for item in tree.args["hint"].expressions] == ["SHUFFLE", "CLUSTERED"]
    roundtrip(tree)


@pytest.mark.parametrize("value", ["-1", "1.5"])
def test_numeric_version_literals_follow_native_grammar(value):
    tree = parse(f"SELECT demo_key FROM demo_table FOR SYSTEM_VERSION AS OF {value}")
    assert f"AS OF {value}" in roundtrip(tree)


def test_sample_alias_and_quoted_target():
    tree = parse("SELECT s.demo_key FROM `demo_table` s TABLESAMPLE SYSTEM(10) REPEATABLE(1)")
    assert tree.find(exp.Table).alias == "s"
    assert isinstance(tree.find(exp.Table).args["sample"], exp.TableSample)
    assert tree.args.get("sample") is None
    roundtrip(tree)


@pytest.mark.parametrize("operation", ["INSERT", "UPSERT"])
def test_write_cte_and_ordinary_comments(operation):
    tree = parse(f"WITH demo_cte AS (SELECT demo_key FROM demo_input) {operation} /* ordinary write comment */ [SHUFFLE] INTO demo_output SELECT demo_key FROM demo_cte")
    assert tree.args["with_"]
    assert tree.args["hint"].expressions[0].name == "SHUFFLE"
    assert any("ordinary write comment" in comment for comment in tree.comments)
    generated = roundtrip(tree)
    assert "WITH demo_cte" in generated
    assert generated.count("ordinary write comment") == 1


@pytest.mark.parametrize("schema", [
    "demo_key BIGINT NON UNIQUE",
    "demo_key BIGINT, NON UNIQUE PRIMARY KEY",
    "demo_key BIGINT, NON UNIQUE PRIMARY KEY ()",
])
def test_malformed_non_unique_constraints(schema):
    with pytest.raises(ParseError):
        parse(f"CREATE TABLE demo_table ({schema}) STORED AS KUDU")


@pytest.mark.parametrize("suffix", ["", "()", "(INT)", "(INT...)"])
def test_drop_preserves_presence_of_signature(suffix):
    tree = parse(f"DROP FUNCTION demo_function{suffix}")
    assert tree.args["wrapped"] is bool(suffix)
    assert roundtrip(tree) == f"DROP FUNCTION demo_function{suffix}"


@pytest.mark.parametrize("sql", [
    "DROP FUNCTION demo_function(",
    "DROP FUNCTION demo_function(INT,",
    "DROP FUNCTION demo_function(INT",
    "DROP FUNCTION demo_function(INT...",
])
@pytest.mark.parametrize("error_level", [sqlglot.ErrorLevel.IMMEDIATE, sqlglot.ErrorLevel.RAISE])
def test_incomplete_signature_terminates_with_parse_error(sql, error_level):
    with pytest.raises(ParseError):
        sqlglot.parse_one(sql, read="impala", error_level=error_level)


@pytest.mark.parametrize("types", ["", "INT", "INT...", "STRING, DECIMAL(12, 3)..."])
def test_create_scalar_function_signature_checked_separately(types):
    tree = parse(f"CREATE FUNCTION demo_function({types}) RETURNS INT LOCATION '/synthetic/function.so' SYMBOL='demo_function'")
    assert isinstance(tree, exp.Create)
    assert isinstance(tree.this, exp.UserDefinedFunction)
    if types.endswith("..."):
        assert isinstance(tree.this.expressions[-1], VariadicParameter)
    assert "SYMBOL='demo_function'" in roundtrip(tree)


@pytest.mark.parametrize("hint", ["[BROADCAST]", "/*+BROADCAST*/", "/* +BROADCAST */", "-- +BROADCAST\n"])
@pytest.mark.parametrize("alias", ["", " AS demo_alias"])
def test_table_hints_remain_attached_to_the_table(hint, alias):
    tree = parse(f"SELECT demo_key FROM demo_table{alias} {hint}")
    table = tree.find(exp.Table)
    assert isinstance(table.args["hints"][0], exp.Hint)
    assert table.args["hints"][0].expressions[0].name == "BROADCAST"
    assert not tree.find(exp.Bracket)
    roundtrip(tree)


def test_join_and_table_hints_are_distinct():
    tree = parse("SELECT a.demo_key FROM demo_input a [BROADCAST] JOIN [SHUFFLE] demo_lookup b [RANDOM_REPLICA] ON a.demo_key=b.demo_key")
    join = tree.args["joins"][0]
    assert join.args["hint"].expressions[0].name == "SHUFFLE"
    assert join.this.args["hints"][0].expressions[0].name == "RANDOM_REPLICA"
    assert tree.args["from_"].this.args["hints"][0].expressions[0].name == "BROADCAST"
    roundtrip(tree)


@pytest.mark.parametrize("hint", ["[]", "/*+*/", "-- +\n"])
def test_native_empty_hint_is_a_noop(hint):
    tree = parse(f"SELECT demo_key FROM demo_input JOIN {hint} demo_lookup ON TRUE")
    assert tree.args["joins"][0].args.get("hint") is None
    roundtrip(tree)


@pytest.mark.parametrize("sql", [
    "INSERT [SHUFFLE] INTO demo_output [NOSHUFFLE] VALUES (1)",
    "UPSERT [SHUFFLE] INTO demo_output [CLUSTERED] VALUES (1)",
])
def test_write_hints_cannot_occur_in_both_positions(sql):
    with pytest.raises(ParseError):
        parse(sql)


@pytest.mark.parametrize("sql", [
    "UPSERT INTO demo_output SELECT demo_key FROM demo_input",
    "CREATE TABLE demo_table (demo_key INT NON UNIQUE PRIMARY KEY) STORED AS KUDU",
    "DROP FUNCTION demo_function(INT...)",
])
def test_custom_core_nodes_survive_serialization(sql):
    tree = parse(sql)
    loaded = exp.Expression.load(tree.dump())
    assert loaded == tree
    assert type(loaded) is type(tree)
    roundtrip(loaded)


def test_generation_rejects_other_sampling_methods_and_units():
    for sample in (
        exp.TableSample(method=exp.var("BERNOULLI"), percent=exp.Literal.number(10)),
        exp.TableSample(method=exp.var("SYSTEM"), size=exp.Literal.number(10)),
    ):
        with pytest.raises(ValueError):
            sample.sql(dialect="impala")


def test_generation_does_not_discard_write_options():
    tree = parse("INSERT INTO demo_output VALUES (1)")
    tree.set("returning", exp.Returning(expressions=[exp.Star()]))
    with pytest.raises(ValueError, match="write option"):
        tree.sql(dialect="impala")


@pytest.mark.parametrize("hint", ["[STRAIGHT_JOIN]", "/*+STRAIGHT_JOIN*/"])
def test_distinct_hints_on_both_set_operands(hint):
    tree = parse(f"SELECT DISTINCT {hint} demo_key FROM demo_input UNION ALL SELECT DISTINCT {hint} demo_key FROM demo_other")
    assert isinstance(tree, exp.Union)
    assert tree.this.args["hint"].expressions[0].name == "STRAIGHT_JOIN"
    assert tree.expression.args["hint"].expressions[0].name == "STRAIGHT_JOIN"
    assert roundtrip(tree).count("/*+ STRAIGHT_JOIN */") == 2


def test_multiline_ordinary_comment_does_not_become_hint():
    tree = parse("SELECT /*\n +STRAIGHT_JOIN */ demo_key FROM demo_input")
    assert tree.args.get("hint") is None
    assert tree.expressions[0].name == "demo_key"
    roundtrip(tree)


@pytest.mark.parametrize("sql", [
    "SELECT /*+STRAIGHT_JOIN*/ /* ordinary comment */ demo_key FROM demo_input",
    "SELECT demo_key FROM demo_input JOIN /*+BROADCAST*/ /* ordinary comment */ demo_lookup ON TRUE",
    "INSERT /*+SHUFFLE*/ /* ordinary comment */ INTO demo_output VALUES (1)",
])
def test_ordinary_comment_after_hint_is_not_lost(sql):
    tree = parse(sql)
    assert roundtrip(tree).count("ordinary comment") == 1


@pytest.mark.parametrize("operation", ["INSERT", "UPSERT"])
@pytest.mark.parametrize("columns", ["()", "(demo_key)", "(demo_key, demo_value)"])
def test_native_source_less_write_requires_column_permutation(operation, columns):
    tree = parse(f"{operation} INTO demo_output {columns}")
    assert isinstance(tree.this, exp.Schema)
    assert tree.expression is None
    assert roundtrip(tree) == f"{operation} INTO demo_output {columns}"


@pytest.mark.parametrize("columns", ["('demo_key')", "(1)", "(demo_key INT)", "(demo_key,)", "("])
@pytest.mark.parametrize("error_level", [sqlglot.ErrorLevel.IMMEDIATE, sqlglot.ErrorLevel.RAISE])
def test_write_columns_must_be_identifiers(columns, error_level):
    with pytest.raises(ParseError):
        sqlglot.parse_one(
            f"UPSERT INTO demo_output {columns} SELECT demo_key FROM demo_input",
            read="impala", error_level=error_level,
        )


@pytest.mark.parametrize("hint", ["[BROADCAST]", "/*+BROADCAST*/", "-- +BROADCAST\n"])
def test_inline_view_hint_preserves_alias_and_source(hint):
    tree = parse(f"SELECT s.demo_key FROM (SELECT demo_key FROM demo_input) s {hint}")
    view = tree.find(ImpalaSubquery)
    assert view.alias == "s"
    assert view.args["hints"][0].expressions[0].name == "BROADCAST"
    roundtrip(tree)
    loaded = exp.Expression.load(tree.dump())
    assert loaded == tree
    roundtrip(loaded)


def test_table_hint_after_sampling():
    tree = parse("SELECT demo_key FROM demo_input TABLESAMPLE SYSTEM(10) [RANDOM_REPLICA]")
    table = tree.find(exp.Table)
    assert table.args["sample"]
    assert table.args["hints"][0].expressions[0].name == "RANDOM_REPLICA"
    roundtrip(tree)


@pytest.mark.parametrize("hint", ["[SHUFFLE]", "/*+SHUFFLE*/", "-- +SHUFFLE\n"])
def test_create_table_as_select_hint(hint):
    tree = parse(f"CREATE {hint} TABLE demo_output STORED AS PARQUET AS SELECT demo_key FROM demo_input")
    assert isinstance(tree, exp.Create)
    prop = tree.find(PlanHintProperty)
    assert prop.this.expressions[0].name == "SHUFFLE"
    assert roundtrip(tree).startswith("CREATE /*+ SHUFFLE */ TABLE")


@pytest.mark.parametrize("name", ["demo_hint", "max"])
@pytest.mark.parametrize("arguments", ["10", "demo_key", "demo_key, demo_value"])
def test_hint_arguments_are_not_sql_function_semantics(name, arguments):
    tree = parse(f"SELECT /*+{name}({arguments})*/ demo_key FROM demo_input")
    hint = tree.args["hint"].expressions[0]
    assert isinstance(hint, exp.Anonymous)
    assert hint.name == name.upper()
    roundtrip(tree)


@pytest.mark.parametrize("arguments", ["", "1, 2", "demo_key, 1", "demo_key + 1"])
def test_hint_arguments_follow_native_identifier_or_single_integer_grammar(arguments):
    with pytest.raises(ParseError):
        parse(f"SELECT /*+demo_hint({arguments})*/ demo_key FROM demo_input")


@pytest.mark.parametrize("error_level", [sqlglot.ErrorLevel.IGNORE, sqlglot.ErrorLevel.WARN])
def test_comment_hint_respects_requested_error_level(error_level):
    sqlglot.parse_one(
        "SELECT /*+demo_hint(demo_key + 1)*/ demo_key FROM demo_input",
        read="impala", error_level=error_level,
    )


@pytest.mark.parametrize("sql", [
    "DROP FUNCTION demo_function(NULL)",
    "CREATE FUNCTION demo_function(NULL) RETURNS INT LOCATION '/synthetic/function.so' SYMBOL='demo_function'",
])
@pytest.mark.parametrize("error_level", [sqlglot.ErrorLevel.IMMEDIATE, sqlglot.ErrorLevel.RAISE])
def test_null_is_not_a_function_parameter_type(sql, error_level):
    with pytest.raises(ParseError):
        sqlglot.parse_one(sql, read="impala", error_level=error_level)


def test_unique_constraint_is_not_silently_deleted_by_hive_preprocessing():
    tree = parse("CREATE TABLE demo_table (demo_key INT UNIQUE)")
    assert tree.find(exp.UniqueColumnConstraint)
    with pytest.raises(ValueError, match="UNIQUE"):
        tree.sql(dialect="impala")


@pytest.mark.parametrize("prefix", [
    "SELECT demo_key FROM demo_input", "COMPUTE STATS demo_input",
])
@pytest.mark.parametrize("sample", ["SYSTEM(101)", "SYSTEM(10) REPEATABLE(9223372036854775808)"])
def test_query_and_statistics_sampling_have_same_limits(prefix, sample):
    with pytest.raises(ParseError):
        parse(f"{prefix} TABLESAMPLE {sample}")


def test_sampling_rejects_unbounded_integer_without_python_conversion():
    sample = exp.TableSample(
        method=exp.var("SYSTEM"), percent=exp.Literal.number(10),
        seed=exp.Literal.number("9" * 5000),
    )
    with pytest.raises(ValueError, match="integer seed"):
        sample.sql(dialect="impala")


@pytest.mark.parametrize("all_hinted", [False, True])
def test_three_way_union_keeps_hints_on_the_correct_operands(all_hinted):
    suffix = "/*+STRAIGHT_JOIN*/ " if all_hinted else ""
    tree = parse(
        "SELECT DISTINCT /*+STRAIGHT_JOIN*/ demo_key FROM demo_first UNION ALL "
        f"SELECT DISTINCT {suffix}demo_key FROM demo_second UNION ALL "
        f"SELECT DISTINCT {suffix}demo_key FROM demo_third"
    )
    assert isinstance(tree, exp.Union) and isinstance(tree.this, exp.Union)
    assert tree.this.this.args["hint"].expressions[0].name == "STRAIGHT_JOIN"
    assert bool(tree.this.expression.args.get("hint")) is all_hinted
    assert bool(tree.expression.args.get("hint")) is all_hinted
    roundtrip(tree)


@pytest.mark.parametrize("comment", [
    "/* +STRAIGHT_JOIN\n */", "/*+STRAIGHT_JOIN\n */",
    "/* +STRAIGHT_JOIN\r\n */", "/*+\n STRAIGHT_JOIN */",
])
def test_multiline_block_comments_are_not_active_plan_hints(comment):
    tree = parse(f"SELECT {comment} demo_key FROM demo_input")
    assert tree.args.get("hint") is None
    roundtrip(tree)


@pytest.mark.parametrize("hint", ["[SCHEDULE_REMOTE]", "/*+SCHEDULE_REMOTE*/", "-- +SCHEDULE_REMOTE\n"])
def test_aliased_sampled_table_keeps_table_hint(hint):
    tree = parse(f"SELECT s.demo_key FROM demo_input s TABLESAMPLE SYSTEM(10) {hint}")
    table = tree.find(exp.Table)
    assert table.alias == "s"
    assert table.args["sample"].args["percent"] == exp.Literal.number(10)
    assert table.args["hints"][0].expressions[0].name == "SCHEDULE_REMOTE"
    assert tree.args.get("sample") is None
    generated = roundtrip(tree)
    assert "demo_input AS s TABLESAMPLE SYSTEM(10)" in generated
    assert generated.count("SCHEDULE_REMOTE") == 1


def test_sampled_inline_view_keeps_hint_and_alias():
    tree = parse("SELECT s.demo_key FROM (SELECT demo_key FROM demo_input) s TABLESAMPLE SYSTEM(10) [SCHEDULE_REMOTE]")
    view = tree.find(ImpalaSubquery)
    assert view.alias == "s"
    assert view.args["sample"].args["percent"] == exp.Literal.number(10)
    assert view.args["hints"][0].expressions[0].name == "SCHEDULE_REMOTE"
    assert "AS s TABLESAMPLE SYSTEM(10)" in roundtrip(tree)
