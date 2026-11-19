"""Synthetic tests for Impala's structured Kudu and schema-reference DDL."""

from __future__ import annotations

import builtins
import socket
from pathlib import Path

import pytest
import sqlglot
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from impala_sqlglot.ddl import (
    ImpalaDDLGeneratorMixin,
    ImpalaDDLParserMixin,
    KuduAlterRangePartition,
    KuduHashPartition,
    KuduPartitioningProperty,
    KuduRangeBound,
    KuduRangePartition,
    KuduRangePartitioning,
    ParquetSchemaReference,
)
from impala_sqlglot.generator import ImpalaGenerator
from impala_sqlglot.impala import Impala
from impala_sqlglot.parser import ImpalaParser


assert issubclass(ImpalaParser, ImpalaDDLParserMixin)
assert issubclass(ImpalaGenerator, ImpalaDDLGeneratorMixin)
_Parser = ImpalaParser
_Generator = ImpalaGenerator
DIALECT = Impala()


def parse(sql: str) -> exp.Expression:
    return parse_one(sql, read=DIALECT)


def round_trip(tree: exp.Expression) -> str:
    sql = tree.sql(dialect=DIALECT)
    assert parse(sql) == tree
    return sql


def kudu_property(tree: exp.Expression) -> KuduPartitioningProperty:
    property_ = tree.find(KuduPartitioningProperty)
    assert isinstance(property_, KuduPartitioningProperty)
    return property_


def alter_partition(tree: exp.Expression) -> KuduAlterRangePartition:
    assert isinstance(tree, exp.Alter)
    assert len(tree.actions) == 1
    action = tree.actions[0]
    assert isinstance(action, KuduAlterRangePartition)
    assert isinstance(action.this, KuduRangePartition)
    return action


def test_create_hash_components_and_storage_are_structured():
    tree = parse(
        "CREATE TABLE db.t (id INT, shard INT) "
        "PARTITION BY HASH (id) PARTITIONS 4, HASH (`shard`) PARTITIONS 8 "
        "STORED AS KUDU"
    )
    assert isinstance(tree, exp.Create)
    assert isinstance(tree.this, exp.Schema)
    assert isinstance(tree.this.this, exp.Table)
    assert tree.this.this.db == "db"
    assert [column.name for column in tree.this.expressions] == ["id", "shard"]
    assert not tree.find(exp.Command)

    components = kudu_property(tree).expressions
    assert all(isinstance(item, KuduHashPartition) for item in components)
    assert [[column.name for column in item.args["columns"]] for item in components] == [
        ["id"],
        ["shard"],
    ]
    assert components[1].args["columns"][0].args["quoted"] is True
    assert [item.args["buckets"].this for item in components] == ["4", "8"]
    storage = tree.find(exp.FileFormatProperty)
    assert isinstance(storage, exp.FileFormatProperty)
    assert storage.name.upper() == "KUDU"
    assert round_trip(tree) == (
        "CREATE TABLE db.t (id INT, shard INT) "
        "PARTITION BY HASH (id) PARTITIONS 4, HASH (`shard`) PARTITIONS 8 "
        "STORED AS KUDU"
    )


def test_hash_can_implicitly_use_primary_key_and_omit_separator():
    tree = parse(
        "CREATE TABLE t (id INT) PARTITION BY HASH PARTITIONS 2 "
        "HASH (id) PARTITIONS 5 STORED AS KUDU"
    )
    first, second = kudu_property(tree).expressions
    assert first.args["columns"] is None
    assert second.args["columns"][0].name == "id"
    assert "HASH PARTITIONS 2, HASH (id) PARTITIONS 5" in round_trip(tree)


def test_hash_does_not_invent_a_storage_format():
    tree = parse("CREATE TABLE t (id INT) PARTITION BY HASH (id) PARTITIONS 2")
    assert tree.find(exp.FileFormatProperty) is None
    assert "STORED AS" not in round_trip(tree)


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE t (id INT PRIMARY KEY, v STRING) "
        "PARTITION BY HASH (id) PARTITIONS 2 STORED AS KUDU",
        "CREATE TABLE t (id INT, v STRING, PRIMARY KEY (id)) "
        "PARTITION BY HASH (id) PARTITIONS 2 STORED AS KUDU",
        "CREATE TABLE t PRIMARY KEY (id) PARTITION BY HASH (id) PARTITIONS 2 "
        "STORED AS KUDU AS SELECT 1 AS id",
    ],
)
def test_hash_composes_with_existing_primary_keys_and_ctas(sql):
    tree = parse(sql)
    assert isinstance(tree, exp.Create)
    assert tree.find(exp.PrimaryKey) or tree.find(exp.PrimaryKeyColumnConstraint)
    assert isinstance(kudu_property(tree).expressions[0], KuduHashPartition)
    assert isinstance(tree.find(exp.FileFormatProperty), exp.FileFormatProperty)
    round_trip(tree)


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE t PARTITION BY RANGE (id) (PARTITION 1 < VALUES < 10) "
        "STORED AS KUDU AS SELECT 1 AS id",
        "CREATE TABLE t PRIMARY KEY (id) PARTITION BY RANGE (id) "
        "(PARTITION VALUE = 1) STORED AS KUDU AS SELECT 1 AS id",
    ],
)
def test_range_composes_with_ctas_and_optional_existing_primary_key(sql):
    tree = parse(sql)
    assert isinstance(tree, exp.Create)
    assert isinstance(tree.expression, exp.Select)
    assert isinstance(kudu_property(tree).expressions[0], KuduRangePartitioning)
    round_trip(tree)


def test_range_preserves_arithmetic_columns_and_function_expressions():
    tree = parse(
        "CREATE TABLE t (a INT, b INT) PARTITION BY RANGE (a) "
        "(PARTITION b < VALUES <= a, PARTITION 1 + 1 < VALUES, "
        "PARTITION now() <= VALUES, PARTITION VALUE = add_months(now(), 2)) "
        "STORED AS KUDU"
    )
    partitions = kudu_property(tree).expressions[0].expressions
    assert isinstance(partitions[0].args["lower"].this, exp.Column)
    assert isinstance(partitions[0].args["upper"].this, exp.Column)
    assert isinstance(partitions[1].args["lower"].this, exp.Add)
    assert isinstance(partitions[2].args["lower"].this, exp.Expression)
    assert isinstance(partitions[3].args["value"], exp.Expression)
    round_trip(tree)


@pytest.mark.parametrize("range_first", [False, True])
def test_combined_hash_and_range_preserves_bounds_and_component_order(range_first):
    hash_ = "HASH (id) PARTITIONS 4"
    range_ = (
        "RANGE (id, shard) (PARTITION (1, 2) <= VALUES < (10, 3), "
        "PARTITION VALUE = (11, 0), PARTITION VALUES > (20, 0))"
    )
    components_sql = f"{range_}, {hash_}" if range_first else f"{hash_}, {range_}"
    tree = parse(
        f"CREATE TABLE t (id INT, shard INT) PARTITION BY {components_sql} STORED AS KUDU"
    )
    components = kudu_property(tree).expressions
    assert isinstance(components[0], KuduRangePartitioning if range_first else KuduHashPartition)
    range_component = next(item for item in components if isinstance(item, KuduRangePartitioning))
    assert [column.name for column in range_component.args["columns"]] == ["id", "shard"]
    bounded, equal, unbounded = range_component.expressions
    assert isinstance(bounded.args["lower"], KuduRangeBound)
    assert bounded.args["lower"].args["inclusive"] is True
    assert bounded.args["upper"].args["inclusive"] is False
    assert isinstance(bounded.args["lower"].this, exp.Tuple)
    assert isinstance(bounded.args["upper"].this, exp.Tuple)
    assert isinstance(equal.args["value"], exp.Tuple)
    assert unbounded.args["lower"].args["inclusive"] is False
    assert unbounded.args["upper"] is None
    assert "PARTITION (20, 0) < VALUES" in round_trip(tree)


def test_range_without_column_list_and_custom_hashes():
    tree = parse(
        "CREATE TABLE t (id INT) PARTITION BY RANGE "
        "(PARTITION VALUE = 10 HASH (id) PARTITIONS 3, HASH PARTITIONS 2, "
        "PARTITION 11 <= VALUES HASH (id) PARTITIONS 4) STORED AS KUDU"
    )
    range_component = kudu_property(tree).expressions[0]
    assert isinstance(range_component, KuduRangePartitioning)
    assert range_component.args["columns"] is None
    first, second = range_component.expressions
    assert [item.args["buckets"].this for item in first.args["hashes"]] == ["3", "2"]
    assert first.args["hashes"][1].args["columns"] is None
    assert second.args["hashes"][0].args["columns"][0].name == "id"
    round_trip(tree)


@pytest.mark.parametrize("kind,option", [("ADD", "IF NOT EXISTS"), ("DROP", "IF EXISTS")])
@pytest.mark.parametrize(
    "spec,lower,lower_inclusive,upper,upper_inclusive,canonical",
    [
        ("1 <= VALUES < 10", "1", True, "10", False, "1 <= VALUES < 10"),
        ("1 < VALUES <= 10", "1", False, "10", True, "1 < VALUES <= 10"),
        ("VALUES < 10", None, None, "10", False, "VALUES < 10"),
        ("VALUES <= 10", None, None, "10", True, "VALUES <= 10"),
        ("10 < VALUES", "10", False, None, None, "10 < VALUES"),
        ("10 <= VALUES", "10", True, None, None, "10 <= VALUES"),
        ("VALUES > 10", "10", False, None, None, "10 < VALUES"),
        ("VALUES >= 10", "10", True, None, None, "10 <= VALUES"),
        ("20 > VALUES", None, None, "20", False, "VALUES < 20"),
        ("20 >= VALUES", None, None, "20", True, "VALUES <= 20"),
        ("20 > VALUES > 10", "10", False, "20", False, "10 < VALUES < 20"),
        ("20 >= VALUES >= 10", "10", True, "20", True, "10 <= VALUES <= 20"),
        ("1 <= VALUES <= 1", "1", True, "1", True, "1 <= VALUES <= 1"),
    ],
)
def test_alter_range_bounds(kind, option, spec, lower, lower_inclusive, upper, upper_inclusive, canonical):
    tree = parse(f"ALTER TABLE db.t {kind} {option} RANGE PARTITION {spec}")
    action = alter_partition(tree)
    assert tree.this.db == "db"
    assert action.args["kind"] == kind
    assert action.args["exists"] is True
    partition = action.this
    for name, expected, inclusive in (
        ("lower", lower, lower_inclusive),
        ("upper", upper, upper_inclusive),
    ):
        bound = partition.args[name]
        if expected is None:
            assert bound is None
        else:
            assert isinstance(bound, KuduRangeBound)
            assert bound.this.this == expected
            assert bound.args["inclusive"] is inclusive
    assert round_trip(tree) == f"ALTER TABLE db.t {kind} {option} RANGE PARTITION {canonical}"


@pytest.mark.parametrize("kind", ["ADD", "DROP"])
@pytest.mark.parametrize("value", ["100", "(1, 'x')", "1 + 1", "(now())"])
def test_alter_equality_values_are_expressions(kind, value):
    tree = parse(f"ALTER TABLE t {kind} RANGE PARTITION VALUE = {value}")
    action = alter_partition(tree)
    assert action.args["exists"] is False
    assert action.this.args["value"] is not None
    assert action.this.args["lower"] is None
    assert action.this.args["upper"] is None
    round_trip(tree)


def test_alter_custom_hash_and_negative_bound():
    tree = parse(
        "ALTER TABLE t ADD RANGE PARTITION VALUES >= -10 "
        "HASH (id) PARTITIONS 3, HASH PARTITIONS 2"
    )
    partition = alter_partition(tree).this
    assert isinstance(partition.args["lower"].this, exp.Neg)
    assert [hash_.args["buckets"].this for hash_ in partition.args["hashes"]] == ["3", "2"]
    assert round_trip(tree) == (
        "ALTER TABLE t ADD RANGE PARTITION -10 <= VALUES "
        "HASH (id) PARTITIONS 3, HASH PARTITIONS 2"
    )


@pytest.mark.parametrize(
    "spec,bound_name",
    [("VALUES >= (1, 2)", "lower"), ("(1, 2) >= VALUES", "upper")],
)
def test_alter_reversed_tuple_bounds(spec, bound_name):
    tree = parse(f"ALTER TABLE t DROP RANGE PARTITION {spec}")
    bound = alter_partition(tree).this.args[bound_name]
    assert isinstance(bound.this, exp.Tuple)
    assert [item.this for item in bound.this.expressions] == ["1", "2"]
    assert bound.args["inclusive"] is True
    round_trip(tree)


@pytest.mark.parametrize("identifier", ["VALUE", "VALUES", "HASH", "RANGE", "PARTITION"])
def test_quoted_keyword_bound_remains_an_identifier(identifier):
    tree = parse(f"ALTER TABLE t ADD RANGE PARTITION `{identifier}` < VALUES")
    lower = alter_partition(tree).this.args["lower"]
    assert isinstance(lower.this, exp.Column)
    assert lower.this.name == identifier
    assert lower.this.this.args["quoted"] is True
    round_trip(tree)


def test_quoted_partition_keyword_can_be_a_range_column():
    tree = parse(
        "CREATE TABLE t (id INT) PARTITION BY RANGE (`PARTITION`) "
        "(PARTITION VALUE = 1)"
    )
    component = kudu_property(tree).expressions[0]
    assert component.args["columns"][0].name == "PARTITION"
    assert component.args["columns"][0].args["quoted"] is True
    round_trip(tree)


@pytest.mark.parametrize("count", ["0", "1", "-2", "+2", "2.5", "2e1", "'4'", "many", ""])
def test_invalid_hash_bucket_counts_raise(count):
    with pytest.raises(ParseError):
        parse(f"CREATE TABLE t (id INT) PARTITION BY HASH (id) PARTITIONS {count}")


@pytest.mark.parametrize(
    "component",
    [
        "HASH () PARTITIONS 2",
        "HASH (id,) PARTITIONS 2",
        "HASH (1) PARTITIONS 2",
        "HASH (id) PARTITIONS",
        "RANGE (id) ()",
        "RANGE (id) (PARTITION VALUE = 1), RANGE (id) (PARTITION VALUES < 2)",
        "HASH (id) PARTITIONS 2, RANGE (id) (PARTITION VALUE = 1), HASH PARTITIONS 2",
    ],
)
def test_invalid_partition_components_raise(component):
    with pytest.raises(ParseError):
        parse(f"CREATE TABLE t (id INT) PARTITION BY {component}")


@pytest.mark.parametrize(
    "spec",
    [
        "VALUES",
        "VALUES = 1",
        "VALUE < 1",
        "1 < VALUE < 2",
        "1 < VALUES > 2",
        "1 > VALUES < 2",
        "10 < VALUES < 1",
        "1 < VALUES <= 1",
        "VALUES <",
        "VALUE = ()",
        "VALUE = (1,)",
        "VALUE = 1 PURGE",
        "1 < VALUES < 2, PARTITION 2 < VALUES < 3",
    ],
)
def test_invalid_alter_ranges_raise(spec):
    with pytest.raises(ParseError):
        parse(f"ALTER TABLE t ADD RANGE PARTITION {spec}")


@pytest.mark.parametrize(
    "action",
    [
        "ADD IF EXISTS RANGE PARTITION VALUE = 1",
        "DROP IF NOT EXISTS RANGE PARTITION VALUE = 1",
    ],
)
def test_invalid_alter_range_existence_options_raise(action):
    with pytest.raises(ParseError):
        parse(f"ALTER TABLE t {action}")


def test_like_parquet_is_an_unresolved_schema_reference(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Parsing must not read a schema file or use the network")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(socket, "create_connection", forbidden)
        tree = parse(
            "CREATE EXTERNAL TABLE IF NOT EXISTS db.t "
            "LIKE PARQUET 's3a://example/schema.parquet' "
            "STORED AS PARQUET LOCATION '/synthetic/output'"
        )
        assert isinstance(tree, exp.Create)
        assert isinstance(tree.this, exp.Table)
        assert tree.this.db == "db"
        assert tree.args["exists"] is True
        assert not list(tree.find_all(exp.ColumnDef))
        like = tree.find(exp.LikeProperty)
        assert isinstance(like, exp.LikeProperty)
        assert isinstance(like.this, ParquetSchemaReference)
        assert like.this.this.is_string
        assert like.this.this.this == "s3a://example/schema.parquet"
        location = tree.find(exp.LocationProperty)
        assert location.this.this == "/synthetic/output"
        assert round_trip(tree) == (
            "CREATE EXTERNAL TABLE IF NOT EXISTS db.t "
            "LIKE PARQUET 's3a://example/schema.parquet' "
            "STORED AS PARQUET LOCATION '/synthetic/output'"
        )


def test_like_parquet_keeps_explicit_partition_columns_without_inventing_schema():
    tree = parse(
        "CREATE TABLE t LIKE PARQUET '/synthetic/file.parquet' "
        "PARTITIONED BY (p STRING) STORED AS PARQUET"
    )
    assert isinstance(tree.this, exp.Table)
    partitioning = tree.find(exp.PartitionedByProperty)
    assert isinstance(partitioning, exp.PartitionedByProperty)
    assert [column.name for column in partitioning.this.expressions] == ["p"]
    round_trip(tree)


@pytest.mark.parametrize("quoted", ["'/synthetic/file.parquet'", '"/synthetic/file.parquet"'])
def test_like_parquet_accepts_both_string_quoting_styles(quoted):
    tree = parse(f"CREATE TABLE t LIKE PARQUET {quoted}")
    assert tree.find(ParquetSchemaReference).this.this == "/synthetic/file.parquet"
    assert isinstance(tree.this, exp.Table)
    round_trip(tree)


def test_like_parquet_preserves_escaped_reference_and_table_options():
    tree = parse(
        r"CREATE TABLE t LIKE PARQUET 's3a://example/o\'brien.parquet' "
        "COMMENT 'synthetic' STORED AS PARQUET TBLPROPERTIES ('k'='v')"
    )
    reference = tree.find(ParquetSchemaReference)
    assert reference.this.this == "s3a://example/o'brien.parquet"
    assert isinstance(tree.find(exp.SchemaCommentProperty), exp.SchemaCommentProperty)
    assert isinstance(tree.find(exp.FileFormatProperty), exp.FileFormatProperty)
    assert isinstance(tree.this, exp.Table)
    round_trip(tree)


@pytest.mark.parametrize("reference", ["", "NULL", "12", "''", "/unquoted"])
def test_like_parquet_rejects_missing_or_non_string_reference(reference):
    with pytest.raises(ParseError):
        parse(f"CREATE TABLE t LIKE PARQUET {reference}")


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE t (id INT) LIKE PARQUET '/synthetic/file.parquet'",
        "CREATE TABLE t LIKE PARQUET '/synthetic/file.parquet' AS SELECT 1",
    ],
)
def test_like_parquet_cannot_be_combined_with_columns_or_ctas(sql):
    with pytest.raises(ParseError):
        parse(sql)


@pytest.mark.parametrize(
    "sql,node_type",
    [
        ("CREATE TABLE t (id INT) PARTITION BY HASH (id) PARTITIONS 2", KuduPartitioningProperty),
        ("ALTER TABLE t ADD RANGE PARTITION 1 <= VALUES < 10", KuduAlterRangePartition),
        ("CREATE TABLE t LIKE PARQUET '/synthetic/file.parquet'", ParquetSchemaReference),
    ],
)
def test_custom_nodes_survive_sqlglot_dump_and_load(sql, node_type):
    tree = parse(sql)
    loaded = exp.Expression.load(tree.dump())
    assert loaded == tree
    assert isinstance(loaded.find(node_type), node_type)
    round_trip(loaded)


def core_key_classes():
    from impala_sqlglot.syntax import (
        ImpalaSyntaxParserMixin,
        NonUniquePrimaryKey,
        NonUniquePrimaryKeyColumnConstraint,
    )
    assert issubclass(ImpalaParser, ImpalaSyntaxParserMixin)
    return NonUniquePrimaryKey, NonUniquePrimaryKeyColumnConstraint


@pytest.mark.parametrize(
    "definition,key_index",
    [
        ("id INT NON UNIQUE PRIMARY KEY, v STRING", 1),
        ("id INT, v STRING, NON UNIQUE PRIMARY KEY (id)", 0),
    ],
)
def test_parent_non_unique_key_composes_with_kudu_partitioning(definition, key_index):
    key_class = core_key_classes()[key_index]
    tree = parse(
        f"CREATE TABLE t ({definition}) PARTITION BY HASH (id) PARTITIONS 2 STORED AS KUDU"
    )
    key = tree.find(key_class)
    assert isinstance(key, key_class)
    assert key.args["unique"] is False
    assert isinstance(kudu_property(tree).expressions[0], KuduHashPartition)
    round_trip(tree)


def test_parent_non_unique_ctas_composes_after_property_wiring():
    key_class = core_key_classes()[0]
    assert "NON UNIQUE PRIMARY KEY" in _Parser.PROPERTY_PARSERS
    assert key_class in _Generator.PROPERTIES_LOCATION
    tree = parse(
        "CREATE TABLE t NON UNIQUE PRIMARY KEY (id) PARTITION BY HASH (id) PARTITIONS 2 "
        "STORED AS KUDU AS SELECT 1 AS id"
    )
    assert isinstance(tree, exp.Create)
    assert isinstance(tree.find(key_class), key_class)
    assert isinstance(kudu_property(tree).expressions[0], KuduHashPartition)
    round_trip(tree)


@pytest.mark.parametrize("error_level", [sqlglot.ErrorLevel.IMMEDIATE, sqlglot.ErrorLevel.RAISE])
def test_incomplete_partition_by_is_not_silently_dropped(error_level):
    with pytest.raises(ParseError):
        sqlglot.parse_one(
            "CREATE TABLE demo_table PRIMARY KEY (demo_key) PARTITION BY",
            read="impala", error_level=error_level,
        )


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE t (id INT) STORED AS PARQUET",
        "CREATE TABLE t LIKE other",
        "CREATE TABLE t LIKE `PARQUET`",
        "CREATE TABLE t (id INT) PARTITIONED BY (p STRING) STORED AS PARQUET",
        "CREATE TABLE t (id INT, PRIMARY KEY (id))",
        "CREATE TABLE t PRIMARY KEY (id)",
        "CREATE TABLE t PRIMARY KEY (id) AS SELECT 1 AS id",
        "ALTER TABLE t ADD COLUMN extra INT",
        "ALTER TABLE t DROP COLUMN extra",
    ],
)
def test_ordinary_ddl_still_uses_the_existing_parser_and_generator(sql):
    expected = parse_one(sql, read=Impala())
    actual = parse(sql)
    assert not isinstance(actual, exp.Command)
    assert actual == expected
    assert actual.sql(dialect=DIALECT) == expected.sql(dialect=Impala())
