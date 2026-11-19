# Impala-specific syntax and analysis

The dialect preserves the following constructs as structured expressions
and regenerates Impala SQL:

| Construct | Expression / fields |
| --- | --- |
| UPSERT | `impala_sqlglot.syntax.Upsert`, a distinct subclass of `exp.Insert`, with target, source, columns and hint |
| Plan hints | `exp.Hint` on a SELECT, write, JOIN, table or inline view; `PlanHintProperty` for CREATE TABLE AS SELECT; bracket, block-comment and line-comment inputs generate comment hints |
| Hinted inline views | `ImpalaSubquery`, a subclass of `exp.Subquery` with `hints` |
| SELECT STRAIGHT_JOIN | `Select.operation_modifiers`; comment-form STRAIGHT_JOIN remains a hint |
| Time travel | `exp.Version`, generated as `FOR SYSTEM_TIME AS OF` or `FOR SYSTEM_VERSION AS OF` |
| TABLESAMPLE | `exp.TableSample` with `method=SYSTEM`, `percent` and optional `seed`, generated with `REPEATABLE` |
| Non-unique primary keys | `NonUniquePrimaryKey` or `NonUniquePrimaryKeyColumnConstraint`, with `unique=False` |
| Statistics | `ImpalaComputeStats` / `ImpalaDropStats`, with a table target and applicable incremental, partition, column-list and sampling options |
| Metadata | `ImpalaRefresh` / `ImpalaInvalidateMetadata`; refresh distinguishes tables, functions and authorization |
| Kudu partitioning | `KuduPartitioningProperty`, ordered hash/range components, hash columns/counts, range endpoints and inclusivity |
| ADD / DROP RANGE PARTITION | `KuduAlterRangePartition` actions inside `exp.Alter` |
| LIKE PARQUET | `ParquetSchemaReference` holding a string reference, not a resolved schema |
| Function signatures | `VariadicParameter` around the final type; scalar CREATE and DROP signatures are checked separately |
| DROP FUNCTION | `DropFunction`; `wrapped` distinguishes omitted signatures from `()` |

Syntax expression extensions are in `impala_sqlglot.syntax`, metadata
extensions in `impala_sqlglot.metadata`, and Kudu/schema extensions in
`impala_sqlglot.ddl`. Ordinary primary keys keep SQLGlot's existing types.
Targets remain ordinary `exp.Table` nodes where the syntax identifies a table.

For example:

```python
import sqlglot
import impala_sqlglot
from impala_sqlglot.syntax import Upsert

tree = sqlglot.parse_one(
    "UPSERT [SHUFFLE] INTO demo_output SELECT demo_key FROM demo_input",
    read="impala",
)
assert isinstance(tree, Upsert)
assert tree.this.name == "demo_output"
assert tree.sql(dialect="impala").startswith("UPSERT /*+ SHUFFLE */ INTO")
```

## Validation boundaries

- Statistics/metadata paths require the documented targets and reject
  unsupported option combinations. Targetless `INVALIDATE METADATA` is valid.
  `COMPUTE STATS ... ()` preserves an explicitly empty column list.
- Only incremental statistics accept `PARTITION`; incremental computation
  does not accept sampling. Incremental DROP requires a partition.
- Query/statistics sample percentages must be integers from 0 to 100.
  Seeds from 0 to 2^63-1 are the plugin's conservative integer policy, not an
  independently established engine limit. The public documentation says
  positive seeds; the pinned grammar accepts unsigned integer literals.
  Table aliases precede TABLESAMPLE; table hints follow sampling.
- An explicit write column permutation permits the native source-less form.
  Empty `()` is retained. Hints may be leading or trailing, not both.
  Empty plan hints are a no-op; ordinary multiline comments do not activate
  hints merely because a later line begins with `+`.
  Multiline block comments are not active plan hints, including those whose
  first line begins with `+`.
- `SYSTEM_VERSION` accepts numeric literals, not column references or
  arithmetic expressions. Acceptance by the parser does not establish that
  a snapshot exists.
- Hive `DISTRIBUTE BY`, `CLUSTER BY` and `SORT BY` query clauses are rejected.
  Ordinary ORDER BY and window PARTITION BY remain supported.
- Unsupported UNIQUE constraints are rejected by generation, not silently
  removed by Hive preprocessing. Unsupported write/sampling options are
  rejected rather than dropped or converted into a different operation.
- A schema-file reference is never opened or fetched. The dialect does not
  invent external columns, resolve primary keys or existing ranges, or
  validate data against a live catalog.
- Aggregate CREATE FUNCTION and unrelated legacy statements may still fall
  back to `exp.Command`. A Command preserves text; it is not detailed grammar
  validation or structured lineage.
- Generic SQLGlot optimizers, other dialect generators and lineage consumers
  are not automatically aware of these extension classes. Consumers must
  retain their fields or explicitly reject unsupported transformations.

## Reference and compatibility limits

Syntax expectations are based on Apache's public SQL reference and native
grammar blob `5424488331aba80f43853409c4d56eada20beb86`, with parser-test blob
`d6addf016e37cfb90d540b5eb7013f578e18e359`. The identifier checks in metadata
also consult public scanner blob `9e461a4d5416e7e0975c9378800a5e18bd0d66c1`;
its exact association with the grammar revision has not been established.
This is a pinned source reference, not a declared Impala release range.
In particular, NON UNIQUE PRIMARY KEY requires an engine supporting that
grammar. No server execution, engine-release, optimizer or performance
qualification has been performed.

SQLGlot 30.10.0 and 30.33.0 were checked locally using Python 3.13.16.
The project targets Python 3.11 and 3.12, but those interpreters were not
available for local qualification. The open-ended dependency requirement is
not a guarantee that every future SQLGlot version is compatible.

New regression SQL and implementation code use only public references and
independently authored synthetic examples.

## Running the regression tests

With the package and pytest installed in the chosen environment:

```bash
python -m pytest tests/test_impala_gaps.py tests/test_metadata.py \
  tests/test_kudu_ddl.py tests/test_generation_controls.py \
  tests/test_roundtrip_helper.py tests/test_fixture_controls.py
python -m pytest
```

The round-trip helper requires structural equality and correct hint
attachment; it does not swallow parser/generator failures. Expected lexical
and syntax rejections accept `TokenError` / `ParseError`, not arbitrary
runtime crashes. Some older extracted negative fixtures remain expected
failures because syntax and catalog/type analysis are not the same check.
Use SQLGlot's default immediate errors or `ErrorLevel.RAISE` for validation;
`IGNORE` and `WARN` may return partial trees and are not validation modes.
The generated `tests/coverage_matrix.yaml` is historical extraction metadata,
not a current proof of implemented syntax.
