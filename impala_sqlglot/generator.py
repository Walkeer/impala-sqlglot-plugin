"""Impala Generator — extends Hive, removes Hive-only generation transforms."""
from __future__ import annotations

from sqlglot import exp
from sqlglot.generators.hive import HiveGenerator
from sqlglot.transforms import (
    preprocess, ctas_with_tmp_tables_to_create_tmp_view,
    move_schema_columns_to_partitioned_by,
)
from impala_sqlglot.syntax import (
    ImpalaSyntaxGeneratorMixin, Upsert, NonUniquePrimaryKey,
    NonUniquePrimaryKeyColumnConstraint, VariadicParameter, DropFunction,
    ImpalaSubquery, PlanHintProperty, ImpalaPrimaryKey, ImpalaForeignKey,
)
from impala_sqlglot.metadata import ImpalaMetadataGeneratorMixin
from impala_sqlglot.ddl import ImpalaDDLGeneratorMixin
from impala_sqlglot.compat_generation import ImpalaGenerationGeneratorMixin
from impala_sqlglot.show import ImpalaShowGeneratorMixin


class ImpalaGenerator(
    ImpalaShowGeneratorMixin, ImpalaGenerationGeneratorMixin, ImpalaMetadataGeneratorMixin,
    ImpalaDDLGeneratorMixin, ImpalaSyntaxGeneratorMixin, HiveGenerator,
):
    """Impala SQL generator.

    Phase 0 changes:
    - Remove exp.ILike → LIKE rewrite (Impala supports ILIKE natively)
    - Remove exp.Unnest → EXPLODE rewrite (Impala uses UNNEST())
    - Enable JOIN/QUERY/TABLE hints

    Cooperative mixins render Impala writes, hints, metadata and Kudu syntax.
    """

    # Impala natively supports plan hints
    JOIN_HINTS: bool = True
    TABLE_HINTS: bool = True
    QUERY_HINTS: bool = True

    PROPERTIES_LOCATION = {
        **HiveGenerator.PROPERTIES_LOCATION,
        **ImpalaDDLGeneratorMixin.DDL_PROPERTY_LOCATIONS,
        ImpalaPrimaryKey: exp.Properties.Location.POST_SCHEMA,
        NonUniquePrimaryKey: exp.Properties.Location.POST_SCHEMA,
        PlanHintProperty: exp.Properties.Location.POST_CREATE,
    }

    TRANSFORMS = {
        ImpalaPrimaryKey: lambda self, e: self.impalaprimarykey_sql(e),
        ImpalaForeignKey: lambda self, e: self.impalaforeignkey_sql(e),
        Upsert: lambda self, e: self.upsert_sql(e),
        NonUniquePrimaryKey: lambda self, e: self.nonuniqueprimarykey_sql(e),
        NonUniquePrimaryKeyColumnConstraint: lambda self, e: self.nonuniqueprimarykeycolumnconstraint_sql(e),
        VariadicParameter: lambda self, e: self.variadicparameter_sql(e),
        DropFunction: lambda self, e: self.dropfunction_sql(e),
        ImpalaSubquery: lambda self, e: self.impalasubquery_sql(e),
        PlanHintProperty: lambda self, e: self.planhintproperty_sql(e),
        **ImpalaMetadataGeneratorMixin.METADATA_TRANSFORMS,
        **ImpalaDDLGeneratorMixin.DDL_TRANSFORMS,
        **ImpalaGenerationGeneratorMixin.GENERATION_TRANSFORMS,
        **ImpalaShowGeneratorMixin.SHOW_TRANSFORMS,
        **{
            k: v
            for k, v in HiveGenerator.TRANSFORMS.items()
            # Block Hive transforms that are wrong for Impala
            if k
            not in (
                exp.Create,
                exp.ILike,      # Hive rewrites ILIKE → LIKE; Impala supports ILIKE
                exp.Unnest,     # Hive rewrites UNNEST → EXPLODE; Impala uses UNNEST()
            )
        },
        exp.Create: preprocess([
            ctas_with_tmp_tables_to_create_tmp_view,
            move_schema_columns_to_partitioned_by,
        ]),
    }
