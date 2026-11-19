"""Apache Impala SQL dialect for sqlglot.

Inherits from Hive (the closest shared ancestor) and overrides the parts
that differ.  Changes are organised by phase; see the plan at:
  .cursor/plans/impala_parser_coverage_*.plan.md

Phase 0 — Foundation:
  - Tokenizer: block Hive-only keywords/tokens (ADD JAR, $-params, numeric
    suffixes, MSCK REPAIR, wrong time-travel keywords)
  - Parser:   remove TRANSFORM … USING; block DISTRIBUTE/CLUSTER BY
  - Generator: remove ILike→LIKE and Unnest→EXPLODE rewrites; enable hints
"""
from __future__ import annotations

from sqlglot.dialects.hive import Hive

from impala_sqlglot.tokenizer import ImpalaTokenizer
from impala_sqlglot.parser import ImpalaParser
from impala_sqlglot.generator import ImpalaGenerator


class Impala(Hive):
    """SQLGlot dialect for Apache Impala."""

    # Native syntax places the alias before TABLESAMPLE, unlike Hive.
    ALIAS_POST_TABLESAMPLE = False

    # Impala identifiers are case-insensitive (same as Hive) but do NOT
    # start with $ like Hive parameters do, so we inherit the setting.

    Tokenizer = ImpalaTokenizer
    Parser = ImpalaParser
    Generator = ImpalaGenerator
