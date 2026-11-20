"""Structured Impala statistics and metadata statements.

The dialect tokenizer must not classify COMPUTE STATS, COMPUTE INCREMENTAL
STATS or INVALIDATE METADATA as opaque COMMANDs. Remove those three keyword
entries so this mixin receives their individual tokens. Other commands can
retain their existing tokenizer and parser behavior.
"""
from __future__ import annotations

from sqlglot import exp
from sqlglot.tokens import TokenType
from impala_sqlglot.syntax import is_impala_unsigned_integer


class ImpalaComputeStats(exp.Expression):
    arg_types = {
        "this": True,
        "incremental": False,
        "partition": False,
        "columns": False,
        "sample": False,
    }


class ImpalaDropStats(exp.Expression):
    arg_types = {"this": True, "incremental": False, "partition": False}


class ImpalaRefresh(exp.Expression):
    # kind is TABLE, FUNCTIONS, or AUTHORIZATION. The target is a Table for
    # TABLE and an Identifier for FUNCTIONS; AUTHORIZATION has no target.
    arg_types = {"this": False, "kind": True, "partitions": False}


class ImpalaInvalidateMetadata(exp.Expression):
    arg_types = {"this": False}


# Spellings of the public Impala scanner's keyword terminals, except the
# grammar's explicitly unreserved DEFAULT, KILL, QUERY, UUID, VARIANT and
# GEOMETRY. SQLGlot treats many as identifiers, so token type is insufficient.
# Additional reserved words depend on the Impala scanner's version and its
# built-in function registry.
_IMPALA_RESERVED_IDENTIFIERS = frozenset(
    """
    ADD AGGREGATE ALL ALTER ANALYTIC AND ANTI API_VERSION ARRAY AS ASC AUTHORIZATION AVRO
    BETWEEN BIGINT BINARY BLOCK_SIZE BOOLEAN BOTH BUCKETS BY CACHED CASCADE CASE CAST
    CHANGE CHAR CLASS CLOSE_FN CLUSTERED COLUMN COLUMNS COMMENT COMPRESSION COMPUTE
    CONSTRAINT CONVERT COPY CREATE CROSS CUBE CURRENT DATA DATABASE DATABASES DATE
    DATETIME DECIMAL DELETE DELIMITED DESC DESCRIBE DISABLE DISTINCT DIV DOUBLE DROP
    ELSE ENABLE ENCODING END ENFORCED ESCAPED EXCEPT EXECUTE EXISTS EXPLAIN EXTENDED
    EXTERNAL FALSE FIELDS FILEFORMAT FILES FILE_SIZE_THRESHOLD_MB FINALIZE_FN FIRST
    FLOAT FOLLOWING FOR FOREIGN FORMAT FORMATTED FROM FULL FUNCTION FUNCTIONS GRANT
    GROUP GROUPING GROUPS HASH HUDIPARQUET IGNORE HAVING ICEBERG IF ILIKE IN
    INCREMENTAL INIT_FN INNER INPATH INSERT INT INTEGER INTERMEDIATE INTERSECT INTERVAL INTO
    INVALIDATE IREGEXP IS JDBC JOIN JSONFILE KUDU LAST LEADING LEFT LEXICAL LIKE LIMIT
    LINES LOAD LOCATION MANAGEDLOCATION MAP MATCHED MERGE MERGE_FN METADATA
    MINUS NON NORELY NOT NOVALIDATE NULL NULLS OF OFFSET ON OPTIMIZE OR ORC ORDER
    OUTER OVER OVERWRITE PARQUET PARQUETFILE PARTITION PARTITIONED PARTITIONS PIVOT
    PRECEDING PREPARE_FN PAIMON PRIMARY PRODUCED PURGE RANGE RCFILE REAL RECOVER REFERENCES
    REFRESH REGEXP RELY RENAME REPEATABLE REPLACE REPLICATION RESTRICT RETURNS REVOKE
    RIGHT RLIKE ROLE ROLES ROLLUP ROW ROWS RWSTORAGE SCHEMA SCHEMAS SELECT SELECTIVITY
    SEMI SEQUENCEFILE SERDEPROPERTIES SERIALIZE_FN SET SHOW SMALLINT SETS SORT SPEC
    STORAGEHANDLER_URI STORED STRAIGHT_JOIN STRING STRUCT SYMBOL SYSTEM_TIME
    SYSTEM_VERSION TABLE TABLES TABLESAMPLE TBLPROPERTIES TERMINATED TEXTFILE THEN
    TIMESTAMP TINYINT TRUNCATE STATS TO TRAILING TRUE USER_DEFINED_FN UNBOUNDED UNCACHED UNION
    UNIQUE UNKNOWN UNNEST UNPIVOT UNSET UPDATE UPDATE_FN UPSERT USE USING VALIDATE
    VALUES VARCHAR VIEW VIEWS WHEN WHERE WITH ZORDER
    """.split()
)

# The native scanner uses Java String.trim() for backtick identifiers.
_IMPALA_TRIM_CHARS = "".join(chr(code) for code in range(0x21))


def _trim_impala_identifier(text: str) -> str:
    return text.strip(_IMPALA_TRIM_CHARS)


_METADATA_STRING_TOKENS = frozenset(
    token_type for token_type in TokenType
    if token_type.name == "STRING" or token_type.name.endswith("_STRING")
)


def _metadata_word(token, word: str) -> bool:
    return bool(
        token
        and token.token_type != TokenType.IDENTIFIER
        and token.token_type not in _METADATA_STRING_TOKENS
        and token.text.upper() == word
    )


class ImpalaMetadataParserMixin:
    """Cooperative statement parser for the structured metadata statements."""

    def _parse_statement(self) -> exp.Expression | None:
        token = self._curr
        if _metadata_word(token, "COMPUTE"):
            parse = self._parse_metadata_compute_stats
        elif _metadata_word(token, "INVALIDATE"):
            parse = self._parse_metadata_invalidate
        elif _metadata_word(token, "REFRESH"):
            parse = self._parse_metadata_refresh
        elif _metadata_word(token, "DROP") and (
            _metadata_word(self._next, "STATS")
            or _metadata_word(self._next, "INCREMENTAL")
        ):
            parse = self._parse_metadata_drop_stats
        else:
            return super()._parse_statement()

        comments = list(token.comments)
        self._advance()
        self._prev_comments = []
        previous = getattr(self, "_impala_metadata_comments", None)
        self._impala_metadata_comments = []
        try:
            statement = parse()
            statement.add_comments(comments + self._impala_metadata_comments, prepend=True)
            return statement
        finally:
            self._impala_metadata_comments = previous

    def _metadata_capture_comments(self) -> None:
        self._impala_metadata_comments.extend(self._prev_comments)
        self._prev_comments = []

    def _metadata_match_word(self, word: str) -> bool:
        if _metadata_word(self._curr, word):
            self._advance()
            self._metadata_capture_comments()
            return True
        return False

    def _metadata_match_token(self, token_type: TokenType) -> bool:
        if self._match(token_type):
            self._metadata_capture_comments()
            return True
        return False

    def _metadata_expect_word(self, word: str) -> None:
        if not self._metadata_match_word(word):
            self.raise_error(f"Expected {word}")

    def _metadata_expect_token(self, token_type: TokenType, text: str) -> None:
        if not self._metadata_match_token(token_type):
            self.raise_error(f"Expected {text}")

    def _parse_metadata_identifier(self) -> exp.Identifier | None:
        token = self._curr
        if not token or token.token_type == TokenType.STRING:
            return None
        if token.token_type == TokenType.IDENTIFIER and not _trim_impala_identifier(token.text):
            return None
        if token.token_type != TokenType.IDENTIFIER:
            if token.text.upper() in _IMPALA_RESERVED_IDENTIFIERS:
                return None
            if token.token_type not in self.ID_VAR_TOKENS:
                return None
        identifier = self._parse_id_var(any_token=False)
        if isinstance(identifier, exp.Identifier) and identifier.args.get("quoted"):
            identifier.set("this", _trim_impala_identifier(identifier.name))
        return identifier if isinstance(identifier, exp.Identifier) else None

    def _parse_metadata_table(self) -> exp.Table | None:
        first = self._parse_metadata_identifier()
        if first is None:
            self.raise_error("Expected an Impala table name")
            return None

        if self._metadata_match_token(TokenType.DOT):
            second = self._parse_metadata_identifier()
            if second is None:
                self.raise_error("Expected a table name after the database")
            table = self.expression(exp.Table(this=second, db=first))
        else:
            table = self.expression(exp.Table(this=first))

        if self._curr and self._curr.token_type == TokenType.DOT:
            self.raise_error("Impala table names have at most two parts")
        return table

    def _parse_metadata_columns(self) -> list[exp.Identifier]:
        self._metadata_expect_token(TokenType.L_PAREN, "(")
        columns = []
        if self._metadata_match_token(TokenType.R_PAREN):
            return columns
        while True:
            column = self._parse_metadata_identifier()
            if column is None:
                self.raise_error("Expected a statistics column name")
                break
            columns.append(column)
            if not self._metadata_match_token(TokenType.COMMA):
                break
        self._metadata_expect_token(TokenType.R_PAREN, ")")
        return columns

    def _parse_metadata_partition(self, *, static: bool) -> exp.Partition:
        self._metadata_expect_word("PARTITION")
        self._metadata_expect_token(TokenType.L_PAREN, "(")
        expressions = []
        while True:
            if static:
                key = self._parse_metadata_identifier()
                if key is None:
                    self.raise_error("Expected a partition column name")
                    break
                if not self._metadata_match_token(TokenType.EQ):
                    self.raise_error("Expected = after a partition column")
                value = self._parse_disjunction()
                if value is None:
                    self.raise_error("Expected a partition value")
                item = self.expression(exp.EQ(this=key, expression=value))
            else:
                item = self._parse_disjunction()
                if item is None:
                    self.raise_error("Expected a partition expression")
                    break
            expressions.append(item)
            if not self._metadata_match_token(TokenType.COMMA):
                break
        self._metadata_expect_token(TokenType.R_PAREN, ")")
        return self.expression(exp.Partition(expressions=expressions))

    def _parse_metadata_integer(self) -> exp.Literal | None:
        token = self._curr
        if (
            not token
            or token.token_type != TokenType.NUMBER
            or not token.text.isascii()
            or not token.text.isdecimal()
        ):
            self.raise_error("Expected an unsigned integer")
            return None
        self._advance()
        return self.expression(
            exp.Literal(this=token.text, is_string=False), token=token
        )

    def _parse_metadata_sample(self) -> exp.TableSample:
        self._metadata_expect_word("TABLESAMPLE")
        if (
            self._curr
            and self._curr.token_type == TokenType.IDENTIFIER
            and _trim_impala_identifier(self._curr.text).upper() == "SYSTEM"
        ):
            self._advance()
            self._metadata_capture_comments()
        else:
            self._metadata_expect_word("SYSTEM")
        self._metadata_expect_token(TokenType.L_PAREN, "(")
        percent = self._parse_metadata_integer()
        if not is_impala_unsigned_integer(percent, 100):
            self.raise_error("Sample percentage must be an integer from 0 to 100")
        self._metadata_expect_token(TokenType.R_PAREN, ")")
        seed = None
        if self._metadata_match_word("REPEATABLE"):
            self._metadata_expect_token(TokenType.L_PAREN, "(")
            seed = self._parse_metadata_integer()
            if not is_impala_unsigned_integer(seed, 9223372036854775807):
                self.raise_error("Expected a non-negative 64-bit integer seed")
            self._metadata_expect_token(TokenType.R_PAREN, ")")
        return self.expression(
            exp.TableSample(
                method=exp.Var(this="SYSTEM"),
                percent=percent,
                seed=seed,
            )
        )

    def _parse_metadata_compute_stats(self) -> ImpalaComputeStats:
        incremental = self._metadata_match_word("INCREMENTAL")
        self._metadata_expect_word("STATS")
        table = self._parse_metadata_table()
        partition = None
        if _metadata_word(self._curr, "PARTITION"):
            if not incremental:
                self.raise_error("PARTITION requires COMPUTE INCREMENTAL STATS")
            partition = self._parse_metadata_partition(static=False)
        columns = (
            self._parse_metadata_columns()
            if self._curr and self._curr.token_type == TokenType.L_PAREN
            else None
        )
        sample = None
        if _metadata_word(self._curr, "TABLESAMPLE"):
            if incremental:
                self.raise_error("TABLESAMPLE is not valid with incremental statistics")
            sample = self._parse_metadata_sample()
        return self.expression(
            ImpalaComputeStats(
                this=table,
                incremental=incremental,
                partition=partition,
                columns=columns,
                sample=sample,
            )
        )

    def _parse_metadata_drop_stats(self) -> ImpalaDropStats:
        incremental = self._metadata_match_word("INCREMENTAL")
        self._metadata_expect_word("STATS")
        table = self._parse_metadata_table()
        partition = None
        if incremental:
            if not _metadata_word(self._curr, "PARTITION"):
                self.raise_error("DROP INCREMENTAL STATS requires a partition")
            else:
                partition = self._parse_metadata_partition(static=False)
        return self.expression(
            ImpalaDropStats(
                this=table,
                incremental=incremental,
                partition=partition,
            )
        )

    def _parse_metadata_refresh(self) -> ImpalaRefresh:
        if self._metadata_match_word("FUNCTIONS"):
            database = self._parse_metadata_identifier()
            if database is None:
                self.raise_error("REFRESH FUNCTIONS requires a database")
            return self.expression(ImpalaRefresh(this=database, kind="FUNCTIONS"))

        if self._metadata_match_word("AUTHORIZATION"):
            return self.expression(ImpalaRefresh(kind="AUTHORIZATION"))

        table = self._parse_metadata_table()
        partitions = []
        while _metadata_word(self._curr, "PARTITION"):
            partitions.append(self._parse_metadata_partition(static=True))
        return self.expression(
            ImpalaRefresh(this=table, kind="TABLE", partitions=partitions)
        )

    def _parse_metadata_invalidate(self) -> ImpalaInvalidateMetadata:
        self._metadata_expect_word("METADATA")
        table = self._parse_metadata_table() if self._curr else None
        return self.expression(ImpalaInvalidateMetadata(this=table))


class ImpalaMetadataGeneratorMixin:
    """Generator handlers registered through METADATA_TRANSFORMS."""

    METADATA_TRANSFORMS = {
        ImpalaComputeStats: lambda self, e: self.impala_compute_stats_sql(e),
        ImpalaDropStats: lambda self, e: self.impala_drop_stats_sql(e),
        ImpalaRefresh: lambda self, e: self.impala_refresh_sql(e),
        ImpalaInvalidateMetadata: lambda self, e: self.impala_invalidate_metadata_sql(e),
    }

    def _metadata_check_identifier(self, identifier: exp.Identifier) -> None:
        if not isinstance(identifier, exp.Identifier) or not identifier.name:
            raise ValueError("Expected an Impala identifier")
        if (
            identifier.args.get("quoted")
            and identifier.name != _trim_impala_identifier(identifier.name)
        ):
            raise ValueError("Impala quoted identifiers cannot have outer whitespace")
        if (
            not identifier.args.get("quoted")
            and identifier.name.upper() in _IMPALA_RESERVED_IDENTIFIERS
        ):
            raise ValueError("Impala reserved identifiers must be quoted")

    def _metadata_table_sql(self, table: exp.Table | None) -> str:
        if (
            not isinstance(table, exp.Table)
            or not isinstance(table.this, exp.Identifier)
            or (table.args.get("db") and not isinstance(table.args["db"], exp.Identifier))
            or any(
                value is not None
                for key, value in table.args.items()
                if key not in ("this", "db")
            )
        ):
            raise ValueError("Expected an unaliased Impala table name")
        self._metadata_check_identifier(table.this)
        if table.args.get("db") is not None:
            self._metadata_check_identifier(table.args["db"])
        return self.sql(table)

    def _metadata_partition_sql(self, partition: exp.Partition, *, static: bool) -> str:
        if (
            not isinstance(partition, exp.Partition)
            or not partition.expressions
            or partition.args.get("subpartition")
        ):
            raise ValueError("A partition must contain at least one expression")
        if static and any(
            not isinstance(item, exp.EQ)
            or not isinstance(item.this, exp.Identifier)
            or item.expression is None
            for item in partition.expressions
        ):
            raise ValueError("REFRESH partitions require column = value pairs")
        if static:
            for item in partition.expressions:
                self._metadata_check_identifier(item.this)
        return "PARTITION (" + ", ".join(self.sql(item) for item in partition.expressions) + ")"

    def _metadata_sample_sql(self, sample: exp.TableSample) -> str:
        if not isinstance(sample, exp.TableSample):
            raise ValueError("Expected a TABLESAMPLE option")
        method = sample.args.get("method")
        percent = sample.args.get("percent")
        seed = sample.args.get("seed")
        if any(
            value is not None and value != []
            for key, value in sample.args.items()
            if key not in ("method", "percent", "seed")
        ):
            raise ValueError("Unsupported statistics sampling option")
        if not isinstance(method, exp.Var) or method.name.upper() != "SYSTEM":
            raise ValueError("Statistics sampling requires the SYSTEM method")
        for value in (percent, seed):
            if value is not None and (
                not isinstance(value, exp.Literal)
                or value.args.get("is_string")
                or not isinstance(value.this, str)
                or not value.this.isascii()
                or not value.this.isdecimal()
            ):
                raise ValueError("Statistics sampling requires unsigned integers")
        if percent is None:
            raise ValueError("TABLESAMPLE requires a percentage")
        if not is_impala_unsigned_integer(percent, 100):
            raise ValueError("Sample percentage must be an integer from 0 to 100")
        if seed is not None and not is_impala_unsigned_integer(seed, 9223372036854775807):
            raise ValueError("Expected a non-negative 64-bit integer seed")
        sql = f"TABLESAMPLE SYSTEM({self.sql(percent)})"
        if seed is not None:
            sql += f" REPEATABLE({self.sql(seed)})"
        return sql

    def impala_compute_stats_sql(self, expression: ImpalaComputeStats) -> str:
        incremental = bool(expression.args.get("incremental"))
        partition = expression.args.get("partition")
        columns = expression.args.get("columns")
        sample = expression.args.get("sample")
        if partition is not None and not incremental:
            raise ValueError("PARTITION requires COMPUTE INCREMENTAL STATS")
        if sample is not None and incremental:
            raise ValueError("TABLESAMPLE is not valid with incremental statistics")
        sql = "COMPUTE " + ("INCREMENTAL " if incremental else "") + "STATS "
        sql += self._metadata_table_sql(expression.this)
        if partition is not None:
            sql += " " + self._metadata_partition_sql(partition, static=False)
        if columns is not None:
            if any(not isinstance(column, exp.Identifier) for column in columns):
                raise ValueError("Statistics columns must be identifiers")
            for column in columns:
                self._metadata_check_identifier(column)
            sql += " (" + ", ".join(self.sql(column) for column in columns) + ")"
        if sample is not None:
            sql += " " + self._metadata_sample_sql(sample)
        return sql

    def impala_drop_stats_sql(self, expression: ImpalaDropStats) -> str:
        incremental = bool(expression.args.get("incremental"))
        partition = expression.args.get("partition")
        if incremental != (partition is not None):
            raise ValueError("Only DROP INCREMENTAL STATS requires a partition")
        sql = "DROP " + ("INCREMENTAL " if incremental else "") + "STATS "
        sql += self._metadata_table_sql(expression.this)
        if partition is not None:
            sql += " " + self._metadata_partition_sql(partition, static=False)
        return sql

    def impala_refresh_sql(self, expression: ImpalaRefresh) -> str:
        kind = expression.args.get("kind")
        partitions = expression.args.get("partitions") or []
        if kind == "AUTHORIZATION" and expression.this is None and not partitions:
            return "REFRESH AUTHORIZATION"
        if kind == "FUNCTIONS" and isinstance(expression.this, exp.Identifier) and not partitions:
            self._metadata_check_identifier(expression.this)
            return "REFRESH FUNCTIONS " + self.sql(expression.this)
        if kind != "TABLE":
            raise ValueError("Invalid REFRESH target or options")
        sql = "REFRESH " + self._metadata_table_sql(expression.this)
        for partition in partitions:
            sql += " " + self._metadata_partition_sql(partition, static=True)
        return sql

    def impala_invalidate_metadata_sql(self, expression: ImpalaInvalidateMetadata) -> str:
        if expression.this is None:
            return "INVALIDATE METADATA"
        return "INVALIDATE METADATA " + self._metadata_table_sql(expression.this)
