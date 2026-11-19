"""Structured Impala ALTER TABLE maintenance actions.

Place the parser and generator mixins before their Kudu DDL counterparts.
Merge MAINTENANCE_TRANSFORMS into the dialect generator's TRANSFORMS. The
metadata mixins provide the shared table-name and partition helpers.

These nodes describe SQL; they do not access a catalog, filesystem or cache.
The grammar accepts an unsigned decimal replication literal. Whether its
value, the pool, the table and the selected partitions are usable is left to
Impala's semantic analysis.
"""

from __future__ import annotations

from sqlglot import exp
from sqlglot.errors import TokenError
from sqlglot.tokens import TokenType


class ImpalaRecoverPartitions(exp.Expression):
    """Recover partition metadata for the ALTER target."""

    arg_types = {}


class ImpalaDropPartition(exp.Expression):
    """Drop the partitions selected by ``this`` from the ALTER target."""

    arg_types = {"this": True, "exists": False, "purge": False}


class ImpalaSetCached(exp.Expression):
    """Set cache metadata, optionally for selected partitions."""

    arg_types = {"this": True, "replication": False, "partition": False}


class ImpalaSetUncached(exp.Expression):
    """Clear cache metadata, optionally for selected partitions."""

    arg_types = {"partition": False}


_MAINTENANCE_ACTIONS = (
    ImpalaRecoverPartitions,
    ImpalaDropPartition,
    ImpalaSetCached,
    ImpalaSetUncached,
)


class ImpalaMaintenanceParserMixin:
    """Intercept only recognized maintenance ALTER TABLE forms."""

    def _maintenance_word(self, token, word: str) -> bool:
        return bool(
            token
            and token.token_type != TokenType.IDENTIFIER
            and token.token_type not in self.STRING_PARSERS
            and token.text.upper() == word
        )

    def _starts_maintenance_alter(self) -> bool:
        tokens = self._tokens
        start = self._index
        if start >= len(tokens) or not self._maintenance_word(tokens[start], "TABLE"):
            return False

        def word(index: int, text: str) -> bool:
            return index < len(tokens) and self._maintenance_word(tokens[index], text)

        def starts_action(index: int) -> bool:
            return any(
                word(index, action)
                for action in ("RECOVER", "DROP", "SET", "PARTITION", "CACHED", "UNCACHED")
            )

        def starts_cache(index: int) -> bool:
            if word(index, "SET"):
                index += 1
                while word(index, "SET"):
                    index += 1
            return word(index, "CACHED") or word(index, "UNCACHED")

        # Look only immediately after the target (including a malformed
        # qualified target). Do not scan through an unrelated ALTER action.
        index = start + 1
        while any(word(index, flag) for flag in ("IF", "NOT", "EXISTS", "ONLY")):
            index += 1
        if index >= len(tokens):
            return False
        if not starts_action(index):
            if tokens[index].token_type in (TokenType.L_PAREN, TokenType.R_PAREN, TokenType.COMMA):
                return False
            index += 1
            while index < len(tokens) and tokens[index].token_type == TokenType.DOT:
                index += 1
                if index >= len(tokens) or starts_action(index):
                    break
                if tokens[index].token_type != TokenType.DOT:
                    index += 1

        if word(index, "RECOVER") or word(index, "CACHED") or word(index, "UNCACHED"):
            return True
        if word(index, "DROP"):
            index += 1
            while any(word(index, flag) for flag in ("IF", "NOT", "EXISTS", "PURGE")):
                index += 1
            return word(index, "PARTITION")
        if word(index, "SET"):
            return starts_cache(index)
        if word(index, "PARTITION"):
            # Partition selectors precede SET. Recognize repeated selectors
            # and a missing closing parenthesis as malformed cache forms.
            while word(index, "PARTITION"):
                index += 1
                beginning = index
                if index >= len(tokens) or tokens[index].token_type != TokenType.L_PAREN:
                    return any(starts_cache(pos) for pos in range(index, len(tokens)))
                depth = 0
                while index < len(tokens):
                    token_type = tokens[index].token_type
                    if token_type == TokenType.L_PAREN:
                        depth += 1
                    elif token_type == TokenType.R_PAREN:
                        depth -= 1
                        if depth == 0:
                            index += 1
                            break
                    index += 1
                if depth:
                    return any(starts_cache(pos) for pos in range(beginning, len(tokens)))
            return starts_cache(index)
        return False

    def _parse_alter(self):
        if not self._starts_maintenance_alter():
            return super()._parse_alter()

        previous = getattr(self, "_impala_metadata_comments", None)
        self._impala_metadata_comments = []
        try:
            self._metadata_expect_word("TABLE")
            table = self._parse_metadata_table()

            if self._metadata_match_word("RECOVER"):
                self._metadata_expect_word("PARTITIONS")
                action = self.expression(ImpalaRecoverPartitions())
            elif self._metadata_match_word("DROP"):
                exists = self._metadata_match_word("IF")
                if exists:
                    self._metadata_expect_word("EXISTS")
                partition = self._parse_metadata_partition(static=False)
                purge = self._metadata_match_word("PURGE")
                action = self.expression(
                    ImpalaDropPartition(this=partition, exists=exists, purge=purge)
                )
            else:
                partition = None
                if self._maintenance_word(self._curr, "PARTITION"):
                    partition = self._parse_metadata_partition(static=False)
                self._metadata_expect_word("SET")
                if self._metadata_match_word("CACHED"):
                    self._metadata_expect_word("IN")
                    if not self._curr or self._curr.token_type != TokenType.STRING:
                        self.raise_error("CACHED IN requires a string pool name")
                        pool = None
                    else:
                        pool = self._parse_string()
                    replication = None
                    if self._metadata_match_word("WITH"):
                        self._metadata_expect_word("REPLICATION")
                        self._metadata_expect_token(TokenType.EQ, "=")
                        replication = self._parse_metadata_integer()
                    action = self.expression(
                        ImpalaSetCached(this=pool, replication=replication, partition=partition)
                    )
                else:
                    self._metadata_expect_word("UNCACHED")
                    action = self.expression(ImpalaSetUncached(partition=partition))

            if self._curr:
                self.raise_error("Unexpected input after ALTER TABLE maintenance action")
            statement = self.expression(exp.Alter(this=table, kind="TABLE", actions=[action]))
            statement.add_comments(self._impala_metadata_comments)
            return statement
        finally:
            self._impala_metadata_comments = previous


def _check_args(expression: exp.Expression, allowed: set[str]) -> None:
    extra = set(expression.args) - allowed
    if extra:
        names = sorted(map(str, extra))
        raise ValueError(f"Unsupported {type(expression).__name__} fields: {names}")


def _check_flag(expression: exp.Expression, key: str) -> bool:
    value = expression.args.get(key)
    if value is not None and type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return bool(value)


def _check_literal(value, *, string: bool) -> None:
    if (
        not isinstance(value, exp.Literal)
        or value.args.get("is_string") is not string
        or not isinstance(value.this, str)
    ):
        raise ValueError("Expected a string pool" if string else "Expected an unsigned integer")
    _check_args(value, {"this", "is_string"})
    if not string and (not value.this.isascii() or not value.this.isdecimal()):
        raise ValueError("Replication must be an unsigned decimal integer")


class ImpalaMaintenanceGeneratorMixin:
    """Render and validate maintenance actions without changing other ALTERs."""

    MAINTENANCE_TRANSFORMS = {
        ImpalaRecoverPartitions: lambda self, e: self.impala_recover_partitions_sql(e),
        ImpalaDropPartition: lambda self, e: self.impala_drop_partition_sql(e),
        ImpalaSetCached: lambda self, e: self.impala_set_cached_sql(e),
        ImpalaSetUncached: lambda self, e: self.impala_set_uncached_sql(e),
    }

    def _maintenance_table_sql(self, table: exp.Table) -> str:
        if not isinstance(table, exp.Table):
            raise ValueError("ALTER TABLE requires a table target")
        for key, value in table.args.items():
            if key not in ("this", "db") and (key not in table.arg_types or value is not None):
                raise ValueError(f"Unsupported table target field: {key}")
        for identifier in (table.this, table.args.get("db")):
            if identifier is None:
                continue
            if not isinstance(identifier, exp.Identifier):
                raise ValueError("Expected an Impala table identifier")
            _check_args(identifier, {"this", "quoted"})
            quoted = identifier.args.get("quoted")
            if quoted is not None and type(quoted) is not bool:
                raise ValueError("Identifier quoting must be a boolean")
            if not isinstance(identifier.this, str):
                raise ValueError("An Impala table identifier must be text")
            spelling = self.sql(identifier) if quoted else identifier.this
            try:
                tokens = self.dialect.tokenize(spelling)
            except TokenError as error:
                raise ValueError("Invalid Impala table identifier") from error
            if (
                len(tokens) != 1
                or tokens[0].text != identifier.this
                or (
                    tokens[0].token_type != TokenType.IDENTIFIER
                    if quoted
                    else tokens[0].token_type not in self.dialect.parser_class.ID_VAR_TOKENS
                )
            ):
                raise ValueError("Invalid Impala table identifier")
        return self._metadata_table_sql(table)

    def _maintenance_partition_sql(self, partition: exp.Partition) -> str:
        if not isinstance(partition, exp.Partition):
            raise ValueError("Expected a partition selector")
        _check_args(partition, {"expressions", "subpartition"})
        items = partition.args.get("expressions")
        if (
            not isinstance(items, list)
            or not items
            or partition.args.get("subpartition") is not None
            or any(not isinstance(item, exp.Expression) for item in items)
        ):
            raise ValueError("A partition selector requires expressions")
        for item in items:
            if not isinstance(item, (exp.Condition, exp.Subquery)):
                raise ValueError("A partition selector requires value expressions")
            # A subquery may contain tables and aliases, but those wrappers
            # are not themselves partition predicate operands.
            for node in item.walk(prune=lambda value: isinstance(value, exp.Query)):
                if isinstance(node, exp.Subquery):
                    if not isinstance(node.this, exp.Query) or node.args.get("alias") is not None:
                        raise ValueError("Invalid scalar subquery")
                elif isinstance(node, exp.Query):
                    if not isinstance(node.parent, (exp.Exists, exp.In)):
                        raise ValueError("A query must be used as a subquery")
                elif (
                    isinstance(node, (exp.Identifier, exp.Var))
                    and isinstance(node.parent, (exp.Unary, exp.Binary, exp.In, exp.Between))
                    and not isinstance(node.parent, exp.Dot)
                ):
                    raise ValueError("A bare identifier is not a value expression")
                elif isinstance(
                    node,
                    (
                        exp.Alias,
                        exp.Table,
                        exp.Where,
                        exp.Partition,
                        exp.Version,
                        exp.Command,
                        exp.DDL,
                        exp.DML,
                        exp.Alter,
                        exp.Drop,
                        exp.Tuple,
                        exp.Limit,
                        exp.Offset,
                    ),
                ):
                    raise ValueError("Invalid partition predicate operand")
            for node in item.walk():
                _check_args(node, set(node.arg_types))
                if node.error_messages():
                    raise ValueError("Invalid partition expression")
        return self.maybe_comment(
            self._metadata_partition_sql(partition, static=False), partition
        )

    def alter_sql(self, expression: exp.Alter) -> str:
        actions = expression.args.get("actions")
        candidates = actions if isinstance(actions, (list, tuple)) else (actions,)
        if not any(isinstance(action, _MAINTENANCE_ACTIONS) for action in candidates):
            return super().alter_sql(expression)

        if (
            not isinstance(actions, list)
            or len(actions) != 1
            or expression.args.get("kind") != "TABLE"
        ):
            raise ValueError("A maintenance ALTER requires one TABLE action")
        for key, value in expression.args.items():
            if key in ("this", "kind", "actions"):
                continue
            if key == "options" and (value is None or value == []):
                continue
            if key == "cluster" and value is None:
                continue
            if key in ("exists", "only", "not_valid", "check", "cascade", "iceberg") and (
                value is None or value is False
            ):
                continue
            raise ValueError(f"Unsupported maintenance ALTER field: {key}")
        self._maintenance_table_sql(expression.this)
        return super().alter_sql(expression)

    def impala_recover_partitions_sql(self, expression: ImpalaRecoverPartitions) -> str:
        _check_args(expression, set())
        return "RECOVER PARTITIONS"

    def impala_drop_partition_sql(self, expression: ImpalaDropPartition) -> str:
        _check_args(expression, {"this", "exists", "purge"})
        exists = " IF EXISTS" if _check_flag(expression, "exists") else ""
        purge = " PURGE" if _check_flag(expression, "purge") else ""
        return f"DROP{exists} {self._maintenance_partition_sql(expression.this)}{purge}"

    def impala_set_cached_sql(self, expression: ImpalaSetCached) -> str:
        _check_args(expression, {"this", "replication", "partition"})
        pool = expression.this
        _check_literal(pool, string=True)
        partition = expression.args.get("partition")
        prefix = self._maintenance_partition_sql(partition) + " " if partition is not None else ""
        sql = f"{prefix}SET CACHED IN {self.sql(pool)}"
        replication = expression.args.get("replication")
        if replication is not None:
            _check_literal(replication, string=False)
            sql += f" WITH REPLICATION = {self.sql(replication)}"
        return sql

    def impala_set_uncached_sql(self, expression: ImpalaSetUncached) -> str:
        _check_args(expression, {"partition"})
        partition = expression.args.get("partition")
        prefix = self._maintenance_partition_sql(partition) + " " if partition is not None else ""
        return f"{prefix}SET UNCACHED"
