"""Bounded structured support for native Impala SHOW statements.

The tokenizer must remove SHOW from COMMANDS and the parser must dispatch
TokenType.SHOW to _parse_show. Other SHOW families remain opaque commands.
The metadata parser and generator mixins provide the identifier helpers.
"""
from __future__ import annotations

import re

from sqlglot import exp
from sqlglot.errors import TokenError
from sqlglot.tokens import TokenType

from impala_sqlglot.metadata import _metadata_word


class ImpalaShow(exp.Expression):
    # SCHEMAS normalizes to DATABASES. Omitted options remain absent.
    arg_types = {
        "kind": True, "this": False, "db": False, "pattern": False,
        "partition": False, "where": False, "version": False,
        "function_kind": False, "with_stats": False,
    }


class ImpalaShowParserMixin:
    """Parse only the selected SHOW families; delegate other families."""

    def _parse_show(self) -> exp.Expression:
        start = self._prev
        body_start = self._curr
        if not self._curr:
            self.raise_error("Expected a SHOW family")
            return self.expression(ImpalaShow(kind=""))

        word = self._curr.text.upper()
        create = _metadata_word(self._curr, "CREATE")
        create_object = create and any(
            _metadata_word(self._next, name) for name in ("TABLE", "VIEW")
        )

        if _metadata_word(self._curr, "SCALAR") and _metadata_word(self._next, "FUNCTIONS"):
            self.raise_error("SHOW FUNCTIONS does not accept SCALAR")
        if create and not self._next:
            self.raise_error("Expected a SHOW CREATE object type")
            return self.expression(ImpalaShow(kind=""))
        if not (
            create_object
            or any(
                _metadata_word(self._curr, name)
                for name in (
                    "TABLES", "DATABASES", "SCHEMAS", "TABLE", "COLUMN",
                    "PARTITIONS", "RANGE", "HASH", "FUNCTIONS", "AGGREGATE",
                    "ANALYTIC", "FILES",
                )
            )
        ):
            return self._show_opaque_command(start, body_start, use_super=True)

        # The metadata helpers capture comments on consumed keywords and dots.
        previous = getattr(self, "_impala_metadata_comments", None)
        self._impala_metadata_comments = []
        self._prev_comments = []
        try:
            if create_object:
                self._metadata_expect_word("CREATE")
                object_type = self._curr.text.upper()
                self._metadata_expect_word(object_type)
                table = self._parse_metadata_table()
                options = {}
                if self._metadata_match_word("WITH"):
                    self._metadata_expect_word("STATS")
                    options["with_stats"] = True
                result = ImpalaShow(kind="CREATE " + object_type, this=table, **options)
            elif word in ("TABLES", "DATABASES", "SCHEMAS", "FUNCTIONS", "AGGREGATE", "ANALYTIC"):
                function_kind = None
                if word in ("AGGREGATE", "ANALYTIC"):
                    function_kind = word
                    self._metadata_expect_word(word)
                    self._metadata_expect_word("FUNCTIONS")
                    kind = "FUNCTIONS"
                else:
                    self._metadata_expect_word(word)
                    kind = "DATABASES" if word == "SCHEMAS" else word
                db = None
                if kind in ("TABLES", "FUNCTIONS") and self._metadata_match_word("IN"):
                    db = self._parse_metadata_identifier()
                    if db is None:
                        self.raise_error("Expected a database after IN")
                like = self._metadata_match_word("LIKE")
                pattern = None
                if self._curr and self._curr.token_type == TokenType.STRING:
                    pattern = self._parse_string()
                elif like:
                    self.raise_error("Expected a string pattern after LIKE")
                result = ImpalaShow(kind=kind, db=db, pattern=pattern)
                if function_kind is not None:
                    result.set("function_kind", function_kind)
            else:
                self._metadata_expect_word(word)
                if word in ("TABLE", "COLUMN"):
                    self._metadata_expect_word("STATS")
                    kind = word + " STATS"
                elif word == "RANGE":
                    self._metadata_expect_word("PARTITIONS")
                    kind = "RANGE PARTITIONS"
                elif word == "HASH":
                    self._metadata_expect_word("SCHEMA")
                    kind = "HASH SCHEMA"
                else:
                    kind = word
                if kind == "FILES":
                    self._metadata_expect_word("IN")
                table = self._parse_metadata_table()
                result = ImpalaShow(kind=kind, this=table)
                if kind == "PARTITIONS" and self._metadata_match_word("WHERE"):
                    predicate = self._parse_disjunction()
                    if predicate is None:
                        self.raise_error("Expected an expression after WHERE")
                    result.set("where", predicate)
                if kind == "FILES":
                    if self._metadata_match_word("FOR"):
                        if self._metadata_match_word("SYSTEM_TIME"):
                            version_kind = "TIMESTAMP"
                        elif self._metadata_match_word("SYSTEM_VERSION"):
                            version_kind = "VERSION"
                        else:
                            self.raise_error("Expected SYSTEM_TIME or SYSTEM_VERSION")
                            version_kind = ""
                        self._metadata_expect_word("AS")
                        self._metadata_expect_word("OF")
                        if version_kind == "VERSION":
                            if not self._curr or self._curr.token_type != TokenType.NUMBER:
                                self.raise_error("SYSTEM_VERSION requires a numeric literal")
                                value = None
                            else:
                                value = self._parse_number()
                        else:
                            value = self._parse_disjunction()
                        if value is None:
                            self.raise_error("Expected a time-travel value")
                        result.set("version", self.expression(
                            exp.Version(this=version_kind, expression=value, kind="AS OF")
                        ))
                    if _metadata_word(self._curr, "PARTITION"):
                        result.set("partition", self._parse_metadata_partition(static=False))

            if self._curr:
                self.raise_error("Unexpected token in SHOW statement")
            result = self.expression(result)
            result.add_comments(self._impala_metadata_comments, prepend=True)
            return result
        finally:
            self._impala_metadata_comments = previous

    def _show_opaque_command(self, start, body_start, *, use_super: bool) -> exp.Command:
        command = super()._parse_show() if use_super else self._parse_as_command(start)
        # SQLGlot's opaque fallback includes the comments immediately after
        # SHOW in the raw command while its statement dispatcher also attaches
        # them to the node. Remove that raw gap when it contains comments.
        gap = self.sql[start.end + 1 : body_start.start]
        if "/*" in gap or "--" in gap:
            command.set("expression", " " + self.sql[body_start.start : self._prev.end + 1])
        # The raw SQL slice ends at the last token, before its trailing comments.
        command.add_comments(self._prev.comments)
        self._prev_comments = []
        return command


class ImpalaShowGeneratorMixin:
    """Render the bounded SHOW node using the metadata identifier checks."""

    SHOW_TRANSFORMS = {ImpalaShow: lambda self, e: self.impala_show_sql(e)}

    @staticmethod
    def _show_value_expression(value) -> bool:
        if (
            not isinstance(value, exp.Expression)
            or isinstance(value, (exp.Identifier, exp.Command, exp.Where, exp.Table, exp.Partition, exp.Version))
            or (isinstance(value, exp.Query) and not isinstance(value, exp.Subquery))
        ):
            return False
        pending = [(None, value)]
        while pending:
            parent, node = pending.pop()
            # The native expr rule permits subqueries; their SELECT lists and
            # FROM clauses may legitimately contain aliases and tables.
            if isinstance(node, exp.Subquery):
                query = node.this
                while isinstance(query, exp.Subquery):
                    query = query.this
                if not isinstance(query, exp.Query):
                    return False
                continue
            if isinstance(node, exp.Query):
                if isinstance(parent, exp.Exists) and parent.this is node:
                    continue
                return False
            if isinstance(node, (exp.Alias, exp.Table, exp.TableAlias, exp.Limit, exp.Offset)):
                return False
            pending.extend((node, child) for child in node.iter_expressions())
        return True

    def _show_check_identifier(self, identifier: exp.Identifier) -> None:
        if (
            not isinstance(identifier, exp.Identifier)
            or any(key not in ("this", "quoted") for key in identifier.args)
            or not isinstance(identifier.this, str)
            or (
                identifier.args.get("quoted") is not None
                and not isinstance(identifier.args["quoted"], bool)
            )
        ):
            raise ValueError("Invalid SHOW identifier")
        self._metadata_check_identifier(identifier)
        if not identifier.args.get("quoted"):
            name = identifier.this
            try:
                tokens = self.dialect.tokenize(name)
            except TokenError as error:
                raise ValueError("Invalid unquoted SHOW identifier") from error
            if (
                len(tokens) != 1
                or tokens[0].token_type not in self.dialect.parser_class.ID_VAR_TOKENS
                or tokens[0].token_type == TokenType.IDENTIFIER
                or tokens[0].text != name
                or tokens[0].start != 0
                or tokens[0].end != len(name) - 1
                or tokens[0].comments
            ):
                raise ValueError("Invalid unquoted SHOW identifier")

    def impala_show_sql(self, expression: ImpalaShow) -> str:
        for node in expression.walk():
            if any(key not in node.arg_types for key in node.args):
                raise ValueError("Unsupported SHOW expression field")
        kind = expression.args.get("kind")
        table = expression.args.get("this")
        db = expression.args.get("db")
        pattern = expression.args.get("pattern")
        partition = expression.args.get("partition")
        where = expression.args.get("where")
        version = expression.args.get("version")
        function_kind = expression.args.get("function_kind")
        with_stats = expression.args.get("with_stats")
        if any(key not in ImpalaShow.arg_types for key in expression.args):
            raise ValueError("Unsupported SHOW option")
        table_kinds = (
            "CREATE TABLE", "CREATE VIEW", "TABLE STATS", "COLUMN STATS",
            "PARTITIONS", "RANGE PARTITIONS", "HASH SCHEMA", "FILES",
        )
        if kind in table_kinds:
            if db is not None or pattern is not None or function_kind is not None:
                raise ValueError("Unexpected SHOW table option")
            if with_stats is not None and (
                kind not in ("CREATE TABLE", "CREATE VIEW") or with_stats is not True
            ):
                raise ValueError("WITH STATS is only valid on SHOW CREATE")
            if where is not None and (kind != "PARTITIONS" or not self._show_value_expression(where)):
                raise ValueError("WHERE is only valid on SHOW PARTITIONS")
            if (partition is not None or version is not None) and kind != "FILES":
                raise ValueError("Partition and time travel are only valid on SHOW FILES")
            sql = "SHOW " + kind + (" IN " if kind == "FILES" else " ")
            if isinstance(table, exp.Table):
                if any(key not in exp.Table.arg_types for key in table.args):
                    raise ValueError("Unsupported SHOW table option")
                self._show_check_identifier(table.this)
                if table.args.get("db") is not None:
                    self._show_check_identifier(table.args["db"])
            sql += self._metadata_table_sql(table)
            if with_stats:
                sql += " WITH STATS"
            if where is not None:
                sql += " WHERE " + self.sql(where)
            if version is not None:
                if (
                    not isinstance(version, exp.Version)
                    or any(key not in ("this", "expression", "kind") for key in version.args)
                    or version.this not in ("TIMESTAMP", "VERSION")
                    or version.args.get("kind") != "AS OF"
                    or not self._show_value_expression(version.expression)
                    or (
                        version.this == "VERSION"
                        and (
                            not isinstance(version.expression, exp.Literal)
                            or version.expression.is_string is not False
                            or any(key not in ("this", "is_string") for key in version.expression.args)
                            or not isinstance(version.expression.this, str)
                            or not re.fullmatch(
                                r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
                                version.expression.this,
                            )
                        )
                    )
                ):
                    raise ValueError("Invalid SHOW FILES time travel")
                sql += " " + self.sql(version)
            if partition is not None:
                if (
                    not isinstance(partition, exp.Partition)
                    or any(key != "expressions" for key in partition.args)
                    or not partition.expressions
                    or any(not self._show_value_expression(item) for item in partition.expressions)
                ):
                    raise ValueError("Invalid SHOW FILES partition")
                sql += " " + self.maybe_comment(
                    self._metadata_partition_sql(partition, static=False), partition
                )
            return sql
        if kind not in ("TABLES", "DATABASES", "FUNCTIONS") or table is not None:
            raise ValueError("Invalid SHOW kind or target")
        if any(value is not None for value in (partition, where, version, with_stats)):
            raise ValueError("Unexpected SHOW option")
        if function_kind is not None and (
            kind != "FUNCTIONS" or function_kind not in ("AGGREGATE", "ANALYTIC")
        ):
            raise ValueError("Invalid SHOW function category")
        if db is not None:
            if kind not in ("TABLES", "FUNCTIONS") or not isinstance(db, exp.Identifier):
                raise ValueError("Only SHOW TABLES and FUNCTIONS accept a database identifier")
            self._show_check_identifier(db)
        if pattern is not None and (
            not isinstance(pattern, exp.Literal)
            or pattern.is_string is not True
            or not isinstance(pattern.this, str)
            or any(key not in ("this", "is_string") for key in pattern.args)
        ):
            raise ValueError("SHOW pattern must be a string literal")
        sql = "SHOW " + (function_kind + " " if function_kind else "") + kind
        if db is not None:
            sql += " IN " + self.sql(db)
        if pattern is not None:
            sql += " LIKE " + self.sql(pattern)
        return sql
