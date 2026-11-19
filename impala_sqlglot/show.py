"""Bounded structured support for native Impala SHOW statements.

The tokenizer must remove SHOW from COMMANDS and the parser must dispatch
TokenType.SHOW to _parse_show. Other SHOW families remain opaque commands.
The metadata parser and generator mixins provide the identifier helpers.
"""
from __future__ import annotations

from sqlglot import exp
from sqlglot.tokens import TokenType

from impala_sqlglot.metadata import _metadata_word


class ImpalaShow(exp.Expression):
    # kind is TABLES, DATABASES or CREATE TABLE; SCHEMAS normalizes to DATABASES.
    arg_types = {"kind": True, "this": False, "db": False, "pattern": False}


class ImpalaShowParserMixin:
    """Parse only the selected SHOW families; delegate other families."""

    def _parse_show(self) -> exp.Expression:
        start = self._prev
        body_start = self._curr
        if not self._curr:
            self.raise_error("Expected a SHOW family")
            return self.expression(ImpalaShow(kind=""))

        tables = _metadata_word(self._curr, "TABLES")
        databases = _metadata_word(self._curr, "DATABASES") or _metadata_word(
            self._curr, "SCHEMAS"
        )
        create = _metadata_word(self._curr, "CREATE")
        create_table = create and _metadata_word(self._next, "TABLE")

        if create and not self._next:
            self.raise_error("Expected a SHOW CREATE object type")
            return self.expression(ImpalaShow(kind=""))
        if not (tables or databases or create_table):
            return self._show_opaque_command(start, body_start, use_super=True)

        # The metadata helpers capture comments on consumed keywords and dots.
        previous = getattr(self, "_impala_metadata_comments", None)
        self._impala_metadata_comments = []
        self._prev_comments = []
        try:
            if create_table:
                self._metadata_expect_word("CREATE")
                self._metadata_expect_word("TABLE")
                table = self._parse_metadata_table()
                # Native Impala also supports WITH STATS, which is outside this
                # node's contract. Preserve it as an opaque command.
                if (
                    _metadata_word(self._curr, "WITH")
                    and _metadata_word(self._next, "STATS")
                    and self._index + 2 == self._tokens_size
                ):
                    return self._show_opaque_command(start, body_start, use_super=False)
                result = ImpalaShow(kind="CREATE TABLE", this=table)
            else:
                self._metadata_match_word("TABLES" if tables else self._curr.text.upper())
                db = None
                if tables and self._metadata_match_word("IN"):
                    db = self._parse_metadata_identifier()
                    if db is None:
                        self.raise_error("Expected a database after IN")
                like = self._metadata_match_word("LIKE")
                pattern = None
                if self._curr and self._curr.token_type == TokenType.STRING:
                    pattern = self._parse_string()
                elif like:
                    self.raise_error("Expected a string pattern after LIKE")
                result = ImpalaShow(
                    kind="TABLES" if tables else "DATABASES", db=db, pattern=pattern
                )

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

    def impala_show_sql(self, expression: ImpalaShow) -> str:
        kind = expression.args.get("kind")
        table = expression.args.get("this")
        db = expression.args.get("db")
        pattern = expression.args.get("pattern")
        if any(
            value is not None
            for key, value in expression.args.items()
            if key not in ImpalaShow.arg_types
        ):
            raise ValueError("Unsupported SHOW option")
        if kind == "CREATE TABLE":
            if db is not None or pattern is not None:
                raise ValueError("SHOW CREATE TABLE does not accept a database or pattern")
            return "SHOW CREATE TABLE " + self._metadata_table_sql(table)
        if kind not in ("TABLES", "DATABASES") or table is not None:
            raise ValueError("Invalid SHOW kind or target")
        if db is not None:
            if kind != "TABLES" or not isinstance(db, exp.Identifier):
                raise ValueError("Only SHOW TABLES accepts a database identifier")
            self._metadata_check_identifier(db)
        if pattern is not None and (
            not isinstance(pattern, exp.Literal) or not pattern.is_string
        ):
            raise ValueError("SHOW pattern must be a string literal")
        sql = "SHOW " + kind
        if db is not None:
            sql += " IN " + self.sql(db)
        if pattern is not None:
            sql += " LIKE " + self.sql(pattern)
        return sql
