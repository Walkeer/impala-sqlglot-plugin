"""Structured Impala query, write, hint and constraint syntax."""
from __future__ import annotations

from sqlglot import exp
from sqlglot.tokens import TokenType


def is_impala_unsigned_integer(value, maximum):
    """Check a decimal integer without converting an unbounded input to int."""
    if not isinstance(value, exp.Literal) or value.is_string:
        return False
    text = value.this
    if not isinstance(text, str) or not text.isascii() or not text.isdecimal():
        return False
    text = text.lstrip("0") or "0"
    limit = str(maximum)
    return len(text) < len(limit) or len(text) == len(limit) and text <= limit


class Upsert(exp.Insert):
    """Kudu UPSERT: distinct from INSERT, with the same target/source fields."""


class ImpalaPrimaryKey(exp.PrimaryKey):
    """Table key with independently declared, optional native modifiers."""

    arg_types = {
        **exp.PrimaryKey.arg_types, "enabled": False, "validated": False, "rely": False,
    }


class ImpalaForeignKey(exp.ForeignKey):
    arg_types = {
        **exp.ForeignKey.arg_types, "enabled": False, "validated": False, "rely": False,
    }


class NonUniquePrimaryKey(ImpalaPrimaryKey):
    arg_types = {**ImpalaPrimaryKey.arg_types, "unique": False}


class NonUniquePrimaryKeyColumnConstraint(exp.PrimaryKeyColumnConstraint):
    arg_types = {**exp.PrimaryKeyColumnConstraint.arg_types, "unique": False}


class VariadicParameter(exp.Expression):
    arg_types = {"this": True}


class DropFunction(exp.Expression):
    arg_types = {
        "this": True, "expressions": False, "exists": False,
        "aggregate": False, "wrapped": False,
    }

class ImpalaSubquery(exp.Subquery):
    arg_types = {**exp.Subquery.arg_types, "hints": False}


class PlanHintProperty(exp.Property):
    arg_types = {"this": True}


class ImpalaSyntaxParserMixin:
    def _parse_create(self):
        hint = self._parse_hint()
        result = super()._parse_create()
        if (
            isinstance(result, exp.Create)
            and result.kind == "TABLE"
            and result.expression is not None
            and any(
                key.args.get(field) is not None
                for key in result.find_all(exp.PrimaryKey, exp.ForeignKey)
                for field in ("enabled", "validated", "rely")
            )
        ):
            self.raise_error("Impala CREATE TABLE AS SELECT does not accept key modifiers")
        if hint is not None:
            if not isinstance(result, exp.Create) or result.kind != "TABLE" or result.expression is None:
                self.raise_error("CREATE hints require CREATE TABLE AS SELECT")
                return result
            properties = result.args.get("properties")
            if properties is None:
                properties = exp.Properties(expressions=[])
                result.set("properties", properties)
            properties.append("expressions", PlanHintProperty(this=hint))
        return result

    def _parse_select_query(self, *args, **kwargs):
        stack = getattr(self, "_impala_select_hint_stack", None)
        if stack is None:
            self._impala_select_hint_stack = stack = []
        frame = {}
        stack.append(frame)
        try:
            query = super()._parse_select_query(*args, **kwargs)
            if frame.get("hint") is not None:
                select = query
                while isinstance(select, (exp.SetOperation, exp.Subquery)):
                    select = select.this
                if not isinstance(select, exp.Select) or select.args.get("hint"):
                    self.raise_error("Unexpected additional SELECT hint")
                else:
                    select.set("hint", frame["hint"])
            return query
        finally:
            stack.pop()

    def _parse_projections(self):
        # Unlike the base parser, Impala accepts hints after ALL/DISTINCT.
        stack = getattr(self, "_impala_select_hint_stack", ())
        if stack and self._prev and self._prev.token_type in (TokenType.ALL, TokenType.DISTINCT):
            hint = self._parse_hint()
            if hint is not None:
                stack[-1]["hint"] = hint
        return super()._parse_projections()

    def _parse_hint_item(self):
        if not self._curr or self._curr.token_type in (
            TokenType.R_BRACKET, TokenType.R_PAREN, TokenType.COMMA,
        ):
            return None
        name = self._parse_var(
            tokens={*self.ID_VAR_TOKENS, TokenType.STRAIGHT_JOIN}, upper=True
        )
        if name is None or not self._match(TokenType.L_PAREN):
            return name
        arguments = self._parse_csv(lambda: self._parse_number() or self._parse_id_var())
        self._match_r_paren()
        return self.expression(exp.Anonymous(this=name.name, expressions=arguments))

    def _parse_hint_body(self):
        items = self._parse_csv(self._parse_hint_item)
        if not items and not self._curr:
            return None
        hint = self.expression(exp.Hint(expressions=items))
        if self._curr:
            self.raise_error("Expected comma-separated plan hints")
        return self._validate_impala_hint(hint)

    def _parse_hint(self):
        if self._match(TokenType.HINT):
            comments = list(self._prev.comments)
            self._prev_comments = []
            if not comments or not comments[0].strip():
                return None
            hint = exp.maybe_parse(
                comments[0], into=exp.Hint, dialect=self.dialect, error_level=self.error_level,
            )
            if hint is not None:
                hint.add_comments(comments[1:])
        elif self._match(TokenType.L_BRACKET):
            items = self._parse_csv(self._parse_hint_item)
            if not self._match(TokenType.R_BRACKET):
                self.raise_error("Expected closing bracket after plan hints")
            if not items:
                return None
            hint = self.expression(exp.Hint(expressions=items))
        else:
            return None
        return self._validate_impala_hint(hint)

    def _validate_impala_hint(self, hint):
        if not hint or not hint.expressions:
            self.raise_error("Expected at least one plan hint")
            return hint
        for item in hint.expressions:
            if not isinstance(item, (exp.Column, exp.Var, exp.Anonymous)):
                self.raise_error("Expected a plan-hint name")
            if isinstance(item, exp.Column) and (item.table or item.db):
                self.raise_error("Plan-hint names cannot be qualified")
            if isinstance(item, exp.Anonymous):
                if not (
                    item.expressions and all(isinstance(arg, exp.Identifier) for arg in item.expressions)
                    or len(item.expressions) == 1
                    and isinstance(item.expressions[0], exp.Literal) and item.expressions[0].is_int
                ):
                    self.raise_error("Expected identifiers or a single integer hint argument")
            if isinstance(item, exp.Column):
                item.this.set("this", item.name.upper())
            else:
                item.set("this", item.name.upper())
        return hint

    def _parse_table(self, *args, **kwargs):
        hint = None
        if self._prev and self._prev.token_type in (TokenType.JOIN, TokenType.STRAIGHT_JOIN):
            hint = self._parse_hint()
        table = super()._parse_table(*args, **kwargs)
        if isinstance(table, exp.Subquery) and self._curr and self._curr.token_type == TokenType.TABLE_SAMPLE:
            table.set("sample", self._parse_table_sample())
        trailing_hint = self._parse_hint()
        if trailing_hint is not None:
            if isinstance(table, exp.Subquery):
                wrapped = ImpalaSubquery(**table.args)
                wrapped.add_comments(table.comments)
                wrapped.meta.update(table.meta)
                table = wrapped
            if isinstance(table, (exp.Table, ImpalaSubquery)):
                table.append("hints", trailing_hint)
            else:
                self.raise_error("Expected a table or inline view before plan hints")
        if hint is not None and table is not None:
            table.meta["impala_join_hint"] = hint
        return table

    def _parse_bracket(self, this=None):
        # A bracket after a table is a legacy plan hint, not table indexing.
        if isinstance(this, exp.Table) and self._curr and self._curr.token_type == TokenType.L_BRACKET:
            return this
        return super()._parse_bracket(this)

    def _parse_table_hints(self):
        hints = super()._parse_table_hints() or []
        hint = self._parse_hint()
        if hint is not None:
            hints.append(hint)
        return hints or None

    def _parse_join(self, *args, **kwargs):
        join = super()._parse_join(*args, **kwargs)
        if join is not None and join.this is not None:
            hint = join.this.meta.pop("impala_join_hint", None)
            if hint is not None:
                join.set("hint", hint)
        return join

    def _parse_insert(self):
        operation = self._prev.text.upper()
        hint = self._parse_hint()
        overwrite = self._match(TokenType.OVERWRITE)
        if operation == "UPSERT" and overwrite:
            self.raise_error("UPSERT does not support OVERWRITE")
        if not overwrite and not self._match(TokenType.INTO):
            self.raise_error("Expected INTO after write operation")
        self._match(TokenType.TABLE)
        target = self._parse_table_parts(schema=True)
        if target is None or not target.name:
            self.raise_error("Expected write target")
        if self._match(TokenType.L_PAREN):
            columns = []
            if not self._match(TokenType.R_PAREN):
                while self._curr:
                    if self._curr.token_type in (TokenType.STRING, TokenType.NUMBER):
                        self.raise_error("Expected a write column name")
                        break
                    column = self._parse_id_var(any_token=False)
                    if not isinstance(column, exp.Identifier):
                        self.raise_error("Expected a write column name")
                        break
                    columns.append(column)
                    if self._match(TokenType.R_PAREN):
                        break
                    if not self._match(TokenType.COMMA):
                        self.raise_error("Expected comma or closing write-column parenthesis")
                        break
                else:
                    self.raise_error("Expected closing write-column parenthesis")
            target = self.expression(exp.Schema(this=target, expressions=columns))
        partition = self._parse_partition()
        if operation == "UPSERT" and partition is not None:
            self.raise_error("UPSERT does not support PARTITION")
        trailing_hint = self._parse_hint()
        if trailing_hint:
            if hint:
                self.raise_error("Write hints must occur in one position")
            else:
                hint = trailing_hint
        source = self._parse_ddl_select()
        if source is None and not isinstance(target, exp.Schema):
            self.raise_error("Expected SELECT or VALUES source")
        cls = Upsert if operation == "UPSERT" else exp.Insert
        return self.expression(
            cls(this=target, expression=source, overwrite=overwrite, partition=partition, hint=hint),
            comments=[],
        )

    def _parse_primary_key_part(self):
        column = self._parse_metadata_identifier()
        if column is None:
            self.raise_error("Expected an Impala primary-key column identifier")
        return column

    def _parse_primary_key(self, *args, **kwargs):
        non_unique = self._prev.text.upper() == "NON UNIQUE PRIMARY KEY"
        table_key = bool(
            kwargs.get("in_props")
            or self._curr and self._curr.token_type == TokenType.L_PAREN
        )
        previous = getattr(self, "_impala_key_options", None)
        self._impala_key_options = "PRIMARY" if table_key else previous
        try:
            constraint = super()._parse_primary_key(*args, **kwargs)
        finally:
            self._impala_key_options = previous
        if non_unique and "NOT ENFORCED" in (constraint.args.get("options") or []):
            self.raise_error("NON UNIQUE and NOT ENFORCED are alternative primary-key forms")
        modifiers, modifier_comments = (
            self._parse_impala_key_modifiers() if table_key else ({}, [])
        )
        if not non_unique:
            if modifiers:
                return self.expression(
                    ImpalaPrimaryKey(**constraint.args, **modifiers),
                    comments=[*(constraint.comments or []), *modifier_comments],
                )
            return constraint
        cls = (
            NonUniquePrimaryKey
            if isinstance(constraint, exp.PrimaryKey)
            else NonUniquePrimaryKeyColumnConstraint
        )
        if isinstance(constraint, exp.PrimaryKeyColumnConstraint) and (
            self._prev and self._prev.token_type == TokenType.PRIMARY_KEY
        ):
            # In a schema's standalone constraint slot a column list is required.
            if getattr(self, "_impala_standalone_key", False):
                self.raise_error("Expected primary-key column list")
        return self.expression(
            cls(**constraint.args, **modifiers, unique=False),
            comments=[*(constraint.comments or []), *modifier_comments],
        )

    def _parse_foreign_key(self):
        previous = getattr(self, "_impala_metadata_comments", None)
        self._impala_metadata_comments = list(self._prev_comments)
        self._prev_comments = []
        try:
            columns = self._parse_impala_key_columns()
            self._metadata_expect_word("REFERENCES")
            table = self._parse_metadata_table()
            referenced_columns = self._parse_impala_key_columns()
            modifiers, modifier_comments = self._parse_impala_key_modifiers()
            reference = exp.Reference(
                this=exp.Schema(this=table, expressions=referenced_columns), options=[],
            )
            cls = ImpalaForeignKey if modifiers else exp.ForeignKey
            return self.expression(
                cls(expressions=columns, reference=reference, options=[], **modifiers),
                comments=[*self._impala_metadata_comments, *modifier_comments],
            )
        finally:
            self._impala_metadata_comments = previous

    def _parse_impala_key_columns(self):
        self._metadata_expect_token(TokenType.L_PAREN, "(")
        columns = []
        while True:
            column = self._parse_metadata_identifier()
            if column is None:
                self.raise_error("Expected an Impala key column identifier")
                break
            columns.append(column)
            if not self._metadata_match_token(TokenType.COMMA):
                break
        self._metadata_expect_token(TokenType.R_PAREN, ")")
        return columns

    def _parse_key_constraint_options(self):
        kind = getattr(self, "_impala_key_options", None)
        if kind is None:
            return super()._parse_key_constraint_options()
        # Native modifiers follow the key, in a fixed order. Do not let the
        # base parser swallow RELY/NORELY as unordered generic options.
        if (
            kind == "PRIMARY"
            and self._curr and self._next
            and self._curr.token_type != TokenType.IDENTIFIER
            and self._curr.token_type not in self.STRING_PARSERS
            and self._next.token_type != TokenType.IDENTIFIER
            and self._next.token_type not in self.STRING_PARSERS
            and self._curr.text.upper() == "NOT"
            and self._next.text.upper() == "ENFORCED"
        ):
            comments = list(self._curr.comments)
            self._advance(2)
            self._prev_comments = [*comments, *self._prev_comments]
            return ["NOT ENFORCED"]
        return []

    def _parse_impala_key_modifiers(self):
        modifiers = {}
        comments = []
        groups = (
            ("enabled", "ENABLE", "DISABLE"),
            ("validated", "VALIDATE", "NOVALIDATE"),
            ("rely", "RELY", "NORELY"),
        )
        for field, positive, negative in groups:
            token = self._curr
            if (
                token
                and token.token_type != TokenType.IDENTIFIER
                and token.token_type not in self.STRING_PARSERS
            ):
                word = token.text.upper()
                if word in (positive, negative):
                    modifiers[field] = word == positive
                    self._advance()
                    comments.extend(self._prev_comments)
                    self._prev_comments = []
        token = self._curr
        if (
            token
            and token.token_type != TokenType.IDENTIFIER
            and token.token_type not in self.STRING_PARSERS
            and token.text.upper() in {word for _, yes, no in groups for word in (yes, no)}
        ):
            self.raise_error("Repeated or out-of-order Impala key modifier")
        return modifiers, comments

    def _parse_unnamed_constraint(self, *args, **kwargs):
        previous = getattr(self, "_impala_standalone_key", False)
        self._impala_standalone_key = True
        try:
            return super()._parse_unnamed_constraint(*args, **kwargs)
        finally:
            self._impala_standalone_key = previous

    def _parse_table_sample(self, as_modifier=False):
        if not self._match(TokenType.TABLE_SAMPLE):
            return None
        if not self._match_text_seq("SYSTEM") or not self._match(TokenType.L_PAREN):
            self.raise_error("Expected TABLESAMPLE SYSTEM(integer percentage)")
        percent = self._parse_number()
        if not is_impala_unsigned_integer(percent, 100):
            self.raise_error("Sample percentage must be an integer from 0 to 100")
        self._match_r_paren()
        seed = None
        if self._match_text_seq("REPEATABLE"):
            if not self._match(TokenType.L_PAREN):
                self.raise_error("Expected REPEATABLE(integer seed)")
            seed = self._parse_number()
            if not is_impala_unsigned_integer(seed, 9223372036854775807):
                self.raise_error("Expected a non-negative 64-bit integer seed")
            self._match_r_paren()
        return self.expression(exp.TableSample(method=exp.var("SYSTEM"), percent=percent, seed=seed))

    def _parse_impala_function_parameters(self):
        parameters = []
        if self._match(TokenType.R_PAREN):
            return parameters
        while self._curr:
            parameter = self._parse_types()
            if parameter is None:
                self.raise_error("Expected function parameter type")
                break
            if parameter.is_type(exp.DataType.Type.NULL):
                self.raise_error("NULL is not a function parameter type")
                break
            if self._match(TokenType.DOT):
                if not self._match(TokenType.DOT) or not self._match(TokenType.DOT):
                    self.raise_error("Expected a final three-dot variadic marker")
                    break
                parameters.append(self.expression(VariadicParameter(this=parameter)))
                if not self._match(TokenType.R_PAREN):
                    self.raise_error("Variadic parameter must be last")
                return parameters
            parameters.append(parameter)
            if self._match(TokenType.R_PAREN):
                return parameters
            if not self._match(TokenType.COMMA):
                self.raise_error("Expected comma or closing parameter parenthesis")
                break
            if self._curr and self._curr.token_type == TokenType.R_PAREN:
                self.raise_error("Expected function parameter after comma")
                break
        self.raise_error("Expected closing function parameter parenthesis")
        return parameters

    def _parse_user_defined_function(self, kind=None):
        if kind != TokenType.FUNCTION:
            return super()._parse_user_defined_function(kind=kind)
        target = self._parse_table_parts(schema=True)
        if not self._match(TokenType.L_PAREN):
            return target
        return self.expression(
            exp.UserDefinedFunction(
                this=target, expressions=self._parse_impala_function_parameters(), wrapped=True,
            )
        )

    def _parse_drop(self, *args, **kwargs):
        if not self._curr or self._curr.text.upper() not in ("FUNCTION", "AGGREGATE"):
            return super()._parse_drop(*args, **kwargs)
        aggregate = self._match_text_seq("AGGREGATE")
        if not self._match(TokenType.FUNCTION):
            self.raise_error("Expected FUNCTION after DROP AGGREGATE")
        exists = self._parse_exists()
        target = self._parse_table_parts(schema=True)
        if target is None or not target.name:
            self.raise_error("Expected function name")
        wrapped = self._match(TokenType.L_PAREN)
        parameters = self._parse_impala_function_parameters() if wrapped else []
        return self.expression(
            DropFunction(
                this=target, expressions=parameters, exists=exists,
                aggregate=aggregate, wrapped=wrapped,
            )
        )


class ImpalaSyntaxGeneratorMixin:
    def table_sql(self, expression, sep=" AS "):
        hints = expression.args.get("hints") or []
        if not hints or expression.args.get("sample") is None:
            return super().table_sql(expression, sep=sep)
        # Hive puts table hints before sampling. Impala puts them after it,
        # but still before joins or other table suffixes.
        clean = expression.copy()
        clean.set("hints", None)
        body = clean.copy()
        for key in ("joins", "laterals", "ordinality"):
            body.set(key, None)
        generated = super().table_sql(clean, sep=sep)
        prefix = super().table_sql(body, sep=sep)
        if not generated.startswith(prefix):
            raise ValueError("Unexpected base sampled-table rendering")
        hint_sql = self.expressions(expression, key="hints", sep=" ", flat=True).strip()
        return prefix + " " + hint_sql + generated[len(prefix):]

    def create_sql(self, expression):
        if expression.find(exp.UniqueColumnConstraint) is not None:
            raise ValueError("Impala does not support UNIQUE constraints")
        if expression.expression is not None and any(
            key.args.get(field) is not None
            for key in expression.find_all(exp.PrimaryKey, exp.ForeignKey)
            for field in ("enabled", "validated", "rely")
        ):
            raise ValueError("Impala CREATE TABLE AS SELECT does not accept key modifiers")
        return super().create_sql(expression)

    def impalasubquery_sql(self, expression):
        clean = expression.copy()
        clean.set("hints", None)
        generated = super().subquery_sql(clean)
        hints = self.expressions(expression, key="hints", sep=" ", flat=True).strip()
        return generated + (" " + hints if hints else "")

    def planhintproperty_sql(self, expression):
        return self.sql(expression, "this").strip()

    def select_sql(self, expression):
        hint = expression.args.get("hint")
        if hint is None or expression.args.get("distinct") is None:
            return super().select_sql(expression)
        clean = expression.copy()
        clean.set("hint", None)
        clean.set("with_", None)
        generated = super().select_sql(clean)
        prefix = "SELECT " + self.sql(expression, "distinct")
        if not generated.startswith(prefix):
            raise ValueError("Unexpected base SELECT rendering")
        generated = prefix + " " + self.sql(hint).strip() + generated[len(prefix):]
        return self.prepend_ctes(expression, generated)

    def version_sql(self, expression):
        if expression.this not in ("TIMESTAMP", "VERSION") or expression.args.get("kind") != "AS OF":
            raise ValueError("Impala time travel requires SYSTEM_TIME or SYSTEM_VERSION AS OF")
        if expression.expression is None:
            raise ValueError("Time travel requires an AS OF value")
        kind = "SYSTEM_TIME" if expression.this == "TIMESTAMP" else "SYSTEM_VERSION"
        return f"FOR {kind} AS OF {self.sql(expression, 'expression')}"

    def tablesample_sql(self, expression):
        method = expression.args.get("method")
        percent = expression.args.get("percent")
        seed = expression.args.get("seed")
        if not isinstance(method, exp.Var) or method.name.upper() != "SYSTEM":
            raise ValueError("Impala sampling requires the SYSTEM method")
        if not is_impala_unsigned_integer(percent, 100):
            raise ValueError("Impala sampling requires an integer percentage from 0 to 100")
        if seed is not None and (
            not is_impala_unsigned_integer(seed, 9223372036854775807)
        ):
            raise ValueError("Impala sampling requires a non-negative 64-bit integer seed")
        if any(
            value not in (None, False, [])
            for key, value in expression.args.items()
            if key not in ("method", "percent", "seed")
        ):
            raise ValueError("Unsupported Impala sampling option")
        sql = f"TABLESAMPLE SYSTEM({self.sql(expression, 'percent')})"
        if expression.args.get("seed") is not None:
            sql += f" REPEATABLE({self.sql(expression, 'seed')})"
        return " " + sql

    def join_sql(self, expression):
        hint = expression.args.get("hint")
        if not isinstance(hint, exp.Hint):
            return super().join_sql(expression)
        clean = expression.copy()
        clean.set("hint", None)
        # Prevent the base generator from treating a hint-only JOIN as a comma.
        if not any(clean.args.get(key) for key in ("method", "side", "kind", "on", "using")):
            clean.set("kind", "INNER")
        operator = " ".join(
            part for part in (expression.method, expression.side, expression.kind, "JOIN") if part
        )
        if expression.kind == "STRAIGHT_JOIN":
            operator = "STRAIGHT_JOIN"
        base_operator = " ".join(
            part for part in (clean.method, clean.side, clean.kind, "JOIN") if part
        )
        if clean.kind == "STRAIGHT_JOIN":
            base_operator = "STRAIGHT_JOIN"
        generated = super().join_sql(clean)
        prefix = self.seg(base_operator)
        if not generated.startswith(prefix):
            raise ValueError("Unexpected base JOIN rendering")
        return self.seg(operator) + " " + self.sql(hint).strip() + generated[len(prefix):]

    def _impala_write_sql(self, expression, operation):
        if any(
            value not in (None, False, [])
            for key, value in expression.args.items()
            if key not in ("this", "expression", "overwrite", "partition", "hint", "with_")
        ):
            raise ValueError("Unsupported Impala write option")
        if operation == "UPSERT" and (
            expression.args.get("overwrite") or expression.args.get("partition")
        ):
            raise ValueError("UPSERT does not support OVERWRITE or PARTITION")
        overwrite = " OVERWRITE TABLE" if expression.args.get("overwrite") else " INTO"
        hint = self.sql(expression, "hint").strip()
        hint = f" {hint}" if hint else ""
        partition = self.sql(expression, "partition")
        partition = f" {partition}" if partition else ""
        source = self.sql(expression, "expression")
        target = expression.this
        if source == "" and not isinstance(target, exp.Schema):
            raise ValueError("A source-less write requires a column permutation")
        source = self.sep() + source if source else ""
        target_sql = (
            self.sql(target.this) + " ()"
            if isinstance(target, exp.Schema) and not target.expressions
            else self.sql(target)
        )
        sql = f"{operation}{hint}{overwrite} {target_sql}{partition}{source}"
        return self.prepend_ctes(expression, sql)

    def insert_sql(self, expression):
        return self._impala_write_sql(expression, "INSERT")

    def upsert_sql(self, expression):
        return self._impala_write_sql(expression, "UPSERT")

    def nonuniqueprimarykey_sql(self, expression):
        return "NON UNIQUE " + self.impalaprimarykey_sql(expression)

    def _impala_key_modifiers_sql(self, expression):
        allowed = ["NOT ENFORCED"] if isinstance(expression, exp.PrimaryKey) else []
        if any(option not in allowed for option in expression.args.get("options") or []):
            raise ValueError("Use native Impala key modifier fields, not generic key options")
        modifiers = []
        for field, positive, negative in (
            ("enabled", "ENABLE", "DISABLE"),
            ("validated", "VALIDATE", "NOVALIDATE"),
            ("rely", "RELY", "NORELY"),
        ):
            value = expression.args.get(field)
            if value is not None:
                if type(value) is not bool:
                    raise ValueError(f"Impala key {field} must be a boolean or None")
                modifiers.append(positive if value else negative)
        return (" " + " ".join(modifiers)) if modifiers else ""

    def impalaprimarykey_sql(self, expression):
        if isinstance(expression, NonUniquePrimaryKey) and "NOT ENFORCED" in (
            expression.args.get("options") or []
        ):
            raise ValueError("NON UNIQUE and NOT ENFORCED are alternative primary-key forms")
        modifiers = self._impala_key_modifiers_sql(expression)
        if not expression.expressions or not all(
            isinstance(column, exp.Identifier) for column in expression.expressions
        ):
            raise ValueError("Impala primary keys require column identifiers")
        for column in expression.expressions:
            self._metadata_check_identifier(column)
        return super().primarykey_sql(expression) + modifiers

    def impalaforeignkey_sql(self, expression):
        modifiers = self._impala_key_modifiers_sql(expression)
        reference = expression.args.get("reference")
        target = reference.this if isinstance(reference, exp.Reference) else None
        if (
            not expression.expressions
            or not all(isinstance(column, exp.Identifier) for column in expression.expressions)
            or not isinstance(target, exp.Schema)
            or not target.expressions
            or not all(isinstance(column, exp.Identifier) for column in target.expressions)
            or expression.args.get("delete") or expression.args.get("update")
            or reference.args.get("options")
        ):
            raise ValueError("Impala foreign keys require column lists and a complete reference")
        self._metadata_table_sql(target.this)
        for column in [*expression.expressions, *target.expressions]:
            self._metadata_check_identifier(column)
        return super().foreignkey_sql(expression) + modifiers

    def nonuniqueprimarykeycolumnconstraint_sql(self, expression):
        return "NON UNIQUE " + super().primarykeycolumnconstraint_sql(expression)

    def variadicparameter_sql(self, expression):
        return self.sql(expression, "this") + "..."

    def dropfunction_sql(self, expression):
        aggregate = " AGGREGATE" if expression.args.get("aggregate") else ""
        exists = " IF EXISTS" if expression.args.get("exists") else ""
        signature = ""
        if expression.args.get("wrapped") or expression.expressions:
            signature = f"({self.expressions(expression, flat=True)})"
        return f"DROP{aggregate} FUNCTION{exists} {self.sql(expression, 'this')}{signature}"
