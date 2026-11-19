"""Structured Impala Kudu partitioning and Parquet schema references.

Integrate by placing the parser and generator mixins before the existing
Impala parser and generator in their respective MROs. Merge
``ImpalaDDLGeneratorMixin.DDL_TRANSFORMS`` into the generator's existing
``TRANSFORMS``, and merge ``DDL_PROPERTY_LOCATIONS`` into its existing
``PROPERTIES_LOCATION``. The maps are separate so existing overrides remain
under the dialect owner's control.

The parser also rejects obviously invalid bucket counts and numeric ranges.
A Parquet reference is never opened, and table schemas, partition columns,
data types, and existing ranges are not resolved.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from sqlglot import exp
from sqlglot.tokens import TokenType


class KuduPartitioningProperty(exp.Property):
    """A PARTITION BY clause, with ordered hash and range components."""

    arg_types = {"expressions": True}


class KuduHashPartition(exp.Expression):
    """A hash component; absent columns mean the primary key is implicit."""

    arg_types = {"columns": False, "buckets": True}


class KuduRangePartitioning(exp.Expression):
    """A range component; absent columns mean the primary key is implicit."""

    arg_types = {"columns": False, "expressions": True}


class KuduRangeBound(exp.Expression):
    """One range endpoint and whether it includes its value."""

    arg_types = {"this": True, "inclusive": False}


class KuduRangePartition(exp.Expression):
    """A range or equality partition, optionally with its own hash components."""

    arg_types = {"lower": False, "upper": False, "value": False, "hashes": False}


class KuduAlterRangePartition(exp.Expression):
    """An ADD or DROP RANGE PARTITION action within an ordinary Alter."""

    arg_types = {"this": True, "kind": True, "exists": False}


class ParquetSchemaReference(exp.Expression):
    """A string reference to a Parquet schema source, not its resolved schema."""

    arg_types = {"this": True}


_COMPARISONS = frozenset((TokenType.LT, TokenType.LTE, TokenType.GT, TokenType.GTE))


def _numeric_literal(expression: exp.Expression | None) -> Decimal | None:
    """Return only an unambiguously numeric literal, without evaluating SQL."""
    if isinstance(expression, exp.Paren):
        return _numeric_literal(expression.this)
    if isinstance(expression, exp.Neg):
        number = _numeric_literal(expression.this)
        return number.copy_negate() if number is not None else None
    if isinstance(expression, exp.Literal) and not expression.is_string:
        try:
            number = Decimal(expression.this)
            return number if number.is_finite() else None
        except InvalidOperation:
            return None
    return None


class ImpalaDDLParserMixin:
    """Extend the existing parser only for the covered Impala DDL forms."""

    def _match_kudu_keywords(self, *words: str, advance: bool = True) -> bool:
        # SQLGlot 30.10 can text-match quoted identifiers as keywords.
        tokens = self._tokens[self._index : self._index + len(words)]
        if len(tokens) != len(words) or any(
            token.token_type == TokenType.IDENTIFIER or token.token_type in self.STRING_PARSERS
            for token in tokens
        ):
            return False
        return self._match_text_seq(*words, advance=advance)

    def _parse_create(self):
        result = super()._parse_create()
        if isinstance(result, exp.Create):
            properties = result.args.get("properties")
            reference = properties and any(
                isinstance(prop, exp.LikeProperty)
                and isinstance(prop.this, ParquetSchemaReference)
                for prop in properties.expressions
            )
            if reference and (
                result.kind != "TABLE"
                or isinstance(result.this, exp.Schema)
                or result.args.get("expression") is not None
            ):
                self.raise_error("LIKE PARQUET requires a table without columns or AS SELECT")
        return result

    def _parse_partitioned_by(self):
        # PARTITIONED BY remains the existing Hive-style table property.
        if self._prev.text.upper() == "PARTITION BY":
            if (
                self._match_kudu_keywords("HASH", advance=False)
                or self._match_kudu_keywords("RANGE", advance=False)
            ):
                return self._parse_kudu_partitioning()
            self.raise_error("Expected HASH or RANGE after PARTITION BY")
            return None
        return super()._parse_partitioned_by()

    def _parse_create_like(self):
        if not self._match_kudu_keywords("PARQUET"):
            return super()._parse_create_like()

        if not self._match(TokenType.STRING, advance=False):
            self.raise_error("Expected a string schema reference after LIKE PARQUET")
            return None

        reference = self._parse_string()
        if not isinstance(reference, exp.Literal) or not reference.is_string or not reference.this:
            self.raise_error("Expected a nonempty string schema reference after LIKE PARQUET")
            return None
        return self.expression(
            exp.LikeProperty(this=self.expression(ParquetSchemaReference(this=reference)))
        )

    def _parse_index_params(self):
        # SQLGlot otherwise treats a Kudu PARTITION BY immediately following a
        # table-level PRIMARY KEY as an index option. Leave the key itself to
        # the existing parser, including any dialect-specific key handling.
        if (
            self._curr.token_type == TokenType.PARTITION_BY
            and self._curr.text.upper() == "PARTITION BY"
            and self._prev.token_type == TokenType.R_PAREN
        ):
            depth = 0
            for index in range(self._index - 1, -1, -1):
                token_type = self._tokens[index].token_type
                if token_type == TokenType.R_PAREN:
                    depth += 1
                elif token_type == TokenType.L_PAREN:
                    depth -= 1
                    if depth == 0:
                        if index and self._tokens[index - 1].token_type == TokenType.PRIMARY_KEY:
                            return self.expression(exp.IndexParameters())
                        break
        return super()._parse_index_params()

    def _parse_alter_table_add(self):
        if self._starts_kudu_alter_range():
            return self._parse_kudu_alter_range("ADD")
        return super()._parse_alter_table_add()

    def _parse_alter_table_drop(self):
        if self._starts_kudu_alter_range():
            return self._parse_kudu_alter_range("DROP")
        return super()._parse_alter_table_drop()

    def _starts_kudu_alter_range(self) -> bool:
        return any(
            self._match_kudu_keywords(*words, advance=False)
            for words in (
                ("RANGE", "PARTITION"),
                ("IF", "NOT", "EXISTS", "RANGE", "PARTITION"),
                ("IF", "EXISTS", "RANGE", "PARTITION"),
            )
        )

    def _parse_kudu_alter_range(self, kind: str):
        exists = False
        if self._match_kudu_keywords("IF"):
            not_ = self._match_kudu_keywords("NOT")
            if not self._match_kudu_keywords("EXISTS") or not_ != (kind == "ADD"):
                self.raise_error(f"Invalid IF EXISTS option for {kind} RANGE PARTITION")
                return []
            exists = True

        if not self._match_kudu_keywords("RANGE", "PARTITION"):
            self.raise_error("Expected RANGE PARTITION")
            return []
        partition = self._parse_kudu_range_partition()
        if self._curr:
            self.raise_error("Unexpected input after RANGE PARTITION")
        return [
            self.expression(KuduAlterRangePartition(this=partition, kind=kind, exists=exists))
        ]

    def _parse_kudu_partitioning(self):
        components = []
        saw_range = False
        saw_hash_before_range = False
        while True:
            if self._match_kudu_keywords("HASH", advance=False):
                if saw_range and saw_hash_before_range:
                    self.raise_error("HASH components cannot surround a RANGE component")
                saw_hash_before_range = saw_hash_before_range or not saw_range
                components.append(self._parse_kudu_hash())
            elif self._match_kudu_keywords("RANGE", advance=False):
                if saw_range:
                    self.raise_error("Only one RANGE component is allowed")
                saw_range = True
                components.append(self._parse_kudu_range_component())
            else:
                self.raise_error("Expected a HASH or RANGE partition component")
                break

            if self._match(TokenType.COMMA):
                if not (
                    self._match_kudu_keywords("HASH", advance=False)
                    or self._match_kudu_keywords("RANGE", advance=False)
                ):
                    self.raise_error("Expected HASH or RANGE after comma")
                    break
            elif not (
                self._match_kudu_keywords("HASH", advance=False)
                or self._match_kudu_keywords("RANGE", advance=False)
            ):
                break

        return self.expression(KuduPartitioningProperty(expressions=components))

    def _parse_kudu_identifiers(self):
        """Parse a nonempty identifier list after its opening parenthesis."""
        identifiers = []
        while True:
            identifier = self._parse_id_var(any_token=False)
            if not isinstance(identifier, exp.Identifier):
                self.raise_error("Expected a partition column identifier")
                return identifiers
            identifiers.append(identifier)
            if self._match(TokenType.R_PAREN):
                return identifiers
            if not self._match(TokenType.COMMA):
                self.raise_error("Expected a comma or closing parenthesis in partition columns")
                return identifiers

    def _parse_kudu_hash(self):
        self._match_kudu_keywords("HASH")
        columns = self._parse_kudu_identifiers() if self._match(TokenType.L_PAREN) else None

        if not self._match_kudu_keywords("PARTITIONS") or not self._match(
            TokenType.NUMBER, advance=False
        ):
            self.raise_error("Expected an integer after HASH PARTITIONS")
            return self.expression(KuduHashPartition(columns=columns))
        text = self._curr.text
        if not text or any(character < "0" or character > "9" for character in text):
            self.raise_error("HASH PARTITIONS requires an integer")
            return self.expression(KuduHashPartition(columns=columns))
        if (text.lstrip("0") or "0") in ("0", "1"):
            self.raise_error("HASH PARTITIONS requires at least two buckets")
        self._advance()
        return self.expression(
            KuduHashPartition(columns=columns, buckets=exp.Literal.number(text))
        )

    def _parse_kudu_range_component(self):
        self._match_kudu_keywords("RANGE")
        if not self._match(TokenType.L_PAREN):
            self.raise_error("Expected a parenthesized RANGE definition")
            return self.expression(KuduRangePartitioning())

        if self._match_kudu_keywords("PARTITION", advance=False):
            columns = None
        else:
            columns = self._parse_kudu_identifiers()
            if not self._match(TokenType.L_PAREN):
                self.raise_error("Expected a parenthesized list of RANGE partitions")
                return self.expression(KuduRangePartitioning(columns=columns))

        partitions = []
        while True:
            if not self._match_kudu_keywords("PARTITION"):
                self.raise_error("Expected a RANGE PARTITION")
                break
            partitions.append(self._parse_kudu_range_partition())
            if self._match(TokenType.COMMA):
                if not self._match_kudu_keywords("PARTITION", advance=False):
                    self.raise_error("Expected another RANGE PARTITION after comma")
                    break
            elif not self._match_kudu_keywords("PARTITION", advance=False):
                break

        if not self._match(TokenType.R_PAREN):
            self.raise_error("Expected closing parenthesis after RANGE partitions")
        return self.expression(KuduRangePartitioning(columns=columns, expressions=partitions))

    def _parse_kudu_bound_value(self):
        if not self._match(TokenType.L_PAREN):
            value = self._parse_bitwise()
            if value is None:
                self.raise_error("Expected a RANGE partition value")
            return value

        values = []
        while True:
            if self._match(TokenType.R_PAREN, advance=False):
                self.raise_error("Expected a value in RANGE partition parentheses")
                return None
            value = self._parse_expression()
            if value is None:
                self.raise_error("Expected a RANGE partition value")
                return None
            values.append(value)
            if self._match(TokenType.R_PAREN):
                break
            if not self._match(TokenType.COMMA):
                self.raise_error("Expected a comma or closing parenthesis in RANGE values")
                return None

        if len(values) == 1:
            return self.expression(exp.Paren(this=values[0]))
        return self.expression(exp.Tuple(expressions=values))

    def _parse_kudu_range_partition(self):
        lower = upper = value = None
        if self._match_kudu_keywords("VALUE"):
            if not self._match(TokenType.EQ):
                self.raise_error("Expected = after RANGE PARTITION VALUE")
            value = self._parse_kudu_bound_value()
        else:
            if not self._match_kudu_keywords("VALUES", advance=False):
                first = self._parse_kudu_bound_value()
                if not self._match_set(_COMPARISONS):
                    self.raise_error("Expected a comparison before VALUES")
                    return self.expression(KuduRangePartition())
                op = self._prev.token_type
                bound = self.expression(
                    KuduRangeBound(this=first, inclusive=op in (TokenType.LTE, TokenType.GTE))
                )
                if op in (TokenType.LT, TokenType.LTE):
                    lower = bound
                else:
                    upper = bound

            if not self._match_kudu_keywords("VALUES"):
                self.raise_error("Expected VALUES in RANGE PARTITION")
                return self.expression(KuduRangePartition())

            if self._match_set(_COMPARISONS):
                op = self._prev.token_type
                second = self._parse_kudu_bound_value()
                bound = self.expression(
                    KuduRangeBound(this=second, inclusive=op in (TokenType.LTE, TokenType.GTE))
                )
                if op in (TokenType.GT, TokenType.GTE):
                    if lower is not None:
                        self.raise_error("Multiple lower RANGE bounds")
                    lower = bound
                else:
                    if upper is not None:
                        self.raise_error("Multiple upper RANGE bounds")
                    upper = bound

            if lower is None and upper is None:
                self.raise_error("RANGE PARTITION requires at least one bound")

            if lower is not None and upper is not None:
                low_number = _numeric_literal(lower.this)
                high_number = _numeric_literal(upper.this)
                if low_number is not None and high_number is not None and (
                    low_number > high_number
                    or (
                        low_number == high_number
                        and not (lower.args["inclusive"] and upper.args["inclusive"])
                    )
                ):
                    self.raise_error("RANGE PARTITION has an empty numeric range")

        hashes = []
        while True:
            if self._match_kudu_keywords("HASH", advance=False):
                hashes.append(self._parse_kudu_hash())
            elif self._match(TokenType.COMMA, advance=False):
                index = self._index
                self._advance()
                if not self._match_kudu_keywords("HASH", advance=False):
                    self._retreat(index)
                    break
                hashes.append(self._parse_kudu_hash())
            else:
                break

        return self.expression(
            KuduRangePartition(lower=lower, upper=upper, value=value, hashes=hashes)
        )


class ImpalaDDLGeneratorMixin:
    """Generate the custom nodes without changing existing DDL transforms."""

    DDL_TRANSFORMS = {
        KuduPartitioningProperty: lambda self, e: self.kudupartitioningproperty_sql(e),
        KuduHashPartition: lambda self, e: self.kuduhashpartition_sql(e),
        KuduRangePartitioning: lambda self, e: self.kudurangepartitioning_sql(e),
        KuduRangeBound: lambda self, e: self.kudurangebound_sql(e),
        KuduRangePartition: lambda self, e: self.kudurangepartition_sql(e),
        KuduAlterRangePartition: lambda self, e: self.kudualterrangepartition_sql(e),
        ParquetSchemaReference: lambda self, e: self.parquetschemareference_sql(e),
    }
    DDL_PROPERTY_LOCATIONS = {
        KuduPartitioningProperty: exp.Properties.Location.POST_SCHEMA,
    }

    def kudupartitioningproperty_sql(self, expression: KuduPartitioningProperty) -> str:
        return "PARTITION BY " + ", ".join(self.sql(e) for e in expression.expressions)

    def kuduhashpartition_sql(self, expression: KuduHashPartition) -> str:
        columns = expression.args.get("columns")
        column_sql = (
            " (" + ", ".join(self.sql(column) for column in columns) + ")"
            if columns is not None
            else ""
        )
        return f"HASH{column_sql} PARTITIONS {self.sql(expression, 'buckets')}"

    def kudurangepartitioning_sql(self, expression: KuduRangePartitioning) -> str:
        columns = expression.args.get("columns")
        column_sql = (
            " (" + ", ".join(self.sql(column) for column in columns) + ")"
            if columns is not None
            else ""
        )
        partitions = ", ".join(self.sql(partition) for partition in expression.expressions)
        return f"RANGE{column_sql} ({partitions})"

    def kudurangebound_sql(self, expression: KuduRangeBound) -> str:
        return self.sql(expression, "this")

    def kudurangepartition_sql(self, expression: KuduRangePartition) -> str:
        value = expression.args.get("value")
        if value is not None:
            specification = f"VALUE = {self.sql(value)}"
        else:
            lower = expression.args.get("lower")
            upper = expression.args.get("upper")
            specification = "VALUES"
            if lower is not None:
                operator = "<=" if lower.args.get("inclusive") else "<"
                specification = f"{self.sql(lower)} {operator} {specification}"
            if upper is not None:
                operator = "<=" if upper.args.get("inclusive") else "<"
                specification = f"{specification} {operator} {self.sql(upper)}"
        hashes = expression.args.get("hashes") or []
        hash_sql = " " + ", ".join(self.sql(item) for item in hashes) if hashes else ""
        return f"PARTITION {specification}{hash_sql}"

    def kudualterrangepartition_sql(self, expression: KuduAlterRangePartition) -> str:
        kind = expression.args["kind"]
        exists = ""
        if expression.args.get("exists"):
            exists = " IF NOT EXISTS" if kind == "ADD" else " IF EXISTS"
        return f"{kind}{exists} RANGE {self.sql(expression, 'this')}"

    def parquetschemareference_sql(self, expression: ParquetSchemaReference) -> str:
        return f"PARQUET {self.sql(expression, 'this')}"
