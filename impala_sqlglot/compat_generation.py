"""Narrow Impala generation fixes for view, function, describe and admin SQL.

The parser already records the fields needed here.  The empty parser mixin
provides a symmetric integration point without changing parsing behavior.
"""
from __future__ import annotations

from sqlglot import exp


class ImpalaGenerationParserMixin:
    """No parser override is required for these generation fixes."""


class _AlterViewValues(exp.Expression, exp.Query):
    """Temporary wrapper so SQLGlot's ALTER generator emits its AS prefix."""

    arg_types = {"this": True}


class ImpalaGenerationGeneratorMixin:
    """Render only the affected Impala constructs, delegating all others."""

    GENERATION_TRANSFORMS = {
        _AlterViewValues: lambda self, expression: self.sql(expression, "this"),
    }
    _NATIVE_FUNCTION_OPTIONS = frozenset({"SYMBOL", "PREPARE_FN", "CLOSE_FN"})

    def alter_sql(self, expression: exp.Alter) -> str:
        if (expression.args.get("kind") or "").upper() != "VIEW":
            return super().alter_sql(expression)
        if not any(isinstance(action, exp.Values) for action in expression.actions):
            return super().alter_sql(expression)

        # Keep SQLGlot's handling of the target, options and formatting; only
        # change how a VALUES action is presented to the generic ALTER writer.
        rewritten = expression.copy()
        rewritten.set(
            "actions",
            [
                _AlterViewValues(this=action) if isinstance(action, exp.Values) else action
                for action in rewritten.actions
            ],
        )
        return super().alter_sql(rewritten)

    def describe_sql(self, expression: exp.Describe) -> str:
        sql = super().describe_sql(expression)
        if (expression.args.get("kind") or "").upper() != "DATABASE":
            return sql
        if not sql.startswith("DESCRIBE"):
            raise ValueError("Unexpected superclass DESCRIBE output")
        return "DESCRIBE DATABASE" + sql[len("DESCRIBE") :]

    def properties_sql(self, expression: exp.Properties) -> str:
        parent = expression.parent
        if not isinstance(parent, exp.Create) or (parent.args.get("kind") or "").upper() != "FUNCTION":
            return super().properties_sql(expression)

        def is_native_option(prop: exp.Expression) -> bool:
            return (
                type(prop) is exp.Property
                and isinstance(prop.this, exp.Var)
                and prop.this.name.upper() in self._NATIVE_FUNCTION_OPTIONS
            )

        if not any(is_native_option(prop) for prop in expression.expressions):
            return super().properties_sql(expression)

        parts = []
        ordinary = []

        def flush_ordinary() -> None:
            if ordinary:
                props = exp.Properties(expressions=[prop.copy() for prop in ordinary])
                props.parent = parent
                parts.append(super(ImpalaGenerationGeneratorMixin, self).properties_sql(props))
                ordinary.clear()

        for prop in expression.expressions:
            if not is_native_option(prop):
                ordinary.append(prop)
                continue
            flush_ordinary()
            option = f"{self.sql(prop, 'this')}={self.sql(prop, 'value')}"
            parts.append(self.maybe_comment(option, prop))
        flush_ordinary()
        return self.sep().join(part for part in parts if part)

    def command_sql(self, expression: exp.Command) -> str:
        if expression.this == ":":
            return f":{expression.text('expression')}"
        return super().command_sql(expression)
