"""Impala Tokenizer — extends Hive, removes Hive-only tokens."""
from __future__ import annotations
import re

from sqlglot import tokens
from sqlglot.dialects.hive import Hive
from sqlglot.tokens import TokenType


# Multi-word keyword phrases to REMOVE from Hive's keyword table
# (keys that exist in Hive.Tokenizer.KEYWORDS but must not apply to Impala)
_HIVE_ONLY_KEYWORDS: frozenset[str] = frozenset(
    [
        # Hive-specific commands
        "ADD ARCHIVE",
        "ADD ARCHIVES",
        "ADD FILE",
        "ADD FILES",
        "ADD JAR",
        "ADD JARS",
        "MSCK REPAIR",
        # Hive time-travel syntax — Impala uses FOR SYSTEM_TIME/VERSION AS OF
        "TIMESTAMP AS OF",
        "VERSION AS OF",
    ]
)

# Additional multi-word keyword phrases to ADD for Impala
_IMPALA_KEYWORDS: dict[str, TokenType] = {
    # Share dispatch, while the parser preserves the distinct UPSERT operation.
    "UPSERT": TokenType.INSERT,
    # Share dispatch, while the parser preserves NON UNIQUE.
    "NON UNIQUE PRIMARY KEY": TokenType.PRIMARY_KEY,
    # Impala admin statements parsed as commands
    "COPY TESTCASE": TokenType.COMMAND,
}


class ImpalaTokenizer(Hive.Tokenizer):
    """Impala SQL tokenizer.

    Differences from Hive:
    - No $ PARAMETER tokens
    - No numeric literal suffixes (1L, 1BD, etc.)
    - No ADD JAR/FILE/ARCHIVE or MSCK REPAIR keywords
    - FOR SYSTEM_TIME/VERSION AS OF replaces Hive's TIMESTAMP/VERSION AS OF
    """

    # Remove Hive's $ parameter single-token mapping
    SINGLE_TOKENS = {
        k: v
        for k, v in Hive.Tokenizer.SINGLE_TOKENS.items()
        if v != TokenType.PARAMETER
    }

    # No Hive numeric suffixes
    NUMERIC_LITERALS: dict[str, str] = {}

    # Impala also allows comment hints after JOIN and after a write target.
    # Keep hint tokens at other positions too, so the parser can reject them
    # instead of silently degrading them into ordinary comments.
    TOKENS_PRECEDING_HINT = set(TokenType)

    KEYWORDS = {
        **{k: v for k, v in Hive.Tokenizer.KEYWORDS.items() if k not in _HIVE_ONLY_KEYWORDS},
        **_IMPALA_KEYWORDS,
    }

    def tokenize(self, sql, *args, **kwargs):
        result = []
        for token in super().tokenize(sql, *args, **kwargs):
            if (
                token.token_type == TokenType.HINT and token.comments
                and any(character in token.comments[0] for character in ("\r", "\n"))
                and result
            ):
                # The native scanner treats multiline block hints as comments.
                result[-1].comments.extend(["+" + token.comments[0], *token.comments[1:]])
                continue
            hints = []
            ordinary = []
            for comment in token.comments:
                match = re.match(r"^[ \t]*\+[ \t]*([^\r\n]*?)[ \t]*$", comment)
                if match:
                    hints.append(match.group(1))
                else:
                    ordinary.append(comment)
            token.comments = ordinary
            result.append(token)
            for hint in hints:
                result.append(
                    tokens.Token(
                        TokenType.HINT, "/*+" + hint + "*/", line=token.line,
                        col=token.col, start=token.end, end=token.end, comments=[hint],
                    )
                )
        return result
