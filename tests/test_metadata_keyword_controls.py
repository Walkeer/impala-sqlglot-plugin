"""Synthetic controls: literal contents must never activate metadata keywords."""
import pytest
import sqlglot
from sqlglot.errors import ParseError, TokenError
from sqlglot.tokens import Token, TokenType

import impala_sqlglot  # noqa: F401
from impala_sqlglot.metadata import _metadata_word


@pytest.mark.parametrize("token_type", [
    token_type for token_type in TokenType
    if token_type.name == "STRING" or token_type.name.endswith("_STRING")
])
def test_metadata_keyword_matching_excludes_every_string_token(token_type):
    assert not _metadata_word(Token(token_type, "PARTITIONS"), "PARTITIONS")


@pytest.mark.parametrize("source", [
    "ALTER TABLE demo RECOVER N'PARTITIONS'",
    "ALTER TABLE demo DROP PARTITION (demo_key = 1) N'PURGE'",
    "ALTER TABLE demo SET CACHED N'IN' 'invented_pool'",
    "SHOW TABLE N'STATS' demo",
    "SHOW CREATE TABLE demo WITH N'STATS'",
    "COMPUTE N'STATS' demo",
    "COMPUTE INCREMENTAL N'STATS' demo",
    "INVALIDATE N'METADATA' demo",
    "REFRESH N'AUTHORIZATION'",
])
def test_national_string_contents_are_not_native_metadata_keywords(source):
    with pytest.raises((ParseError, TokenError)):
        sqlglot.parse_one(source, read="impala")
