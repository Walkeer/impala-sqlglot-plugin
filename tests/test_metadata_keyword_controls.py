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


@pytest.mark.parametrize("key", [
    "PRIMARY KEY (demo_key)",
    "FOREIGN KEY (demo_key) REFERENCES demo_parent (demo_key)",
])
@pytest.mark.parametrize("flags", [
    "N'DISABLE' NOVALIDATE RELY",
    "DISABLE N'NOVALIDATE' RELY",
    "DISABLE NOVALIDATE N'RELY'",
    "N'ENABLE' VALIDATE NORELY",
    "ENABLE N'VALIDATE' NORELY",
    "ENABLE VALIDATE N'NORELY'",
])
def test_national_strings_do_not_become_table_key_flags(key, flags):
    source = f"CREATE TABLE demo (demo_key BIGINT, {key} {flags}) STORED AS PARQUET"
    with pytest.raises((ParseError, TokenError)):
        sqlglot.parse_one(source, read="impala")


@pytest.mark.parametrize("flags", ["N'NOT' ENFORCED", "NOT N'ENFORCED'"])
def test_national_strings_do_not_become_not_enforced(flags):
    source = (
        f"CREATE TABLE demo (demo_key BIGINT, PRIMARY KEY (demo_key) {flags}) "
        "STORED AS PARQUET"
    )
    with pytest.raises((ParseError, TokenError)):
        sqlglot.parse_one(source, read="impala")
