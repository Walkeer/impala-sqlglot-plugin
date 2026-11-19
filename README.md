# impala-sqlglot-plugin

Impala dialect plugin for [sqlglot](https://github.com/tobymao/sqlglot).

This package provides an Impala dialect for sqlglot, enabling SQL parsing and generation with Impala-specific syntax and features.

## Installation

```bash
uv pip install impala-sqlglot-plugin
```

## Usage

After installing the package, the `impala` dialect is automatically registered with sqlglot:

```python
import sqlglot

result = sqlglot.transpile("SELECT * FROM users WHERE id = 1", read="hive", write="impala")[0]
print(result)
# Output: SELECT * FROM users WHERE id = 1
```

## Development

See [Impala-specific syntax and analysis](docs/impala_syntax.md) for structured
UPSERT, plan hints, statistics/metadata, Kudu partitioning, schema references
and function signatures, including validation and compatibility limits.

This project uses [uv](https://docs.astral.sh/uv/) and supports Python 3.11 and 3.12.

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.11 and 3.12 (uv can install them for you)

### Setup

```bash
git clone https://github.com/your-org/impala-sqlglot-plugin.git
cd impala-sqlglot-plugin

# Install Python versions and create both virtual environments
uv python install 3.11 3.12
UV_PROJECT_ENVIRONMENT=.venv-3.11 uv sync --python 3.11
UV_PROJECT_ENVIRONMENT=.venv-3.12 uv sync --python 3.12
```

The default local environment is Python 3.12 (see `.python-version`):

```bash
uv sync
```

### Running Tests

Run tests in a specific environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-3.11 uv run --python 3.11 pytest
UV_PROJECT_ENVIRONMENT=.venv-3.12 uv run --python 3.12 pytest
```

Or use the default 3.12 environment:

```bash
uv run pytest
```
