# DARTS-devkit

## Development

### First installation

```
# One time install of 'uv'
curl -LsSf https://astral.sh/uv/install.sh | sh

cd darts-devkit
uv sync --all-extras
```

### Run examples

```
# Editable for development - changes in package reflected instantly
uv run main.py
# Non-Editable - can delete src/darts_devkit but code will still work
uv pip install .[dev,docs]
source .venv/bin/activate
python main.py
```

### Quality check

```
make format
make check
```

### Building documentation

```
make docs
```