.PHONY: format lint check docs

format:
	uv run ruff format src/
	uv run ruff check src/ --fix

lint:
	uv run ruff check src/

check: lint
	uv run mypy src/

docs:
	rm -rf docs/build
	uv run sphinx-build -b html docs/source docs/build/html