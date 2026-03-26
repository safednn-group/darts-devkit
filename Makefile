.PHONY: format lint check docs

format:
	uv run ruff format src/ tests/
	uv run ruff check src/ --fix

lint:
	uv run ruff check src/

check: lint
	uv run mypy src/

test:
	uv run pytest -v --random-order --cov=darts --cov-report=html -vv

docs:
	rm -rf docs/build
	uv run sphinx-build -b html docs/source docs/build/html