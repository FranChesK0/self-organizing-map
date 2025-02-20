.PHONY: help clean format lint check pc run

.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  help    - Display this message"
	@echo "  clean   - Remove all build artifacts and cache files"
	@echo "  format  - Format code with isort and black"
	@echo "  lint    - Lint code with flake8 and mypy"
	@echo "  check   - Run format, lint and tests"
	@echo "  pc      - Run pre-commit hooks"
	@echo "  run     - Run application"

clean:
	uv clean
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

format:
	uv run ruff format

lint:
	uv run ruff check
	uv run mypy .

check: format lint

pc:
	pre-commit run --all --hook-stage pre-push

run:
	uv run python src/main.py
