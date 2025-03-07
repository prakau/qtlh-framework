# QTL-H Framework Makefile

.PHONY: help install test lint format docs clean dist

help:
	@echo "qtlh-framework development tasks:"
	@echo ""
	@echo "make install        - Install for development with all dependencies"
	@echo "make test          - Run all tests"
	@echo "make test-cov      - Run tests with coverage report"
	@echo "make lint          - Check code style"
	@echo "make format        - Format code"
	@echo "make docs          - Build documentation"
	@echo "make clean         - Clean build artifacts"
	@echo "make dist          - Create distribution package"
	@echo "make gpu-test      - Run tests requiring GPU"
	@echo "make benchmark     - Run performance benchmarks"

install:
	pip install -e ".[dev,docs]"

test:
	pytest -v tests/ -m "not gpu and not slow"

test-cov:
	pytest -v tests/ -m "not gpu and not slow" --cov=qtlh --cov-report=html --cov-report=term-missing

lint:
	flake8 src/qtlh tests
	mypy src/qtlh tests
	black --check src/qtlh tests

format:
	black src/qtlh tests
	isort src/qtlh tests

docs:
	$(MAKE) -C docs html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

dist: clean
	python setup.py sdist bdist_wheel

gpu-test:
	pytest -v tests/ -m "gpu"

benchmark:
	pytest benchmarks/ --benchmark-only

check-all: lint test test-cov

# Development environment setup
dev-setup: install
	pre-commit install
	
# Run only quantum tests
test-quantum:
	pytest -v tests/quantum/

# Run only HD computing tests
test-hd:
	pytest -v tests/hd/

# Run only topology tests
test-topology:
	pytest -v tests/topology/

# Run only language model tests
test-language:
	pytest -v tests/language/

# Run only integration tests
test-integration:
	pytest -v tests/integration/

# Run only validation tests
test-validation:
	pytest -v tests/validation/

# Run the full pipeline test
test-pipeline:
	pytest -v tests/test_pipeline.py

# Generate test data
test-data:
	python scripts/generate_test_data.py

# Type checking
type-check:
	mypy src/qtlh --strict

# Security checks
security-check:
	bandit -r src/qtlh/

# Dependencies audit
deps-check:
	safety check

# Performance profiling
profile:
	python -m cProfile -o profile.stats scripts/profile_pipeline.py
	snakeviz profile.stats

# Docker commands
docker-build:
	docker build -t qtlh-framework .

docker-run:
	docker run -it --rm qtlh-framework

# Continuous Integration tasks
ci: lint type-check test test-cov security-check

# Release tasks
release: ci clean dist
	twine check dist/*

# Create new version (usage: make version VERSION=X.Y.Z)
version:
	@test $(VERSION) || (echo "VERSION is required"; exit 1)
	sed -i "s/version=.*/version='$(VERSION)',/" setup.py
	git add setup.py
	git commit -m "Bump version to $(VERSION)"
	git tag v$(VERSION)

.DEFAULT_GOAL := help
