.PHONY: format check-lint check-types test install clean

format:
	black --line-length 100 src/ tests/ notebooks/

check-lint:
	flake8 --config config/flake8.cfg src/ tests/

check-types:
	mypy --config-file config/mypy.ini src/

test:
	pytest tests/ -v --cov=src --cov-report=html

install:
	pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage
