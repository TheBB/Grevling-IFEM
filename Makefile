.PHONY: lint format pytest myypy test install wheel sdist build

package := grevling_ifem
testpackage := tests

lint:
	poetry run ruff check $(package)
	poetry run ruff check $(testpackage)
	poetry run ruff format --check $(package)
	poetry run ruff format --check $(testpackage)

format:
	poetry run ruff check --fix $(package)
	poetry run ruff check --fix $(testpackage)
	poetry run ruff format $(package)
	poetry run ruff format $(testpackage)

pytest:
	poetry run pytest

mypy:
	poetry run mypy $(package)

test: mypy lint pytest

install:
	poetry install --with matplotlib,plotly,dev

wheel:
	poetry build -f wheel

sdist:
	poetry build -f sdict

build: wheel sdist
