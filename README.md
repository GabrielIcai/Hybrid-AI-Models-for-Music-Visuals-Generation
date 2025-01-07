# template-project

## What to do at first

At first, create the virtual environment:

    python -m venv .venv

Then, activate the virtual environment:

    source .venv/bin/activate

Each time a new package is included use:

    pip install package


## Sanity Checks

They are executed in each push, but if you want to check code and typing style before pushing please follow these steps:

    pytest .
    black --check .
    ruff check
    mypy src tests
    flake8 src tests
    pylint src tests


## Others

To see the documentation run:

    mkdocs serve
