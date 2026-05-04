# vote_simulation

![PyPI version](https://img.shields.io/pypi/v/vote_simulation.svg)

A framework to compare social choice rules

* Created by **[Ambre Laqueuvre](https://github.com/Ambraser)**
  * PyPI: https://pypi.org/user/ambraser/
* PyPI package: https://pypi.org/project/vote_simulation/
* Free software: MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://Ambraser.github.io/vote-simulation/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/vote_simulation.git
cd vote_simulation

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `vote_simulation`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

vote_simulation was created in 2026 by Ambre Laqueuvre.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
