# Contributing to Knowledge Hub v1

Thanks for improving the project. Please keep changes small, testable, and focused.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before opening a pull request

```powershell
python -m pytest -q
```

- Do not commit API keys, `.env` files, `data/`, or `logs/`.
- Add or update tests for behavioral changes.
- Describe the problem, the solution, and the verification result in the pull request.
- Keep provider-specific code behind the existing factory interfaces where possible.
