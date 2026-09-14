# Publish Knowledge Hub v1 on GitHub

## Before publishing

1. Run `git status` and review every file to be committed.
2. Confirm `config/settings.yaml` contains environment-variable placeholders rather than real API keys.
3. Do not add `.env.local`, `.venv`, `data/`, or `logs/` to Git.
4. Run the tests: `python -m pytest -q`.

## Create and push the repository

Create an empty GitHub repository named `Knowledge_Hub_v1`. Do not initialize it with a README, `.gitignore`, or license, because this local repository already contains its first release.

```powershell
git remote set-url origin https://github.com/<YOUR_GITHUB_USERNAME>/Knowledge_Hub_v1.git
git branch -M main
git push -u origin main
git push origin v1.0.0
```

After the first push, add a concise repository description such as:

> A modular RAG knowledge hub with hybrid retrieval, MCP tools, and an observable Streamlit dashboard.

## Recommended repository topics

`rag` · `mcp` · `llm` · `streamlit` · `knowledge-base` · `retrieval-augmented-generation`

## License

Choose a license deliberately before publishing. No license file is included in this release, so reuse permissions are not granted by default.
