# Repository Guidelines

## Project Structure & Module Organization
- `assistant_axis/` contains the core Python library. Key modules include `axis.py`, `steering.py`, and `judge.py`, with lower-level helpers in `assistant_axis/internals/`.
- `assistant_axis/tests/` holds pytest tests (files named `test_*.py`).
- `pipeline/` contains the 5-step axis computation scripts (`1_generate.py` through `5_axis.py`) plus `run_pipeline.sh` and a focused `pipeline/README.md`.
- `data/` stores role and trait instruction JSON plus extraction questions.
- `notebooks/` holds exploratory analysis notebooks.
- `transcripts/` and `img/` contain paper assets and example conversations.

## Build, Test, and Development Commands
- Install dependencies (recommended):
```bash
uv sync
```
- Run the full axis pipeline script:
```bash
cd pipeline
./run_pipeline.sh
```
- Run a single pipeline step (example):
```bash
cd pipeline
uv run 1_generate.py --model google/gemma-2-27b-it --output_dir outputs/gemma-2-27b/responses
```
- Run tests:
```bash
uv run pytest
```

## Coding Style & Naming Conventions
- Language: Python 3.10+ with 4-space indentation.
- Favor concise, direct implementations; avoid unnecessary abstractions.
- Avoid docstrings, extra logging, or heavy validation unless logic is non-obvious.
- Prefer simple module-level functions and small classes.
- No formatter or linter is enforced in this repo.

## Testing Guidelines
- Framework: `pytest` (declared in `pyproject.toml` under dev deps).
- Test files live in `assistant_axis/tests/` and should follow `test_*.py` naming.
- Keep tests small and fast; avoid GPU requirements unless explicitly scoped.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and sentence-case (e.g., “Add tensor parallel size configuration to scripts”).
- Keep commits scoped to a single change; avoid bundling data and code unless necessary.
- Pull requests should include a clear summary, reproduction steps for new scripts, and any new data paths or configs. If a change depends on secrets, note required env vars (e.g., `OPENAI_API_KEY`).

## Configuration & Runtime Notes
- Some pipeline steps require GPU resources and model checkpoints.
- The judge step uses OpenAI and requires `OPENAI_API_KEY` in the environment.
- See `pipeline/README.md` for step-by-step pipeline details and flags.

## Agent-Specific Instructions
- If you are using agent guidance in this repo, read `CLAUDE.md` and follow its constraints for code style and scope.
