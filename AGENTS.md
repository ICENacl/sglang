# Repository Guidelines

## Project Structure & Module Organization
- `python/`: Core SGLang Python package (runtime, models, server entrypoints).
- `sgl-kernel/`: Low-level kernel package (CPU/GPU ops and extensions).
- `sgl-model-gateway/`: Rust gateway and mesh/router components.
- `test/`: Test suites (SRT backend, language frontend, registered CI tests).
- `docs/`: Documentation sources and notebooks.
- `benchmark/`: Benchmarks and evaluation harnesses.
- `scripts/`, `docker/`, `assets/`: CI helpers, images, and static assets.

## Build, Test, and Development Commands
- Install from source (editable): `pip install -e "python"`.
- Run a local server (example): `python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct`.
- Run unit tests (backend): `python3 test/srt/test_srt_endpoint.py`.
- Run a test suite: `python3 test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu`.
- Docs build (optional): `cd docs && make html` or `bash serve.sh` for live preview.

## Coding Style & Naming Conventions
- Indentation: 4 spaces by default; 2 spaces for `*.md`, `*.json`, `*.yml`. Use LF and trim trailing whitespace (`.editorconfig`).
- Use `pre-commit` before pushing: `pre-commit run --all-files`.
- Prefer small, focused files and avoid duplication in hot paths (see `docs/developer_guide/contribution_guide.md`).
- Tests: name files `test_*.py` and keep test functions descriptive.

## Testing Guidelines
- Framework: Python `unittest` (some pytest in utilities).
- Place backend tests in `test/srt/`, frontend tests in `test/lang/`, and registered CI tests in `test/registered/`.
- Ensure new tests are wired into the relevant `run_suite.py` so CI picks them up.

## Commit & Pull Request Guidelines
- Commit message style in history: short, imperative, often with scope tags like `[NPU]`, `[diffusion]`, `fix(...)`, `feat:` and a PR number suffix `(#12345)`.
- Do not commit directly to `main`; create a feature branch and open a PR.
- PRs should include: clear description, relevant tests, and doc updates when applicable. CI is label-gated; coordinate with maintainers if you cannot trigger CI.

## Optional Notes
- For sgl-kernel changes, follow the multi-PR flow described in `docs/developer_guide/contribution_guide.md`.

# 系统要求
- Avoid over-engineering. Only make changes that are directly requested or clearly necessary. Keep solutions
simple and focused.
- Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability. Don't add docstrings, comments, or type annotations to code you didn't change. Only add comments where the logic isn't self-evident.
- Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use feature flags or backwards-compatibility shims when you can just change the code.
- Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task—three similar lines of code is better than a premature abstraction.
- Avoid backwards-compatibility hacks like renaming unused _vars, re-exporting types, adding // removed comments for removed code, etc. If you are certain that something is unused, you can delete it completely.
- When communicating your results back to me, explain what you did and what happened in plain, clear English. Avoid jargon, technical implementation details, and code-speak in your final responses. Write as if you're explaining to a smart person who isn't looking at the code. Your actual work (how you think, plan, write code, debug, and solve problems) should stay fully technical and rigorous. This only applies to how you talk to me about it.
- Before reporting back to me, if at all possible, verify your own work. Don't just write code and assume it's done. Actually test it using the tools available to you. If possible, run it, check the output, and confirm it does what was asked. If you're building something visual like a web app, view the pages, click through the flows, and check that things render and behave correctly. If you're writing a script, run it against real or representative input and inspect the results. If there are edge cases you can simulate, try them.
- Define finishing criteria for yourself before you start: what does "done" look like for this task? Use that as your checklist before you come back to me. If something fails or looks off, fix it and re-test. Don't just flag it and hand it back. The goal is to keep me out of the loop on iteration. I want to receive finished, working results, not a first draft that needs me to spot-check it. Only come back to me when you've confirmed things work, or when you've genuinely hit a wall that requires my input.

# 改动代码时的要求
- 修改代码后，及时修改相关项目文档
- 所有输出的内容都使用中文
- 当前的环境不支持cuda，如果需要对环境进行判断，你需要临时输出代码，由我给你返回结果

# 对于plan mode
- 在plan mode结束后，立即将相关方案输出到md文档

# 可以参考的其他开源项目
1. trt-llm: /config/workspace/TensorRT-LLM
2. vllm: /config/workspace/vllm
