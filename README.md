# loaf

a terminal-first ai assistant cli built with ink + react.

> this project is entirely vibecoded using codex.

## what it does

- supports additive auth:
  - `openai oauth` (codex account auth)
  - `openrouter api key`
- first-run step-by-step onboarding:
  - choose auth providers
  - optionally add exa key for web search
- auto-fetches models from providers and caches them locally
- `/model` flow:
  - searchable model picker
  - thinking level picker
  - openrouter provider routing picker (`any` + forced providers when available)
- python-first tool runtime:
  - `run_py`
  - `install_pip`
  - `run_py_module`
- optional web search tool via exa (`search_web`) with highlights
- browser automation tools (playwright-backed)
- automatic conversation reset when switching model providers

## quick start

```bash
npm install
npm run dev
```

## slash commands

- `/auth` add another auth provider
- `/onboarding` rerun setup flow
- `/forgeteverything` wipe local config and restart onboarding
- `/model` choose model, thinking level, and (for openrouter) routing provider
- `/history` list/resume saved chats (`/history`, `/history last`, `/history <id>`)
- `/tools` list registered tools
- `/clear` clear conversation messages
- `/help` show command list

## configuration

no env vars are required for normal use.

- use onboarding + `/auth` for provider setup
- use `/model` for model/thinking/provider routing
- use `/onboarding` if you want to reconfigure search/auth later

## local data

on windows, state is persisted under `%APPDATA%\loaf\`:

- `state.json` selected model/thinking/auth, input history, onboarding state
- `auth.json` openai oauth token bundle
- `models-cache.json` provider model cache
- `sessions/YYYY/MM/DD/rollout-*.jsonl` saved chat sessions (used by `/history`)
- `python-runtime/` managed python runtime + venv + transient scripts

## notes

- search guidance is only injected when exa is configured.
- tools live in `src/tools/`.
