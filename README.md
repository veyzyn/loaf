# loaf

a terminal-first ai assistant cli built with ink + react.

> this project is entirely vibecoded using codex.

## what it does

- supports additive auth:
  - `openai oauth` (codex account auth)
    - auto-falls back to device-code login on headless/ssh environments
  - `openrouter api key`
- first-run step-by-step onboarding:
  - choose auth providers
  - optionally add exa key for web search
- auto-fetches models from providers and caches them locally
- `/model` flow:
  - searchable model picker
  - thinking level picker
  - openrouter provider routing picker (`any` + forced providers when available)
- local command + javascript tool runtime:
  - `bash`
  - `run_js`
  - `install_js_packages`
  - `run_js_module`
  - `start_background_js` / `read_background_js` / `write_background_js` / `stop_background_js` / `list_background_js`
    - `start_background_js` supports `session_name` + `reuse_session` (default true) to avoid duplicate interactive sessions
  - `create_persistent_tool` (creates + autoloads custom js tools)
- optional web search tool via exa (`search_web`) with highlights
- custom js tool loading from user folders
- skill loading from repo `.agents/skills/<skill-name>/SKILL.md`, `~/.loaf/skills/<skill-name>/SKILL.md`, and `~/.agents/skills/<skill-name>/SKILL.md`
- automatic skill matching on each prompt (+ explicit `$skill-name` mentions)
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
- `/skills` list available skills from repo `.agents/skills`, `~/.loaf/skills`, and `~/.agents/skills` with description previews
- `/tools` list registered tools
- `/clear` clear conversation messages
- `/help` show command list
- `/quit` exit loaf
- `/exit` exit loaf

## skill usage

- place skills at repo `.agents/skills/<skill-name>/SKILL.md`, `~/.loaf/skills/<skill-name>/SKILL.md`, or `~/.agents/skills/<skill-name>/SKILL.md`
- use `/skills` to inspect what loaf found
- start input with `$` to open skill autocomplete (`enter` / `tab` / `up` / `down`)
- mention `$skill-name` in your prompt for explicit usage
- loaf also auto-matches relevant skills from descriptions on each prompt
- model-facing prompts transform mentions to `use $skill-name skill`, while your visible transcript keeps original `$skill-name` text

## configuration

no env vars are required for normal use.

- use onboarding + `/auth` for provider setup
- use `/model` for model/thinking/provider routing
- use `/onboarding` if you want to reconfigure search/auth later

## local data

loaf persists data in a single home directory:

- macos/linux: `~/.loaf`
- windows: `%USERPROFILE%\.loaf`

- `state.json` selected model/thinking/auth, input history, onboarding state
- `auth.json` openai oauth token bundle
- `models-cache.json` provider model cache
- `sessions/YYYY/MM/DD/rollout-*.jsonl` saved chat sessions (used by `/history`)
- `js-runtime/` transient js scripts created by `run_js`
- `js-runtime/background/` transient js scripts created by `start_background_js`
- `tools/` user-provided js tools (auto-loaded)
- `skills/<skill-name>/SKILL.md` user-provided skill instructions in `~/.loaf` (auto-discovered)
- `~/.agents/skills/<skill-name>/SKILL.md` alternate skill root (also auto-discovered)

## notes

- search guidance is only injected when exa is configured.
- tools live in `src/tools/`.
- custom tools are discovered only from `<loaf data dir>/tools`.
- detailed custom tool docs: `CUSTOM_TOOLS.md`
- while the agent is running, use `enter` to queue a follow-up message, `shift+enter` to steer, or `esc` to interrupt.
