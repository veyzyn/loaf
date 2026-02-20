# loaf

`loaf` is a terminal-first AI coding assistant built with Ink + React.
It supports multi-provider model auth, local tool execution, skills, chat history, and optional web search.

## Highlights

- OpenAI OAuth and OpenRouter API key support in one setup.
- Guided onboarding for first-time configuration.
- Searchable model picker with reasoning-level controls.
- Built-in shell and JavaScript execution tools.
- Optional `search_web` via Exa.
- Skill loading with explicit `$skill-name` usage.
- Resumable chat history and image attachments.

## Requirements

- Node.js `>=20`
- npm

## Quick Start

```bash
npm install
npm run dev
```

On first launch, `loaf` runs onboarding so you can configure providers and (optionally) Exa search.

## Command Reference

| Command | Description |
| --- | --- |
| `/auth` | Add or update auth providers. |
| `/onboarding` | Re-run setup flow (auth + Exa key). |
| `/forgeteverything` | Wipe local config and restart onboarding. |
| `/model` | Choose model, thinking level, and OpenRouter routing provider. |
| `/history` | Resume saved chats (`/history`, `/history last`, `/history <id>`). |
| `/skills` | List discovered skills and description previews. |
| `/tools` | List registered tools. |
| `/clear` | Clear current conversation messages. |
| `/help` | Show command list. |
| `/quit` | Exit `loaf`. |
| `/exit` | Exit `loaf`. |

## Skills and Tools

- Use `/skills` to list discovered skills and mention `$skill-name` to apply one.
- Use `/tools` to list all currently registered tools.
- Custom tools are supported. See `CUSTOM_TOOLS.md` and `src/tools/README.md`.

## Configuration

- Use `/auth` to add or update providers.
- Use `/model` to switch model and reasoning level.
- Use `/onboarding` any time to re-run setup.

## Development

```bash
npm run dev
npm run typecheck
npm run test
npm run test:coverage
```

## License

[MIT](LICENSE)
