# loaf usage guide

## 1) Install and run

Requirements:

- Node.js `>=20`
- npm

Start the TUI:

```bash
npm install
npm run dev
```

Start RPC server mode (stdio JSON-RPC):

```bash
npm run rpc
```

## 2) First launch and onboarding

On first launch, `loaf` opens onboarding.

Typical setup flow:

1. Connect `openai` OAuth and/or set an `openrouter` API key.
2. (Optional) set Exa API key for `search_web`.
3. Pick model + thinking level.

You can re-run onboarding any time with `/onboarding`.

## 3) Everyday workflow

- Type a normal prompt and press `Enter`.
- Use slash commands for local actions (`/model`, `/history`, `/tools`, etc.).
- Use `Up/Down` to navigate prior prompt history when input does not start with `/`.
- Type `/` to open command suggestions. Use `Tab` to autocomplete.

## 4) Skills in prompts

- Type `$` to start a skill mention (example: `$frontend-design build a landing page`).
- While typing `$...`, `loaf` shows skill suggestions.
- Use `Tab` (or `Enter` while completing the skill token) to autocomplete the skill name.
- Use `/skills` to view discovered skills.

See [skills.md](skills.md) for discovery rules and troubleshooting.

## 5) Image attachments

- Paste with `Ctrl+V` (`control+v` on macOS UI label) to attach an image from clipboard.
- Clipboard can contain image bytes or an image file path.
- Supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`.
- Limits: up to 4 attached images per prompt, max 8 MB per image.

## 6) Slash commands

| Command | What it does |
| --- | --- |
| `/auth` | Show auth status and auth-flow guidance. |
| `/onboarding` | Show onboarding status / restart setup flow in UI. |
| `/forgeteverything` | Clear local state/secrets and reset onboarding. |
| `/model` | Show current model/thinking and available model options. |
| `/limits` | Fetch provider usage/limit snapshot (when available). |
| `/history` | List recent sessions. |
| `/history last` | Resume latest saved session. |
| `/history <id>` | Resume a specific saved session ID. |
| `/skills` | List discovered skills. |
| `/tools` | List registered tools (built-in + custom). |
| `/clear` | Clear active conversation messages/history from the current session. |
| `/help` | Show command list. |
| `/quit`, `/exit` | Shutdown `loaf`. |

## 7) Data location

Runtime data is stored under:

- macOS/Linux: `~/.loaf`
- Windows: `%USERPROFILE%\\.loaf`

This includes state and custom tools. `/forgeteverything` clears core state/auth config.

## 8) Next docs

- [skills.md](skills.md)
- [tools.md](tools.md)
- [rpc.md](rpc.md)
