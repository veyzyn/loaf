# loaf tools guide

Tools are executable capabilities the model can call during a run.

Use `/tools` inside `loaf` to view the currently registered tool list.

## Built-in tools

### Shell

- `bash`: run shell commands with session-persisted cwd/env state.

### Web search

- `search_web`: Exa-backed web search with URLs and highlights.
  - Requires Exa API key (set in onboarding or via RPC `auth.set.exa_key`).

### JavaScript execution

- `run_js`
- `install_js_packages`
- `run_js_module`
- `start_background_js`
- `read_background_js`
- `write_background_js`
- `stop_background_js`
- `list_background_js`

### Tool authoring helper

- `create_persistent_tool`: write + autoload a new custom JS tool file.

## Custom tools (user-authored)

Custom tools are loaded from:

- `<loaf data dir>/tools`
  - macOS/Linux: `~/.loaf/tools`
  - Windows: `%USERPROFILE%\\.loaf\\tools`

Supported file extensions:

- `.js`
- `.mjs`
- `.cjs`

Supported module export shapes:

1. Default tool object export.
2. Named `tool` export.
3. `meta` + `run` exports.

Minimal example:

```js
export default {
  name: "echo_text",
  description: "echo input text",
  args: {
    type: "object",
    properties: {
      text: { type: "string" }
    },
    required: ["text"],
    additionalProperties: false
  },
  async run(input) {
    return { echoed: String(input.text ?? "") };
  }
};
```

## Tool naming and schemas

- Tool name pattern: `[a-zA-Z0-9_.:-]+`
- Input schema should be JSON-schema-like object form (`type: "object"` + `properties`).

## Verification flow

1. Add a tool file to your custom tools directory.
2. Restart `loaf`.
3. Run `/tools` and confirm the tool appears.

## Best practices

- Keep tools small and focused.
- Validate inputs and return structured, deterministic outputs.
- Return clear errors instead of throwing when possible.
- Avoid destructive side effects unless explicitly requested by the prompt.

## Related docs

- [../CUSTOM_TOOLS.md](../CUSTOM_TOOLS.md)
- [../src/tools/README.md](../src/tools/README.md)
- [usage.md](usage.md)
