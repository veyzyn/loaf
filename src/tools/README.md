# tools infrastructure

this folder contains the tool system used by `loaf`.

## structure

- `types.ts`: shared tool contracts (`ToolDefinition`, `ToolCall`, `ToolResult`)
- `registry.ts`: registration + lookup (`ToolRegistry`)
- `runtime.ts`: execution wrapper (`ToolRuntime`)
- `builtin/javascript.ts`: js-first tools (`run_js`, `install_js_packages`, `run_js_module`, background session tools)
- `builtin/persistent-tool.ts`: built-in tool for writing+autoloading persistent tools
- `custom.ts`: loader for user-provided js tools
- `index.ts`: default registry/runtime exports + custom tool bootstrap

## custom js tools (recommended)

place `.js`, `.mjs`, or `.cjs` files in one of:

- `<loaf data dir>/tools`

supported module shapes:

1. export a tool object

```js
export default {
  name: "echo_text",
  description: "echo input text",
  args: {
    type: "object",
    properties: {
      text: { type: "string", description: "text to echo" },
    },
    required: ["text"],
    additionalProperties: false,
  },
  async run(input) {
    return { echoed: input.text ?? "" };
  },
};
```

2. export `meta` + `run`

```js
export const meta = {
  name: "sum_numbers",
  description: "sum two numbers",
  args: {
    type: "object",
    properties: {
      a: { type: "number" },
      b: { type: "number" },
    },
    required: ["a", "b"],
  },
};

export async function run(input) {
  return { total: Number(input.a) + Number(input.b) };
}
```

tool names must match `[a-zA-Z0-9_.:-]+`.

## quick check in cli

run the app and use `/tools` to print all currently registered tools.

## full guide

see `CUSTOM_TOOLS.md` at the project root for full authoring docs.
