# Custom JS Tools

This guide explains how to add your own JavaScript tools to Loaf.

## Overview

Loaf auto-loads custom tools at startup and registers them into the same tool registry used by built-in tools.

If a custom tool is valid, it appears in `/tools` and can be called by the model.

## Tool Folders

Loaf discovers tools only from:

- `<loaf data dir>/tools`

Data dir location:

- macos/linux: `~/.loaf`
- windows: `%USERPROFILE%\.loaf`

Supported file extensions:

- `.js`
- `.mjs`
- `.cjs`

## Tool Module Formats

Use one of these export formats.

### 1. Default Tool Object

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
  async run(input, context) {
    return { echoed: String(input.text ?? "") };
  },
};
```

### 2. Named `tool` Export

```js
export const tool = {
  name: "sum_numbers",
  description: "sum two numbers",
  inputSchema: {
    type: "object",
    properties: {
      a: { type: "number" },
      b: { type: "number" },
    },
    required: ["a", "b"],
  },
  run(input) {
    return { total: Number(input.a) + Number(input.b) };
  },
};
```

### 3. Meta + Run (Decorator-Like Style)

```js
export const meta = {
  name: "hello_user",
  description: "build a greeting",
  args: {
    type: "object",
    properties: {
      name: { type: "string" },
    },
    required: ["name"],
  },
};

export async function run(input) {
  return { greeting: `hello ${input.name}` };
}
```

Note:
- JS decorators are not required.
- If you want decorator-like ergonomics, wrap your own helper in the file, but export one of the supported shapes above.

## Required and Optional Fields

Required:

- `name` (string)
- `run` (function)

Optional:

- `description` (string)
- `args` or `inputSchema` (object schema)

Tool name pattern:

- `[a-zA-Z0-9_.:-]+`

## Schema Format

Loaf expects an object schema shape:

```js
{
  type: "object",
  properties: {
    fieldName: { type: "string", description: "..." }
  },
  required: ["fieldName"],
  additionalProperties: false
}
```

`args` and `inputSchema` are treated equivalently for custom tools.

## Runtime Contract

Your `run(input, context)` receives:

- `input`: parsed JSON args from the model
- `context`: runtime context (`now`, etc.)

Return options:

1. Plain JSON-like value:
- Treated as success (`ok: true`) automatically.

2. Explicit tool result:

```js
return {
  ok: false,
  output: { reason: "validation failed" },
  error: "name is required",
};
```

If `run` throws, Loaf captures it and reports tool failure.

## Example: File Writer Tool

```js
import fs from "node:fs";

export default {
  name: "write_note",
  description: "write a utf-8 note to disk",
  args: {
    type: "object",
    properties: {
      path: { type: "string" },
      text: { type: "string" },
    },
    required: ["path", "text"],
    additionalProperties: false,
  },
  run(input) {
    fs.writeFileSync(String(input.path), String(input.text), "utf8");
    return { ok: true, output: { written: true, path: String(input.path) } };
  },
};
```

## Loading and Errors

At startup, Loaf logs custom tool loading outcomes to stdout/stderr.

Common reasons a tool is skipped:

- No valid export shape found
- Invalid `name`
- Duplicate tool name (already registered)
- Syntax/import runtime error in the tool file

## Quick Verification

1. Place a tool file in `~/.loaf/tools` (or `%USERPROFILE%\.loaf\tools` on Windows).
2. Start Loaf.
3. Run `/tools`.
4. Confirm your tool appears in the list.

## Best Practices

- Keep tools small and single-purpose.
- Validate inputs in `run` and return clear error messages.
- Use stable tool names; avoid frequent renames.
- Prefer deterministic output objects over free-form strings.
