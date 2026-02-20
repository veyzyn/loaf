import fs from "node:fs";
import path from "node:path";
import { getCustomToolsDirectory, isValidCustomToolName, loadCustomToolFile } from "../custom.js";
import { type ToolRegistry } from "../registry.js";
import type { JsonValue, ToolDefinition, ToolInput, ToolResult } from "../types.js";

type CreatePersistentToolInput = ToolInput & {
  name?: JsonValue;
  description?: JsonValue;
  args_schema?: JsonValue;
  handler_code?: JsonValue;
  filename?: JsonValue;
  overwrite?: JsonValue;
};

const VALID_FILE_NAME = /^[a-zA-Z0-9._-]+$/;

export function createPersistentToolTool(params: {
  registry: ToolRegistry;
  isBuiltinToolName: (name: string) => boolean;
}): ToolDefinition<CreatePersistentToolInput> {
  return {
    name: "create_persistent_tool",
    description:
      "create or update a persistent js tool in the loaf tools directory and autoload it immediately.",
    inputSchema: {
      type: "object",
      properties: {
        name: {
          type: "string",
          description: "tool name (pattern: [a-zA-Z0-9_.:-]+).",
        },
        description: {
          type: "string",
          description: "short human-readable tool description.",
        },
        args_schema: {
          type: "object",
          description:
            "json schema object for tool args (type/object/properties/required/additionalProperties).",
        },
        handler_code: {
          type: "string",
          description:
            "javascript code for the body of async run(input, context) { ... }. return a json-serializable value.",
        },
        filename: {
          type: "string",
          description:
            "optional target filename inside tools dir (e.g. my_tool.mjs). defaults to a name-based .mjs file.",
        },
        overwrite: {
          type: "boolean",
          description: "when true, allows replacing an existing custom tool file and registration.",
        },
      },
      required: ["name", "description", "handler_code"],
      additionalProperties: false,
    },
    run: async (input) => {
      const name = asNonEmptyString(input.name);
      if (!name || !isValidCustomToolName(name)) {
        return invalidInput(
          "create_persistent_tool requires a valid `name` matching [a-zA-Z0-9_.:-]+.",
        );
      }

      const description = asNonEmptyString(input.description);
      if (!description) {
        return invalidInput("create_persistent_tool requires a non-empty `description`.");
      }

      const handlerCode = asNonEmptyString(input.handler_code);
      if (!handlerCode) {
        return invalidInput("create_persistent_tool requires non-empty `handler_code`.");
      }

      const argsSchema = normalizeSchema(input.args_schema);
      if (input.args_schema !== undefined && !argsSchema) {
        return invalidInput(
          "`args_schema` must be an object schema with `type: \"object\"` and `properties`.",
        );
      }

      const overwrite = input.overwrite === true;
      const targetDir = getCustomToolsDirectory();
      const fileName = resolveTargetFileName(input.filename, name);
      if (!fileName) {
        return invalidInput("invalid `filename` (use letters, numbers, ., _, - and .js/.mjs/.cjs).");
      }

      const targetPath = path.join(targetDir, fileName);
      const existedOnDisk = fs.existsSync(targetPath);
      if (existedOnDisk && !overwrite) {
        return invalidInput(`tool file already exists: ${targetPath}. set overwrite=true to replace.`);
      }

      const alreadyRegistered = params.registry.has(name);
      if (alreadyRegistered && params.isBuiltinToolName(name)) {
        return invalidInput(`cannot overwrite built-in tool: ${name}`);
      }
      if (alreadyRegistered && !overwrite) {
        return invalidInput(`tool already registered: ${name}. set overwrite=true to replace.`);
      }

      try {
        fs.mkdirSync(targetDir, { recursive: true });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return errorResult(`failed to create tools directory: ${message}`, {
          tool_name: name,
          path: targetPath,
        });
      }

      const sourceCode = buildPersistentToolSource({
        name,
        description,
        argsSchema,
        handlerCode,
      });

      try {
        fs.writeFileSync(targetPath, sourceCode, "utf8");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return errorResult(`failed writing tool file: ${message}`, {
          tool_name: name,
          path: targetPath,
        });
      }

      try {
        const parsed = await loadCustomToolFile(targetPath);
        if (!parsed) {
          return errorResult("written file does not export a valid tool object.", {
            tool_name: name,
            path: targetPath,
          });
        }

        if (parsed.name !== name) {
          return errorResult(`tool export name mismatch: expected ${name}, got ${parsed.name}`, {
            tool_name: name,
            exported_name: parsed.name,
            path: targetPath,
          });
        }

        if (alreadyRegistered) {
          params.registry.unregister(name);
        }
        params.registry.register(parsed.tool);

        return okResult(
          alreadyRegistered
            ? `persistent tool updated and reloaded: ${name}`
            : `persistent tool created and loaded: ${name}`,
          {
            tool_name: name,
            path: targetPath,
            overwritten: alreadyRegistered || existedOnDisk,
          },
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return errorResult(`tool file written, but autoload failed: ${message}`, {
          tool_name: name,
          path: targetPath,
        });
      }
    },
  };
}

function okResult(message: string, data: Record<string, JsonValue>): ToolResult {
  return {
    ok: true,
    output: {
      status: "ok",
      message,
      ...data,
    },
  };
}

function errorResult(message: string, data: Record<string, JsonValue>): ToolResult {
  return {
    ok: false,
    output: {
      status: "error",
      message,
      ...data,
    },
    error: message,
  };
}

function buildPersistentToolSource(params: {
  name: string;
  description: string;
  argsSchema?: Record<string, unknown>;
  handlerCode: string;
}): string {
  const schema =
    params.argsSchema ??
    ({
      type: "object",
      properties: {},
      additionalProperties: true,
    } satisfies Record<string, unknown>);

  const handlerLines = params.handlerCode
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => `    ${line}`)
    .join("\n");

  return [
    "export default {",
    `  name: ${JSON.stringify(params.name)},`,
    `  description: ${JSON.stringify(params.description)},`,
    `  args: ${JSON.stringify(schema, null, 2).replace(/\n/g, "\n  ")},`,
    "  async run(input, context) {",
    handlerLines || "    return null;",
    "  },",
    "};",
    "",
  ].join("\n");
}

function normalizeSchema(value: JsonValue | undefined): Record<string, unknown> | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!isRecord(value)) {
    return undefined;
  }
  if (value.type !== "object") {
    return undefined;
  }
  if (!isRecord(value.properties)) {
    return undefined;
  }

  try {
    return JSON.parse(JSON.stringify(value)) as Record<string, unknown>;
  } catch {
    return undefined;
  }
}

function resolveTargetFileName(rawValue: JsonValue | undefined, fallbackName: string): string | null {
  const fallback = `${fallbackName.replace(/[^a-zA-Z0-9._-]+/g, "_")}.mjs`;
  if (typeof rawValue !== "string") {
    return fallback;
  }

  const trimmed = rawValue.trim();
  if (!trimmed) {
    return fallback;
  }

  const baseName = path.basename(trimmed);
  if (!VALID_FILE_NAME.test(baseName)) {
    return null;
  }
  const extension = path.extname(baseName).toLowerCase();
  if (extension !== ".js" && extension !== ".mjs" && extension !== ".cjs") {
    return null;
  }

  return baseName;
}

function invalidInput(message: string): ToolResult {
  return {
    ok: false,
    output: {
      status: "invalid_input",
      message,
    },
    error: message,
  };
}

function asNonEmptyString(value: JsonValue | undefined): string {
  if (typeof value !== "string") {
    return "";
  }
  return value.trim();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
