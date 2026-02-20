import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { getLoafDataDir } from "../persistence.js";
import type {
  JsonValue,
  ToolDefinition,
  ToolInput,
  ToolResult,
} from "./types.js";

const SUPPORTED_TOOL_FILE_EXTENSIONS = new Set([".js", ".mjs", ".cjs"]);
const TOOL_NAME_PATTERN = /^[a-zA-Z0-9_.:-]+$/;

type RawToolRunner = (input: ToolInput, context: unknown) => Promise<unknown> | unknown;

type ResolvedToolCandidate = {
  name: string;
  description: string;
  inputSchema?: ToolDefinition["inputSchema"];
  run: RawToolRunner;
};

export type CustomToolsDiscoveryResult = {
  searchedDirectories: string[];
  loaded: Array<{
    name: string;
    sourcePath: string;
    tool: ToolDefinition;
  }>;
  errors: string[];
};

export async function discoverCustomTools(): Promise<CustomToolsDiscoveryResult> {
  const searchedDirectories = resolveToolDirectories();
  const loaded: CustomToolsDiscoveryResult["loaded"] = [];
  const errors: string[] = [];

  for (const directory of searchedDirectories) {
    if (!fs.existsSync(directory)) {
      continue;
    }

    const files = listToolFiles(directory);
    for (const filePath of files) {
      try {
        const moduleUrl = `${pathToFileURL(filePath).href}?t=${Date.now()}`;
        const moduleExports = (await import(moduleUrl)) as Record<string, unknown>;
        const resolved = resolveToolCandidate(moduleExports, filePath);
        if (!resolved) {
          errors.push(`skipped ${filePath}: no valid tool export found`);
          continue;
        }
        if (!TOOL_NAME_PATTERN.test(resolved.name)) {
          errors.push(
            `skipped ${filePath}: invalid tool name "${resolved.name}" (allowed: letters, numbers, _ . : -)`,
          );
          continue;
        }
        loaded.push({
          name: resolved.name,
          sourcePath: filePath,
          tool: toToolDefinition(resolved),
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        errors.push(`failed loading ${filePath}: ${message}`);
      }
    }
  }

  return {
    searchedDirectories,
    loaded,
    errors,
  };
}

function resolveToolDirectories(): string[] {
  return [path.join(getLoafDataDir(), "tools")];
}

function listToolFiles(rootDir: string): string[] {
  const files: string[] = [];
  const stack = [rootDir];

  while (stack.length > 0) {
    const directory = stack.pop();
    if (!directory) {
      continue;
    }

    let entries: fs.Dirent[] = [];
    try {
      entries = fs.readdirSync(directory, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      if (entry.name === "node_modules") {
        continue;
      }
      const fullPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }
      if (!entry.isFile()) {
        continue;
      }
      const extension = path.extname(entry.name).toLowerCase();
      if (!SUPPORTED_TOOL_FILE_EXTENSIONS.has(extension)) {
        continue;
      }
      files.push(fullPath);
    }
  }

  files.sort((left, right) => left.localeCompare(right));
  return files;
}

function resolveToolCandidate(
  moduleExports: Record<string, unknown>,
  sourcePath: string,
): ResolvedToolCandidate | null {
  const roots: Record<string, unknown>[] = [moduleExports];
  if (isRecord(moduleExports.default)) {
    roots.push(moduleExports.default);
  }

  for (const root of roots) {
    const direct = parseToolObject(root, sourcePath);
    if (direct) {
      return direct;
    }
    if (isRecord(root.tool)) {
      const fromToolField = parseToolObject(root.tool, sourcePath);
      if (fromToolField) {
        return fromToolField;
      }
    }
    const fromMeta = parseMetaTool(root, sourcePath);
    if (fromMeta) {
      return fromMeta;
    }
  }

  return null;
}

function parseToolObject(
  value: unknown,
  sourcePath: string,
): ResolvedToolCandidate | null {
  if (!isRecord(value)) {
    return null;
  }

  const name = readTrimmedString(value.name);
  const run = value.run;
  if (!name || typeof run !== "function") {
    return null;
  }

  return {
    name,
    description: readTrimmedString(value.description) || `custom tool from ${path.basename(sourcePath)}`,
    inputSchema: normalizeInputSchema(value.inputSchema ?? value.args),
    run: run as RawToolRunner,
  };
}

function parseMetaTool(
  value: Record<string, unknown>,
  sourcePath: string,
): ResolvedToolCandidate | null {
  const meta = isRecord(value.meta) ? value.meta : null;
  if (!meta) {
    return null;
  }

  const name = readTrimmedString(meta.name ?? value.name);
  if (!name) {
    return null;
  }

  const runner = typeof value.run === "function" ? value.run : typeof value.default === "function" ? value.default : null;
  if (!runner) {
    return null;
  }

  return {
    name,
    description:
      readTrimmedString(meta.description ?? value.description) ||
      `custom tool from ${path.basename(sourcePath)}`,
    inputSchema: normalizeInputSchema(meta.args ?? meta.inputSchema ?? value.args ?? value.inputSchema),
    run: runner as RawToolRunner,
  };
}

function toToolDefinition(candidate: ResolvedToolCandidate): ToolDefinition {
  return {
    name: candidate.name,
    description: candidate.description,
    inputSchema: candidate.inputSchema,
    async run(input, context) {
      const raw = await candidate.run(input, context);
      return normalizeToolResult(raw);
    },
  };
}

function normalizeToolResult(raw: unknown): ToolResult<JsonValue> {
  if (isRecord(raw) && typeof raw.ok === "boolean" && "output" in raw) {
    return {
      ok: raw.ok,
      output: toJsonValue((raw as { output: unknown }).output),
      error: typeof raw.error === "string" ? raw.error : undefined,
    };
  }

  return {
    ok: true,
    output: toJsonValue(raw),
  };
}

function toJsonValue(value: unknown): JsonValue {
  if (value === undefined) {
    return null;
  }
  try {
    return JSON.parse(JSON.stringify(value)) as JsonValue;
  } catch {
    return String(value) as JsonValue;
  }
}

function normalizeInputSchema(value: unknown): ToolDefinition["inputSchema"] | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const properties = isRecord(value.properties) ? value.properties : null;
  if (!properties) {
    return undefined;
  }

  const required =
    Array.isArray(value.required) && value.required.every((item) => typeof item === "string")
      ? (value.required as string[])
      : undefined;
  const additionalProperties =
    typeof value.additionalProperties === "boolean" ? value.additionalProperties : undefined;

  return {
    type: "object",
    properties: properties as Record<string, Record<string, unknown>>,
    required,
    additionalProperties,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readTrimmedString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}
