import fs from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";
import { getLoafDataDir } from "../../persistence.js";
import type { JsonValue, ToolDefinition, ToolInput, ToolResult } from "../types.js";

type RunJsInput = ToolInput & {
  code?: JsonValue;
  args?: JsonValue;
  cwd?: JsonValue;
  timeout_seconds?: JsonValue;
  keep_script?: JsonValue;
  runtime?: JsonValue;
  format?: JsonValue;
};

type InstallJsPackagesInput = ToolInput & {
  packages?: JsonValue;
  args?: JsonValue;
  cwd?: JsonValue;
  timeout_seconds?: JsonValue;
  dev?: JsonValue;
  package_manager?: JsonValue;
};

type RunJsModuleInput = ToolInput & {
  module?: JsonValue;
  args?: JsonValue;
  cwd?: JsonValue;
  timeout_seconds?: JsonValue;
  package_manager?: JsonValue;
};

type ProcessRunResult = {
  command: string;
  args: string[];
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
  durationMs: number;
  truncatedStdout: boolean;
  truncatedStderr: boolean;
  ok: boolean;
};

type PackageManager = "bun" | "pnpm" | "yarn" | "npm";
type ScriptRuntime = "bun" | "node";
type ScriptFormat = "module" | "commonjs";

const MAX_CAPTURE_CHARS = 300_000;
const DEFAULT_TIMEOUT_SECONDS = 120;
const MAX_TIMEOUT_SECONDS = 60 * 20;
const COMMAND_DETECTION_TIMEOUT_MS = 8_000;
const commandAvailabilityCache = new Map<string, Promise<boolean>>();

const runJsTool: ToolDefinition<RunJsInput> = {
  name: "run_js",
  description:
    "run arbitrary javascript in a local runtime (node or bun) and return stdout/stderr/exit details.",
  inputSchema: {
    type: "object",
    properties: {
      code: {
        type: "string",
        description: "the full javascript code to execute.",
      },
      args: {
        type: "array",
        description: "optional argv passed to the script.",
        items: { type: "string" },
      },
      cwd: {
        type: "string",
        description: "optional working directory for script execution.",
      },
      timeout_seconds: {
        type: "number",
        description: "optional timeout in seconds (default 120, max 1200).",
      },
      keep_script: {
        type: "boolean",
        description: "keep generated script file on disk for debugging. default false.",
      },
      runtime: {
        type: "string",
        description: "runtime: auto, node, or bun. default auto.",
      },
      format: {
        type: "string",
        description: "script format: module or commonjs. default module.",
      },
    },
    required: ["code"],
    additionalProperties: false,
  },
  run: async (input) => {
    const code = asNonEmptyString(input.code);
    if (!code) {
      return invalidInput("run_js requires a non-empty `code` string.");
    }

    const args = parseStringArray(input.args);
    const cwd = asNonEmptyString(input.cwd) || process.cwd();
    const timeoutMs = parseTimeoutMs(input.timeout_seconds);
    const keepScript = asBoolean(input.keep_script);
    const requestedRuntime = normalizeScriptRuntime(input.runtime);
    const format = normalizeScriptFormat(input.format);

    const runtime = await resolveScriptRuntime(requestedRuntime);
    if (!runtime) {
      const requestedLabel = requestedRuntime === "auto" ? "auto" : requestedRuntime;
      return invalidInput(`run_js could not resolve runtime "${requestedLabel}".`);
    }

    const runDir = path.join(getLoafDataDir(), "js-runtime", "runs");
    await fs.mkdir(runDir, { recursive: true });
    const scriptPath = path.join(runDir, createScriptFileName(format));
    await fs.writeFile(scriptPath, code, "utf8");

    let result: ProcessRunResult;
    try {
      result = await runCommand(runtime.command, [...runtime.baseArgs, scriptPath, ...args], {
        cwd,
        timeoutMs,
      });
    } finally {
      if (!keepScript) {
        void fs.unlink(scriptPath).catch(() => {
          // best effort cleanup
        });
      }
    }

    return {
      ok: result.ok,
      output: processOutputToJson(result, {
        mode: "run_js",
        cwd,
        scriptPath,
        runtime: runtime.name,
        format,
      }),
      error: result.ok ? undefined : summarizeProcessError(result),
    };
  },
};

const installJsPackagesTool: ToolDefinition<InstallJsPackagesInput> = {
  name: "install_js_packages",
  description:
    "install javascript packages using available managers (bun, pnpm, yarn, npm).",
  inputSchema: {
    type: "object",
    properties: {
      packages: {
        type: "array",
        description: "package names to install.",
        items: { type: "string" },
      },
      args: {
        type: "array",
        description: "extra package-manager args.",
        items: { type: "string" },
      },
      dev: {
        type: "boolean",
        description: "install as dev dependencies.",
      },
      package_manager: {
        type: "string",
        description: "package manager: auto, bun, pnpm, yarn, npm. default auto.",
      },
      cwd: {
        type: "string",
        description: "optional working directory.",
      },
      timeout_seconds: {
        type: "number",
        description: "optional timeout in seconds (default 120, max 1200).",
      },
    },
    required: ["packages"],
    additionalProperties: false,
  },
  run: async (input) => {
    const packages = parseStringArray(input.packages);
    if (packages.length === 0) {
      return invalidInput("install_js_packages requires at least one package name in `packages`.");
    }

    const extraArgs = parseStringArray(input.args);
    const cwd = asNonEmptyString(input.cwd) || process.cwd();
    const timeoutMs = parseTimeoutMs(input.timeout_seconds);
    const dev = asBoolean(input.dev);
    const requestedManager = normalizePackageManager(input.package_manager);
    const manager = await resolveInstallManager(requestedManager);
    if (!manager) {
      return invalidInput("install_js_packages could not resolve an installed package manager.");
    }

    const managerArgs = getInstallArgsForManager(manager, {
      dev,
      extraArgs,
      packages,
    });

    const result = await runCommand(manager, managerArgs, {
      cwd,
      timeoutMs,
    });

    return {
      ok: result.ok,
      output: processOutputToJson(result, {
        mode: "install_js_packages",
        cwd,
        package_manager: manager,
        packages,
      }),
      error: result.ok ? undefined : summarizeProcessError(result),
    };
  },
};

const runJsModuleTool: ToolDefinition<RunJsModuleInput> = {
  name: "run_js_module",
  description:
    "run a javascript module/binary with package-manager executors (bunx/pnpm dlx/yarn dlx/npx).",
  inputSchema: {
    type: "object",
    properties: {
      module: {
        type: "string",
        description: "module or package binary to run.",
      },
      args: {
        type: "array",
        description: "optional module args.",
        items: { type: "string" },
      },
      package_manager: {
        type: "string",
        description: "executor manager: auto, bun, pnpm, yarn, npm. default auto.",
      },
      cwd: {
        type: "string",
        description: "optional working directory.",
      },
      timeout_seconds: {
        type: "number",
        description: "optional timeout in seconds (default 120, max 1200).",
      },
    },
    required: ["module"],
    additionalProperties: false,
  },
  run: async (input) => {
    const moduleName = asNonEmptyString(input.module);
    if (!moduleName) {
      return invalidInput("run_js_module requires a non-empty `module` string.");
    }

    const args = parseStringArray(input.args);
    const cwd = asNonEmptyString(input.cwd) || process.cwd();
    const timeoutMs = parseTimeoutMs(input.timeout_seconds);
    const requestedManager = normalizePackageManager(input.package_manager);
    const resolved = await resolveModuleRunner(requestedManager);
    if (!resolved) {
      return invalidInput("run_js_module could not resolve an installed module runner.");
    }

    const result = await runCommand(resolved.command, [...resolved.baseArgs, moduleName, ...args], {
      cwd,
      timeoutMs,
    });

    return {
      ok: result.ok,
      output: processOutputToJson(result, {
        mode: "run_js_module",
        cwd,
        package_manager: resolved.manager,
        module: moduleName,
      }),
      error: result.ok ? undefined : summarizeProcessError(result),
    };
  },
};

export const JAVASCRIPT_BUILTIN_TOOLS: ToolDefinition[] = [
  runJsTool,
  installJsPackagesTool,
  runJsModuleTool,
];

function normalizeScriptRuntime(value: JsonValue | undefined): "auto" | ScriptRuntime {
  if (typeof value !== "string") {
    return "auto";
  }
  const normalized = value.trim().toLowerCase();
  if (normalized === "node" || normalized === "bun") {
    return normalized;
  }
  return "auto";
}

function normalizeScriptFormat(value: JsonValue | undefined): ScriptFormat {
  if (typeof value !== "string") {
    return "module";
  }
  const normalized = value.trim().toLowerCase();
  if (normalized === "commonjs" || normalized === "cjs") {
    return "commonjs";
  }
  return "module";
}

function normalizePackageManager(value: JsonValue | undefined): "auto" | PackageManager {
  if (typeof value !== "string") {
    return "auto";
  }
  const normalized = value.trim().toLowerCase();
  if (normalized === "bun" || normalized === "pnpm" || normalized === "yarn" || normalized === "npm") {
    return normalized;
  }
  return "auto";
}

async function resolveScriptRuntime(
  requested: "auto" | ScriptRuntime,
): Promise<{ name: ScriptRuntime; command: string; baseArgs: string[] } | null> {
  if (requested === "node") {
    return {
      name: "node",
      command: process.execPath,
      baseArgs: [],
    };
  }

  if (requested === "bun") {
    if (!(await hasCommand("bun"))) {
      return null;
    }
    return {
      name: "bun",
      command: "bun",
      baseArgs: [],
    };
  }

  if (await hasCommand("bun")) {
    return {
      name: "bun",
      command: "bun",
      baseArgs: [],
    };
  }

  return {
    name: "node",
    command: process.execPath,
    baseArgs: [],
  };
}

async function resolveInstallManager(
  requested: "auto" | PackageManager,
): Promise<PackageManager | null> {
  if (requested !== "auto") {
    return (await hasCommand(requested)) ? requested : null;
  }

  const order: PackageManager[] = ["bun", "pnpm", "yarn", "npm"];
  for (const candidate of order) {
    if (await hasCommand(candidate)) {
      return candidate;
    }
  }
  return null;
}

async function resolveModuleRunner(
  requested: "auto" | PackageManager,
): Promise<{ manager: PackageManager; command: string; baseArgs: string[] } | null> {
  if (requested === "bun") {
    if (await hasCommand("bun")) {
      return { manager: "bun", command: "bun", baseArgs: ["x"] };
    }
    return null;
  }
  if (requested === "pnpm") {
    if (await hasCommand("pnpm")) {
      return { manager: "pnpm", command: "pnpm", baseArgs: ["dlx"] };
    }
    return null;
  }
  if (requested === "yarn") {
    if (await hasCommand("yarn")) {
      return { manager: "yarn", command: "yarn", baseArgs: ["dlx"] };
    }
    return null;
  }
  if (requested === "npm") {
    if (await hasCommand("npx")) {
      return { manager: "npm", command: "npx", baseArgs: ["--yes"] };
    }
    return null;
  }

  if (await hasCommand("bun")) {
    return { manager: "bun", command: "bun", baseArgs: ["x"] };
  }
  if (await hasCommand("pnpm")) {
    return { manager: "pnpm", command: "pnpm", baseArgs: ["dlx"] };
  }
  if (await hasCommand("yarn")) {
    return { manager: "yarn", command: "yarn", baseArgs: ["dlx"] };
  }
  if (await hasCommand("npx")) {
    return { manager: "npm", command: "npx", baseArgs: ["--yes"] };
  }

  return null;
}

function getInstallArgsForManager(
  manager: PackageManager,
  params: {
    dev: boolean;
    extraArgs: string[];
    packages: string[];
  },
): string[] {
  if (manager === "bun") {
    return ["add", ...(params.dev ? ["--dev"] : []), ...params.extraArgs, ...params.packages];
  }
  if (manager === "pnpm") {
    return ["add", ...(params.dev ? ["--save-dev"] : []), ...params.extraArgs, ...params.packages];
  }
  if (manager === "yarn") {
    return ["add", ...(params.dev ? ["--dev"] : []), ...params.extraArgs, ...params.packages];
  }
  return ["install", ...(params.dev ? ["--save-dev"] : []), ...params.extraArgs, ...params.packages];
}

function parseTimeoutMs(value: JsonValue | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return DEFAULT_TIMEOUT_SECONDS * 1_000;
  }
  const seconds = Math.max(1, Math.min(MAX_TIMEOUT_SECONDS, Math.floor(value)));
  return seconds * 1_000;
}

function parseStringArray(value: JsonValue | undefined): string[] {
  if (Array.isArray(value)) {
    return value
      .filter((item): item is string => typeof item === "string")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return [];
    }
    if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
      try {
        const parsed = JSON.parse(trimmed) as unknown;
        if (Array.isArray(parsed)) {
          return parsed
            .filter((item): item is string => typeof item === "string")
            .map((item) => item.trim())
            .filter(Boolean);
        }
      } catch {
        // fall through
      }
    }
    return trimmed.split(/\s+/).filter(Boolean);
  }
  return [];
}

function createScriptFileName(format: ScriptFormat): string {
  const suffix = Math.random().toString(16).slice(2, 10);
  const extension = format === "commonjs" ? "cjs" : "mjs";
  return `run-${Date.now()}-${suffix}.${extension}`;
}

function processOutputToJson(
  result: ProcessRunResult,
  details: Record<string, JsonValue>,
): ToolResult["output"] {
  return {
    status: result.ok ? "ok" : "error",
    ...details,
    command: result.command,
    args: result.args,
    exit_code: result.exitCode,
    signal: result.signal ?? null,
    timed_out: result.timedOut,
    duration_ms: result.durationMs,
    stdout: result.stdout,
    stderr: result.stderr,
    truncated_stdout: result.truncatedStdout,
    truncated_stderr: result.truncatedStderr,
  };
}

function summarizeProcessError(result: ProcessRunResult): string {
  const codeLabel = result.exitCode === null ? "no exit code" : `exit code ${result.exitCode}`;
  if (result.timedOut) {
    return `process timed out (${codeLabel})`;
  }
  return `process failed (${codeLabel})`;
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

function asBoolean(value: JsonValue | undefined): boolean {
  return value === true;
}

async function hasCommand(command: string): Promise<boolean> {
  const cached = commandAvailabilityCache.get(command);
  if (cached) {
    return cached;
  }

  const probe = (async () => {
    const result = await runCommand(command, ["--version"], {
      timeoutMs: COMMAND_DETECTION_TIMEOUT_MS,
    });
    return result.ok;
  })();
  commandAvailabilityCache.set(command, probe);
  return probe;
}

async function runCommand(
  command: string,
  args: string[],
  options: {
    cwd?: string;
    env?: NodeJS.ProcessEnv;
    timeoutMs?: number;
  } = {},
): Promise<ProcessRunResult> {
  const cwd = options.cwd || process.cwd();
  const timeoutMs = typeof options.timeoutMs === "number" && options.timeoutMs > 0
    ? options.timeoutMs
    : DEFAULT_TIMEOUT_SECONDS * 1_000;
  const startedAt = Date.now();

  let stdout = "";
  let stderr = "";
  let truncatedStdout = false;
  let truncatedStderr = false;
  let timedOut = false;

  const child = spawn(command, args, {
    cwd,
    env: options.env ?? process.env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  child.stdout?.on("data", (chunk) => {
    const next = `${stdout}${String(chunk)}`;
    if (next.length > MAX_CAPTURE_CHARS) {
      stdout = next.slice(0, MAX_CAPTURE_CHARS);
      truncatedStdout = true;
    } else {
      stdout = next;
    }
  });

  child.stderr?.on("data", (chunk) => {
    const next = `${stderr}${String(chunk)}`;
    if (next.length > MAX_CAPTURE_CHARS) {
      stderr = next.slice(0, MAX_CAPTURE_CHARS);
      truncatedStderr = true;
    } else {
      stderr = next;
    }
  });

  const timeoutHandle = setTimeout(() => {
    timedOut = true;
    child.kill("SIGTERM");
    setTimeout(() => {
      if (!child.killed) {
        child.kill("SIGKILL");
      }
    }, 1_500).unref();
  }, timeoutMs);

  const result = await new Promise<{
    exitCode: number | null;
    signal: NodeJS.Signals | null;
  }>((resolve) => {
    child.on("close", (exitCode, signal) => {
      resolve({
        exitCode,
        signal,
      });
    });

    child.on("error", () => {
      resolve({
        exitCode: null,
        signal: null,
      });
    });
  });

  clearTimeout(timeoutHandle);

  const durationMs = Date.now() - startedAt;
  const ok = !timedOut && result.exitCode === 0;

  return {
    command,
    args,
    exitCode: result.exitCode,
    signal: result.signal,
    stdout,
    stderr,
    timedOut,
    durationMs,
    truncatedStdout,
    truncatedStderr,
    ok,
  };
}
