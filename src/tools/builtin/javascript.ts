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

type StartBackgroundJsInput = ToolInput & {
  code?: JsonValue;
  args?: JsonValue;
  cwd?: JsonValue;
  keep_script?: JsonValue;
  runtime?: JsonValue;
  format?: JsonValue;
  session_name?: JsonValue;
  reuse_session?: JsonValue;
};

type ReadBackgroundJsInput = ToolInput & {
  session_id?: JsonValue;
  max_chars?: JsonValue;
  stream?: JsonValue;
  peek?: JsonValue;
};

type WriteBackgroundJsInput = ToolInput & {
  session_id?: JsonValue;
  input?: JsonValue;
  append_newline?: JsonValue;
};

type StopBackgroundJsInput = ToolInput & {
  session_id?: JsonValue;
  force?: JsonValue;
};

type ListBackgroundJsInput = ToolInput & {
  include_exited?: JsonValue;
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
type BackgroundStreamSelector = "both" | "stdout" | "stderr";

type BackgroundStreamState = {
  buffer: string;
  totalChars: number;
  droppedChars: number;
  readCursor: number;
};

type BackgroundSessionStatus = "running" | "exited";

type BackgroundSession = {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  cwd: string;
  runtime: ScriptRuntime;
  command: string;
  args: string[];
  pid: number | null;
  scriptPath: string;
  keepScript: boolean;
  status: BackgroundSessionStatus;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  stdout: BackgroundStreamState;
  stderr: BackgroundStreamState;
  child: ReturnType<typeof spawn>;
};

const MAX_CAPTURE_CHARS = 300_000;
const MAX_BACKGROUND_CAPTURE_CHARS = 300_000;
const DEFAULT_BACKGROUND_READ_CHARS = 8_000;
const MAX_BACKGROUND_READ_CHARS = 120_000;
const DEFAULT_TIMEOUT_SECONDS = 120;
const MAX_TIMEOUT_SECONDS = 60 * 20;
const COMMAND_DETECTION_TIMEOUT_MS = 8_000;
const commandAvailabilityCache = new Map<string, Promise<boolean>>();
const backgroundSessions = new Map<string, BackgroundSession>();
let backgroundCleanupInstalled = false;

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

const startBackgroundJsTool: ToolDefinition<StartBackgroundJsInput> = {
  name: "start_background_js",
  description:
    "start a javascript script in the background and return a session id that can be read/written later.",
  inputSchema: {
    type: "object",
    properties: {
      code: {
        type: "string",
        description: "full javascript code to execute in the background.",
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
      keep_script: {
        type: "boolean",
        description: "keep generated script file on disk after process exit. default false.",
      },
      runtime: {
        type: "string",
        description: "runtime: auto, node, or bun. default auto.",
      },
      format: {
        type: "string",
        description: "script format: module or commonjs. default module.",
      },
      session_name: {
        type: "string",
        description: "optional friendly label for this background session.",
      },
      reuse_session: {
        type: "boolean",
        description: "when true, reuse an existing running session with the same session_name and cwd. default true.",
      },
    },
    required: ["code"],
    additionalProperties: false,
  },
  run: async (input) => {
    const code = asNonEmptyString(input.code);
    if (!code) {
      return invalidInput("start_background_js requires a non-empty `code` string.");
    }

    const args = parseStringArray(input.args);
    const cwd = asNonEmptyString(input.cwd) || process.cwd();
    const keepScript = asBoolean(input.keep_script);
    const requestedRuntime = normalizeScriptRuntime(input.runtime);
    const format = normalizeScriptFormat(input.format);
    const sessionNameRaw = asNonEmptyString(input.session_name);
    const sessionName = sessionNameRaw || "background-js";
    const reuseSession = input.reuse_session === undefined ? true : asBoolean(input.reuse_session);

    if (reuseSession && sessionNameRaw) {
      const existing = findRunningBackgroundSession(sessionNameRaw, cwd);
      if (existing) {
        return {
          ok: true,
          output: {
            status: "reused",
            session_id: existing.id,
            session_name: existing.name,
            pid: existing.pid,
            runtime: existing.runtime,
            command: existing.command,
            args: existing.args,
            cwd: existing.cwd,
            created_at: existing.createdAt,
          },
        };
      }
    }

    const runtime = await resolveScriptRuntime(requestedRuntime);
    if (!runtime) {
      const requestedLabel = requestedRuntime === "auto" ? "auto" : requestedRuntime;
      return invalidInput(`start_background_js could not resolve runtime "${requestedLabel}".`);
    }

    const runDir = path.join(getLoafDataDir(), "js-runtime", "background");
    await fs.mkdir(runDir, { recursive: true });
    const scriptPath = path.join(runDir, createScriptFileName(format));
    await fs.writeFile(scriptPath, code, "utf8");
    ensureBackgroundCleanupHook();

    const command = runtime.command;
    const commandArgs = [...runtime.baseArgs, scriptPath, ...args];
    const child = spawn(command, commandArgs, {
      cwd,
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    const sessionId = createBackgroundSessionId();
    const now = new Date().toISOString();
    const session: BackgroundSession = {
      id: sessionId,
      name: sessionName,
      createdAt: now,
      updatedAt: now,
      cwd,
      runtime: runtime.name,
      command,
      args: commandArgs,
      pid: child.pid ?? null,
      scriptPath,
      keepScript,
      status: "running",
      exitCode: null,
      signal: null,
      stdout: createBackgroundStreamState(),
      stderr: createBackgroundStreamState(),
      child,
    };
    backgroundSessions.set(sessionId, session);

    child.stdout?.on("data", (chunk) => {
      appendToBackgroundStream(session.stdout, String(chunk));
      session.updatedAt = new Date().toISOString();
    });
    child.stderr?.on("data", (chunk) => {
      appendToBackgroundStream(session.stderr, String(chunk));
      session.updatedAt = new Date().toISOString();
    });
    child.on("error", (error) => {
      appendToBackgroundStream(session.stderr, `[spawn error] ${error.message}\n`);
      session.updatedAt = new Date().toISOString();
    });
    child.on("close", (exitCode, signal) => {
      session.status = "exited";
      session.exitCode = exitCode;
      session.signal = signal;
      session.updatedAt = new Date().toISOString();
      if (!session.keepScript) {
        void fs.unlink(session.scriptPath).catch(() => {
          // best effort cleanup
        });
      }
    });

    return {
      ok: true,
      output: {
        status: "started",
        session_id: session.id,
        session_name: session.name,
        pid: session.pid,
        runtime: session.runtime,
        command: session.command,
        args: session.args,
        cwd: session.cwd,
        created_at: session.createdAt,
      },
    };
  },
};

const readBackgroundJsTool: ToolDefinition<ReadBackgroundJsInput> = {
  name: "read_background_js",
  description:
    "read buffered stdout/stderr from a running or exited background js session.",
  inputSchema: {
    type: "object",
    properties: {
      session_id: {
        type: "string",
        description: "session id returned by start_background_js.",
      },
      max_chars: {
        type: "number",
        description: "max characters per stream to return (default 8000, max 120000).",
      },
      stream: {
        type: "string",
        description: "stream selector: both, stdout, or stderr. default both.",
      },
      peek: {
        type: "boolean",
        description: "when true, do not advance the internal read cursor.",
      },
    },
    required: ["session_id"],
    additionalProperties: false,
  },
  run: (input) => {
    const session = getBackgroundSession(input.session_id);
    if (!session) {
      return invalidInput("read_background_js requires a valid `session_id`.");
    }

    const maxChars = parseBackgroundReadChars(input.max_chars);
    const stream = normalizeBackgroundStreamSelector(input.stream);
    const peek = asBoolean(input.peek);
    if (!stream) {
      return invalidInput("`stream` must be one of: both, stdout, stderr.");
    }

    const stdoutRead = stream === "both" || stream === "stdout"
      ? readBackgroundStream(session.stdout, maxChars, peek)
      : createEmptyBackgroundRead(session.stdout.readCursor);
    const stderrRead = stream === "both" || stream === "stderr"
      ? readBackgroundStream(session.stderr, maxChars, peek)
      : createEmptyBackgroundRead(session.stderr.readCursor);

    return {
      ok: true,
      output: {
        status: "ok",
        session_id: session.id,
        session_name: session.name,
        running: session.status === "running",
        exit_code: session.exitCode,
        signal: session.signal ?? null,
        stdout: stdoutRead.text,
        stderr: stderrRead.text,
        stdout_cursor: stdoutRead.cursor,
        stderr_cursor: stderrRead.cursor,
        stdout_has_more: stdoutRead.hasMore,
        stderr_has_more: stderrRead.hasMore,
        stdout_dropped: stdoutRead.dropped,
        stderr_dropped: stderrRead.dropped,
      },
    };
  },
};

const writeBackgroundJsTool: ToolDefinition<WriteBackgroundJsInput> = {
  name: "write_background_js",
  description:
    "write input text to stdin of a running background js session.",
  inputSchema: {
    type: "object",
    properties: {
      session_id: {
        type: "string",
        description: "session id returned by start_background_js.",
      },
      input: {
        type: "string",
        description: "text to write to stdin.",
      },
      append_newline: {
        type: "boolean",
        description: "append newline to input before writing. default true.",
      },
    },
    required: ["session_id", "input"],
    additionalProperties: false,
  },
  run: async (input) => {
    const session = getBackgroundSession(input.session_id);
    if (!session) {
      return invalidInput("write_background_js requires a valid `session_id`.");
    }
    if (session.status !== "running") {
      return {
        ok: false,
        output: {
          status: "not_running",
          session_id: session.id,
          running: false,
          exit_code: session.exitCode,
          signal: session.signal ?? null,
          bytes_written: null,
        },
        error: "background session is not running",
      };
    }

    if (typeof input.input !== "string") {
      return invalidInput("write_background_js requires `input` as a string.");
    }

    const appendNewline = input.append_newline === undefined ? true : asBoolean(input.append_newline);
    const finalPayload = appendNewline ? `${input.input}\n` : input.input;

    if (!session.child.stdin || session.child.stdin.destroyed) {
      return {
        ok: false,
        output: {
          status: "stdin_unavailable",
          session_id: session.id,
          running: session.status === "running",
          exit_code: session.exitCode,
          signal: session.signal ?? null,
          bytes_written: null,
        },
        error: "background session stdin is unavailable",
      };
    }

    await writeToBackgroundStdin(session.child.stdin, finalPayload);
    session.updatedAt = new Date().toISOString();

    return {
      ok: true,
      output: {
        status: "ok",
        session_id: session.id,
        running: session.status === "running",
        exit_code: session.exitCode,
        signal: session.signal ?? null,
        bytes_written: Buffer.byteLength(finalPayload),
      },
    };
  },
};

const stopBackgroundJsTool: ToolDefinition<StopBackgroundJsInput> = {
  name: "stop_background_js",
  description:
    "stop a background js session.",
  inputSchema: {
    type: "object",
    properties: {
      session_id: {
        type: "string",
        description: "session id returned by start_background_js.",
      },
      force: {
        type: "boolean",
        description: "when true, send SIGKILL. default false (SIGTERM).",
      },
    },
    required: ["session_id"],
    additionalProperties: false,
  },
  run: async (input) => {
    const session = getBackgroundSession(input.session_id);
    if (!session) {
      return invalidInput("stop_background_js requires a valid `session_id`.");
    }

    if (session.status !== "running") {
      return {
        ok: true,
        output: {
          status: "already_stopped",
          session_id: session.id,
          running: false,
          exit_code: session.exitCode,
          signal: session.signal ?? null,
        },
      };
    }

    const force = asBoolean(input.force);
    const signal: NodeJS.Signals = force ? "SIGKILL" : "SIGTERM";
    session.child.kill(signal);
    await sleepMs(force ? 50 : 120);

    return {
      ok: true,
      output: {
        status: "stop_requested",
        session_id: session.id,
        signal,
        running: session.status === "running",
        exit_code: session.exitCode,
      },
    };
  },
};

const listBackgroundJsTool: ToolDefinition<ListBackgroundJsInput> = {
  name: "list_background_js",
  description:
    "list known background js sessions and their state.",
  inputSchema: {
    type: "object",
    properties: {
      include_exited: {
        type: "boolean",
        description: "include exited sessions. default false.",
      },
    },
    additionalProperties: false,
  },
  run: (input) => {
    const includeExited = asBoolean(input.include_exited);
    const sessions = Array.from(backgroundSessions.values())
      .filter((session) => includeExited || session.status === "running")
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
      .map((session) => ({
        session_id: session.id,
        session_name: session.name,
        pid: session.pid,
        status: session.status,
        running: session.status === "running",
        exit_code: session.exitCode,
        signal: session.signal ?? null,
        created_at: session.createdAt,
        updated_at: session.updatedAt,
        unread_stdout_chars: unreadBackgroundChars(session.stdout),
        unread_stderr_chars: unreadBackgroundChars(session.stderr),
      }));

    return {
      ok: true,
      output: {
        status: "ok",
        count: sessions.length,
        sessions,
      },
    };
  },
};

export const JAVASCRIPT_BUILTIN_TOOLS: ToolDefinition[] = [
  runJsTool,
  installJsPackagesTool,
  runJsModuleTool,
  startBackgroundJsTool,
  readBackgroundJsTool,
  writeBackgroundJsTool,
  stopBackgroundJsTool,
  listBackgroundJsTool,
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

function createBackgroundSessionId(): string {
  const suffix = Math.random().toString(16).slice(2, 10);
  return `bg-${Date.now()}-${suffix}`;
}

function findRunningBackgroundSession(name: string, cwd: string): BackgroundSession | null {
  for (const session of backgroundSessions.values()) {
    if (session.status !== "running") {
      continue;
    }
    if (session.name === name && session.cwd === cwd) {
      return session;
    }
  }
  return null;
}

function ensureBackgroundCleanupHook(): void {
  if (backgroundCleanupInstalled) {
    return;
  }
  backgroundCleanupInstalled = true;
  process.on("exit", () => {
    for (const session of backgroundSessions.values()) {
      if (session.status !== "running") {
        continue;
      }
      try {
        session.child.kill("SIGTERM");
      } catch {
        // best effort cleanup
      }
    }
  });
}

function createBackgroundStreamState(): BackgroundStreamState {
  return {
    buffer: "",
    totalChars: 0,
    droppedChars: 0,
    readCursor: 0,
  };
}

function appendToBackgroundStream(stream: BackgroundStreamState, chunk: string): void {
  if (!chunk) {
    return;
  }
  stream.totalChars += chunk.length;
  stream.buffer = `${stream.buffer}${chunk}`;
  if (stream.buffer.length > MAX_BACKGROUND_CAPTURE_CHARS) {
    const dropCount = stream.buffer.length - MAX_BACKGROUND_CAPTURE_CHARS;
    stream.buffer = stream.buffer.slice(dropCount);
    stream.droppedChars += dropCount;
  }
}

function normalizeBackgroundStreamSelector(value: JsonValue | undefined): BackgroundStreamSelector | null {
  if (value === undefined) {
    return "both";
  }
  if (typeof value !== "string") {
    return null;
  }

  const normalized = value.trim().toLowerCase();
  if (normalized === "both" || normalized === "stdout" || normalized === "stderr") {
    return normalized;
  }
  return null;
}

function parseBackgroundReadChars(value: JsonValue | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return DEFAULT_BACKGROUND_READ_CHARS;
  }
  return Math.max(1, Math.min(MAX_BACKGROUND_READ_CHARS, Math.floor(value)));
}

function getBackgroundSession(value: JsonValue | undefined): BackgroundSession | null {
  const sessionId = asNonEmptyString(value);
  if (!sessionId) {
    return null;
  }
  return backgroundSessions.get(sessionId) ?? null;
}

function unreadBackgroundChars(stream: BackgroundStreamState): number {
  const cursor = Math.max(stream.readCursor, stream.droppedChars);
  return Math.max(0, stream.totalChars - cursor);
}

function createEmptyBackgroundRead(cursor: number): {
  text: string;
  cursor: number;
  hasMore: boolean;
  dropped: boolean;
} {
  return {
    text: "",
    cursor,
    hasMore: false,
    dropped: false,
  };
}

function readBackgroundStream(
  stream: BackgroundStreamState,
  maxChars: number,
  peek: boolean,
): {
  text: string;
  cursor: number;
  hasMore: boolean;
  dropped: boolean;
} {
  const dropped = stream.readCursor < stream.droppedChars;
  const startCursor = Math.max(stream.readCursor, stream.droppedChars);
  const availableChars = Math.max(0, stream.totalChars - startCursor);
  const readChars = Math.min(maxChars, availableChars);
  const startIndex = startCursor - stream.droppedChars;
  const text = readChars > 0 ? stream.buffer.slice(startIndex, startIndex + readChars) : "";
  const nextCursor = startCursor + text.length;
  const hasMore = stream.totalChars > nextCursor;

  if (!peek) {
    stream.readCursor = nextCursor;
  }

  return {
    text,
    cursor: nextCursor,
    hasMore,
    dropped,
  };
}

async function writeToBackgroundStdin(stdin: NodeJS.WritableStream, payload: string): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    stdin.write(payload, (error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
}

function sleepMs(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, Math.max(0, ms));
  });
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
