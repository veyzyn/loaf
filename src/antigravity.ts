import { randomUUID } from "node:crypto";
import { loafConfig, type ThinkingLevel } from "./config.js";
import type { ChatMessage, DebugEvent, ModelResult, StreamChunk, StreamSegment } from "./chat-types.js";
import { defaultToolRegistry, defaultToolRuntime } from "./tools/index.js";

const CLOUD_CODE_BASE_URL_STABLE = "https://daily-cloudcode-pa.googleapis.com";
const CLOUD_CODE_BASE_URL_GCP_TOS = "https://cloudcode-pa.googleapis.com";
const ANTIGRAVITY_IDE_VERSION = "1.107.0";
const CLOUD_CODE_TIMEOUT_MS = 60 * 1000;
const CLOUD_CODE_ERROR_SNIPPET_CHARS = 280;
const SAFE_PROVIDER_TOOL_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

type CloudCodeMetadata = {
  ideType: "ANTIGRAVITY";
  ideVersion: string;
  pluginVersion: string;
  platform:
    | "DARWIN_AMD64"
    | "DARWIN_ARM64"
    | "LINUX_AMD64"
    | "LINUX_ARM64"
    | "WINDOWS_AMD64";
  updateChannel: "stable";
  pluginType: "GEMINI";
  ideName: "Antigravity";
};

type LoadCodeAssistResponse = {
  cloudaicompanionProject?: string;
  currentTier?: {
    usesGcpTos?: boolean;
  };
};

type AntigravityInferenceResponse = {
  response?: {
    candidates?: Array<{
      content?: {
        parts?: AntigravityContentPart[];
      };
    }>;
  };
  candidates?: Array<{
    content?: {
      parts?: AntigravityContentPart[];
    };
  }>;
};

type AntigravityContentPart = {
  text?: string;
  thought?: boolean;
  inlineData?: {
    mimeType?: string;
    data?: string;
  };
  functionCall?: {
    name?: string;
    args?: unknown;
    id?: string;
  };
};

export type AntigravityInferenceRequest = {
  accessToken: string;
  model: string;
  messages: ChatMessage[];
  thinkingLevel: ThinkingLevel;
  includeThoughts: boolean;
  systemInstruction?: string;
  signal?: AbortSignal;
  drainSteeringMessages?: () => ChatMessage[];
};

export async function runAntigravityInferenceStream(
  request: AntigravityInferenceRequest,
  onChunk?: (chunk: StreamChunk) => void,
  onDebug?: (event: DebugEvent) => void,
): Promise<ModelResult> {
  const accessToken = request.accessToken.trim();
  if (!accessToken) {
    throw new Error("Missing Antigravity OAuth token. Run /auth and select antigravity oauth.");
  }

  const startedAt = Date.now();
  const systemInstruction = request.systemInstruction?.trim() || loafConfig.systemInstruction;
  const contents = toGeminiContents(request.messages);

  const metadata = buildCloudCodeMetadata();
  const loadCodeAssist = await callCloudCode<LoadCodeAssistResponse>({
    accessToken,
    baseUrl: CLOUD_CODE_BASE_URL_STABLE,
    path: "v1internal:loadCodeAssist",
    body: { metadata },
    signal: request.signal,
  });

  const project = loadCodeAssist.cloudaicompanionProject?.trim() ?? "";
  if (!project) {
    throw new Error("antigravity project resolution failed. try /auth antigravity oauth again.");
  }

  const baseUrl = loadCodeAssist.currentTier?.usesGcpTos ? CLOUD_CODE_BASE_URL_GCP_TOS : CLOUD_CODE_BASE_URL_STABLE;
  let toolRound = 0;

  while (true) {
    assertNotAborted(request.signal);
    const steeringMessages = request.drainSteeringMessages?.() ?? [];
    if (steeringMessages.length > 0) {
      contents.push(...toGeminiContents(steeringMessages));
      onDebug?.({
        stage: "steer_injected",
        data: {
          count: steeringMessages.length,
          messages: steeringMessages.map((message) => ({
            role: message.role,
            preview: message.text.slice(0, 160),
          })),
        },
      });
    }

    toolRound += 1;
    const toolDeclarations = buildToolDeclarations();
    const payload = buildInferenceRequestBody({
      model: request.model,
      project,
      contents,
      systemInstruction,
      toolDeclarations: toolDeclarations.declarations,
    });

    onDebug?.({
      stage: "request",
      data: {
        provider: "antigravity",
        toolRound,
        payload,
      },
    });

    const response = await callCloudCode<AntigravityInferenceResponse>({
      accessToken,
      baseUrl,
      path: "v1internal:generateContent",
      body: payload,
      signal: request.signal,
    });

    onDebug?.({
      stage: "response_raw",
      data: {
        provider: "antigravity",
        toolRound,
        response,
      },
    });

    const parts = extractResponseParts(response);
    const functionCalls = extractFunctionCalls(parts, toolRound);
    if (functionCalls.length > 0) {
      const parsedPreTool = extractInferenceTextFromParts(parts);
      const preToolThoughts = request.includeThoughts ? parsedPreTool.thoughts : [];
      const preToolSegments = request.includeThoughts
        ? parsedPreTool.segments
        : parsedPreTool.segments.filter((segment) => segment.kind === "answer");
      const preToolAnswer = parsedPreTool.answer.trim();
      if (preToolThoughts.length > 0 || preToolAnswer || preToolSegments.length > 0) {
        onChunk?.({
          thoughts: preToolThoughts,
          answerText: preToolAnswer,
          segments: preToolSegments,
        });
      }

      onDebug?.({
        stage: "tool_calls",
        data: {
          toolRound,
          functionCalls,
        },
      });

      contents.push({
        role: "model",
        parts: sanitizeModelPartsForHistory(parts),
      });

      const executed: Array<{
        name: string;
        ok: boolean;
        input?: Record<string, unknown>;
        result: unknown;
        error?: string;
      }> = [];
      const toolResponseParts: Array<Record<string, unknown>> = [];

      for (let index = 0; index < functionCalls.length; index += 1) {
        assertNotAborted(request.signal);
        const call = functionCalls[index];
        if (!call) {
          continue;
        }

        const runtimeToolName =
          toolDeclarations.providerToRuntimeName.get(call.providerToolName) ?? call.providerToolName;

        onDebug?.({
          stage: "tool_call_started",
          data: {
            toolRound,
            call: {
              name: runtimeToolName,
              input: call.args,
              providerToolName: call.providerToolName,
              callId: call.callId,
            },
          },
        });

        const toolResult = await defaultToolRuntime.execute(
          {
            id: call.callId,
            name: runtimeToolName,
            input: call.args as never,
          },
          {
            now: new Date(),
            signal: request.signal,
          },
        );

        executed.push({
          name: runtimeToolName,
          ok: toolResult.ok,
          input: call.args,
          result: toolResult.output,
          error: toolResult.error,
        });
        onDebug?.({
          stage: "tool_call_completed",
          data: {
            toolRound,
            executed: {
              name: runtimeToolName,
              ok: toolResult.ok,
              input: call.args,
              result: toolResult.output,
              error: toolResult.error,
            },
          },
        });

        const responsePayload = toolResult.ok
          ? { output: toolResult.output }
          : { error: toolResult.error ?? "tool execution failed", output: toolResult.output };

        toolResponseParts.push({
          functionResponse: {
            name: call.providerToolName,
            response: {
              result: JSON.stringify(responsePayload),
            },
            id: call.callId,
          },
        });
      }

      if (toolResponseParts.length > 0) {
        contents.push({
          role: "user",
          parts: toolResponseParts,
        });
      }

      onDebug?.({
        stage: "tool_results",
        data: {
          toolRound,
          executed,
        },
      });

      continue;
    }

    const parsed = extractInferenceTextFromParts(parts);
    const answer = parsed.answer.trim() || "(No response text returned)";
    const thoughts = request.includeThoughts ? parsed.thoughts : [];
    const segments = request.includeThoughts
      ? parsed.segments
      : parsed.segments.filter((segment) => segment.kind === "answer");

    onChunk?.({
      thoughts,
      answerText: answer,
      segments,
    });

    onDebug?.({
      stage: "response_final",
      data: {
        provider: "antigravity",
        toolRound,
        thoughtCount: thoughts.length,
        answerLength: answer.length,
        durationMs: Date.now() - startedAt,
        answerPreview: answer.slice(0, 400),
      },
    });

    return {
      thoughts,
      answer,
    };
  }
}

function buildInferenceRequestBody(input: {
  model: string;
  project: string;
  contents: Array<Record<string, unknown>>;
  systemInstruction: string;
  toolDeclarations: Array<Record<string, unknown>>;
}): Record<string, unknown> {
  const request: Record<string, unknown> = {
    model: input.model,
    contents: input.contents,
  };
  // Antigravity desktop follows server-side defaults for thinking behavior.
  // Do not force client-side thinking config/budget to avoid Claude 4.x budget/max-token 400s.

  if (input.toolDeclarations.length > 0) {
    request.tools = [
      {
        functionDeclarations: input.toolDeclarations,
      },
    ];
    request.toolConfig = {
      functionCallingConfig: {
        mode: "AUTO",
      },
    };
  }

  const trimmedSystemInstruction = input.systemInstruction.trim();
  if (trimmedSystemInstruction) {
    request.systemInstruction = {
      role: "user",
      parts: [{ text: trimmedSystemInstruction }],
    };
  }

  return {
    project: input.project,
    requestId: `agent-${randomUUID()}`,
    request,
    model: input.model,
    userAgent: "antigravity",
    requestType: "agent",
  };
}

function buildToolDeclarations(): {
  declarations: Array<Record<string, unknown>>;
  providerToRuntimeName: Map<string, string>;
} {
  const declarations: Array<Record<string, unknown>> = [];
  const providerToRuntimeName = new Map<string, string>();
  const usedProviderNames = new Set<string>();

  for (const tool of defaultToolRegistry.list()) {
    const providerName = toProviderToolName(tool.name, usedProviderNames);
    providerToRuntimeName.set(providerName, tool.name);

    declarations.push({
      name: providerName,
      description: tool.description,
      parameters: normalizeGeminiToolSchema(tool.inputSchema),
    });
  }

  return {
    declarations,
    providerToRuntimeName,
  };
}

function toProviderToolName(runtimeName: string, usedNames: Set<string>): string {
  const baseName = runtimeName.trim() || "tool";
  let candidate = SAFE_PROVIDER_TOOL_NAME_PATTERN.test(baseName)
    ? baseName
    : baseName
        .replace(/[^a-zA-Z0-9_-]+/g, "_")
        .replace(/_+/g, "_")
        .replace(/^_+|_+$/g, "");

  if (!candidate) {
    candidate = "tool";
  }
  if (!/^[a-zA-Z_]/.test(candidate)) {
    candidate = `tool_${candidate}`;
  }
  if (candidate.length > 64) {
    candidate = candidate.slice(0, 64).replace(/_+$/g, "");
    if (!candidate) {
      candidate = "tool";
    }
  }

  let uniqueName = candidate;
  let suffix = 2;
  while (usedNames.has(uniqueName)) {
    const suffixText = `_${suffix++}`;
    const maxBaseLength = Math.max(1, 64 - suffixText.length);
    uniqueName = `${candidate.slice(0, maxBaseLength)}${suffixText}`;
  }
  usedNames.add(uniqueName);
  return uniqueName;
}

function normalizeGeminiToolSchema(rawSchema: unknown): Record<string, unknown> {
  const base = isRecord(rawSchema)
    ? (safeClone(rawSchema) as Record<string, unknown>)
    : {
        type: "object",
        properties: {},
        required: [],
      };

  if (typeof base.type !== "string" || !base.type.trim()) {
    base.type = "object";
  }
  if (!isRecord(base.properties)) {
    base.properties = {};
  }
  if (!Array.isArray(base.required)) {
    base.required = [];
  }
  if (!Object.hasOwn(base, "additionalProperties")) {
    base.additionalProperties = false;
  }

  return base;
}

function toGeminiContents(messages: ChatMessage[]): Array<Record<string, unknown>> {
  const contents: Array<Record<string, unknown>> = [];

  for (const message of messages) {
    const role = message.role === "assistant" ? "model" : "user";
    const parts: Array<Record<string, unknown>> = [];

    const text = message.text.trim();
    if (text) {
      parts.push({ text });
    }

    for (const image of message.images ?? []) {
      const parsed = parseDataUrl(image.dataUrl);
      if (!parsed) {
        continue;
      }

      parts.push({
        inlineData: {
          mimeType: parsed.mimeType,
          data: parsed.base64,
        },
      });
    }

    if (parts.length === 0) {
      continue;
    }

    contents.push({ role, parts });
  }

  if (contents.length === 0) {
    return [{ role: "user", parts: [{ text: "" }] }];
  }

  return contents;
}

function parseDataUrl(dataUrl: string): { mimeType: string; base64: string } | null {
  const match = dataUrl.match(/^data:([^;,]+);base64,(.+)$/i);
  if (!match) {
    return null;
  }

  const mimeType = match[1]?.trim() || "image/png";
  const base64 = match[2]?.trim() || "";
  if (!base64) {
    return null;
  }

  return { mimeType, base64 };
}

function extractResponseParts(response: AntigravityInferenceResponse): AntigravityContentPart[] {
  const raw = response.response ?? response;
  const firstCandidate = raw.candidates?.[0];
  return Array.isArray(firstCandidate?.content?.parts) ? firstCandidate.content.parts : [];
}

function extractFunctionCalls(
  parts: AntigravityContentPart[],
  toolRound: number,
): Array<{
  providerToolName: string;
  callId: string;
  args: Record<string, unknown>;
}> {
  const calls: Array<{
    providerToolName: string;
    callId: string;
    args: Record<string, unknown>;
  }> = [];

  for (let index = 0; index < parts.length; index += 1) {
    const part = parts[index];
    const functionCall = isRecord(part?.functionCall) ? part.functionCall : null;
    if (!functionCall) {
      continue;
    }

    const providerToolName = typeof functionCall.name === "string" ? functionCall.name.trim() : "";
    if (!providerToolName) {
      continue;
    }
    const callIdRaw = typeof functionCall.id === "string" ? functionCall.id.trim() : "";
    calls.push({
      providerToolName,
      callId: callIdRaw || `${providerToolName || "tool"}-${toolRound}-${index}`,
      args: normalizeFunctionCallArgs(functionCall.args),
    });
  }

  return calls;
}

function normalizeFunctionCallArgs(rawArgs: unknown): Record<string, unknown> {
  if (isRecord(rawArgs)) {
    return rawArgs;
  }
  if (typeof rawArgs === "string") {
    return safeParseObject(rawArgs);
  }
  return {};
}

function sanitizeModelPartsForHistory(parts: AntigravityContentPart[]): Array<Record<string, unknown>> {
  const sanitized: Array<Record<string, unknown>> = [];
  for (const part of parts) {
    if (!isRecord(part)) {
      continue;
    }
    sanitized.push(safeClone(part) as Record<string, unknown>);
  }
  return sanitized;
}

function extractInferenceTextFromParts(parts: AntigravityContentPart[]): {
  thoughts: string[];
  answer: string;
  segments: StreamSegment[];
} {
  const thoughts: string[] = [];
  const answerParts: string[] = [];
  const segments: StreamSegment[] = [];

  for (const part of parts) {
    const text = (part.text ?? "").trim();
    if (text) {
      if (part.thought) {
        thoughts.push(text);
        segments.push({ kind: "thought", text });
      } else {
        answerParts.push(text);
        segments.push({ kind: "answer", text });
      }
      continue;
    }

    const imageData = part.inlineData?.data?.trim() ?? "";
    if (imageData) {
      const mimeType = part.inlineData?.mimeType?.trim() || "image/png";
      const markdown = `![image](data:${mimeType};base64,${imageData})`;
      answerParts.push(markdown);
      segments.push({ kind: "answer", text: markdown });
    }
  }

  return {
    thoughts,
    answer: answerParts.join("\n"),
    segments,
  };
}

function safeParseObject(raw: unknown): Record<string, unknown> {
  if (typeof raw !== "string" || !raw.trim()) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw) as unknown;
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function safeClone<T>(value: T): T {
  if (typeof globalThis.structuredClone === "function") {
    return globalThis.structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

function assertNotAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw createAbortError();
  }
}

function createAbortError(): Error {
  const error = new Error("Request interrupted by user.");
  error.name = "AbortError";
  return error;
}

function buildCloudCodeMetadata(): CloudCodeMetadata {
  return {
    ideType: "ANTIGRAVITY",
    ideVersion: ANTIGRAVITY_IDE_VERSION,
    pluginVersion: ANTIGRAVITY_IDE_VERSION,
    platform: resolveCloudCodePlatform(),
    updateChannel: "stable",
    pluginType: "GEMINI",
    ideName: "Antigravity",
  };
}

function resolveCloudCodePlatform(): CloudCodeMetadata["platform"] {
  if (process.platform === "darwin") {
    return process.arch === "arm64" ? "DARWIN_ARM64" : "DARWIN_AMD64";
  }
  if (process.platform === "linux") {
    return process.arch === "arm64" ? "LINUX_ARM64" : "LINUX_AMD64";
  }
  return "WINDOWS_AMD64";
}

async function callCloudCode<T>(request: {
  accessToken: string;
  baseUrl: string;
  path: string;
  body: unknown;
  signal?: AbortSignal;
}): Promise<T> {
  const timeoutController = new AbortController();
  let timedOut = false;
  const timeoutHandle = setTimeout(() => {
    timedOut = true;
    timeoutController.abort();
  }, CLOUD_CODE_TIMEOUT_MS);

  const signal = request.signal
    ? AbortSignal.any([request.signal, timeoutController.signal])
    : timeoutController.signal;

  try {
    const response = await fetch(`${request.baseUrl}/${request.path}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${request.accessToken}`,
        "Content-Type": "application/json",
        "User-Agent": buildCloudCodeUserAgent(),
        "x-client-name": "antigravity",
        "x-client-version": ANTIGRAVITY_IDE_VERSION,
        "x-goog-api-client": "gl-node/18.18.2 fire/0.8.6 grpc/1.10.x",
      },
      body: JSON.stringify(request.body),
      signal,
    });

    const text = await response.text();
    if (!response.ok) {
      throw new Error(formatCloudCodeErrorMessage(response.status, text));
    }

    if (!text.trim()) {
      return {} as T;
    }

    return JSON.parse(text) as T;
  } catch (error) {
    if (request.signal?.aborted) {
      throw error;
    }
    if (timedOut) {
      throw new Error("antigravity request timed out.");
    }
    throw error;
  } finally {
    clearTimeout(timeoutHandle);
  }
}

function formatCloudCodeErrorMessage(status: number, bodyText: string): string {
  const parsed = safeParseJson(bodyText);
  const messageFromJson = readCloudCodeErrorMessage(parsed);
  const compact = compactInline(messageFromJson || bodyText || `http ${status}`);
  const enableApiUrl = readCloudCodeEnableApiUrl(parsed) ?? extractGoogleApiEnableUrl(compact);

  if (status === 403 && isCloudCodeApiDisabled(compact)) {
    return enableApiUrl
      ? `cloud code api disabled (403). enable: ${enableApiUrl}`
      : "cloud code api disabled (403).";
  }

  return `[cloudcode ${status}] ${clipInline(compact, CLOUD_CODE_ERROR_SNIPPET_CHARS)}`;
}

function isCloudCodeApiDisabled(message: string): boolean {
  const normalized = message.toLowerCase();
  return (
    normalized.includes("cloud code private api") &&
    (normalized.includes("has not been used in project") || normalized.includes("is disabled"))
  );
}

function extractGoogleApiEnableUrl(message: string): string | null {
  const match = message.match(/https:\/\/console\.developers\.google\.com\/apis\/api\/[^\s"'`<>)]+/i);
  if (!match) {
    return null;
  }
  return match[0].replace(/[.,;:!?]+$/g, "");
}

function readCloudCodeEnableApiUrl(parsed: unknown): string | null {
  if (!isRecord(parsed)) {
    return null;
  }

  const errorRecord = isRecord(parsed.error) ? parsed.error : null;
  if (!errorRecord) {
    return null;
  }

  const details = Array.isArray(errorRecord.details) ? errorRecord.details : [];
  for (const detail of details) {
    if (!isRecord(detail)) {
      continue;
    }

    const metadata = isRecord(detail.metadata) ? detail.metadata : null;
    const activationUrl = typeof metadata?.activationUrl === "string" ? metadata.activationUrl.trim() : "";
    if (activationUrl) {
      return activationUrl;
    }

    const links = Array.isArray(detail.links) ? detail.links : [];
    for (const link of links) {
      if (!isRecord(link)) {
        continue;
      }
      const url = typeof link.url === "string" ? link.url.trim() : "";
      if (url) {
        return url;
      }
    }
  }

  return null;
}

function readCloudCodeErrorMessage(parsed: unknown): string {
  if (!isRecord(parsed)) {
    return "";
  }

  const errorRecord = isRecord(parsed.error) ? parsed.error : null;
  const message = typeof errorRecord?.message === "string" ? errorRecord.message.trim() : "";
  if (message) {
    return message;
  }

  return "";
}

function safeParseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function compactInline(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function clipInline(value: string, maxChars: number): string {
  if (value.length <= maxChars) {
    return value;
  }
  return `${value.slice(0, Math.max(0, maxChars - 3)).trimEnd()}...`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function buildCloudCodeUserAgent(): string {
  const platform = process.platform === "win32" ? "windows" : process.platform;
  const arch = process.arch === "x64" ? "amd64" : process.arch === "ia32" ? "386" : process.arch;
  return `antigravity/${ANTIGRAVITY_IDE_VERSION} ${platform}/${arch}`;
}
