import OpenAI from "openai";
import { loafConfig, type ThinkingLevel } from "./config.js";
import { defaultToolRegistry, defaultToolRuntime } from "./tools/index.js";
import type { ChatMessage, DebugEvent, ModelResult, StreamChunk } from "./chat-types.js";

export type OpenRouterRequest = {
  apiKey: string;
  model: string;
  messages: ChatMessage[];
  thinkingLevel: ThinkingLevel;
  includeThoughts: boolean;
  forcedProvider: string | null;
  systemInstruction?: string;
  signal?: AbortSignal;
  drainSteeringMessages?: () => ChatMessage[];
};

export type OpenRouterModelCandidate = {
  id: string;
  label: string;
  description: string;
  supportedParameters: string[];
  providerTags: string[];
};

const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";
const OPENROUTER_HTTP_REFERER = "https://github.com/crabmiau/loaf";
const OPENROUTER_X_TITLE = "loaf";
const MAX_429_RETRY_ATTEMPTS = 8;
const RETRY_BASE_DELAY_MS = 1_250;
const RETRY_MAX_DELAY_MS = 20_000;
const SAFE_PROVIDER_TOOL_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

export async function listOpenRouterModels(apiKey: string): Promise<OpenRouterModelCandidate[]> {
  const token = apiKey.trim();
  if (!token) {
    throw new Error("Missing OpenRouter API key.");
  }

  const [modelsResult, endpointsResult] = await Promise.all([
    fetch(`${OPENROUTER_BASE_URL}/models`, {
      method: "GET",
      headers: buildOpenRouterHeaders(token),
    }),
    fetch(`${OPENROUTER_BASE_URL}/models/endpoints`, {
      method: "GET",
      headers: buildOpenRouterHeaders(token),
    }),
  ]);

  if (!modelsResult.ok) {
    const bodyText = await modelsResult.text().catch(() => "");
    throw new Error(`OpenRouter models fetch failed (${modelsResult.status}): ${summarizeHttpError(bodyText)}`);
  }

  const modelsPayload = (await modelsResult.json()) as unknown;
  const providerTagsByModel = endpointsResult.ok
    ? extractProviderTagsByModel((await endpointsResult.json()) as unknown)
    : new Map<string, string[]>();

  return parseOpenRouterModels(modelsPayload, providerTagsByModel);
}

export async function listOpenRouterProvidersForModel(
  apiKey: string,
  modelId: string,
): Promise<string[]> {
  const token = apiKey.trim();
  if (!token) {
    throw new Error("Missing OpenRouter API key.");
  }

  const parsed = parseOpenRouterModelId(modelId);
  if (!parsed) {
    return [];
  }

  const response = await fetch(
    `${OPENROUTER_BASE_URL}/models/${encodeURIComponent(parsed.author)}/${encodeURIComponent(parsed.slug)}/endpoints`,
    {
      method: "GET",
      headers: buildOpenRouterHeaders(token),
    },
  );

  if (!response.ok) {
    const bodyText = await response.text().catch(() => "");
    throw new Error(`OpenRouter model endpoints fetch failed (${response.status}): ${summarizeHttpError(bodyText)}`);
  }

  const payload = (await response.json()) as unknown;
  const data = (payload as { data?: unknown })?.data;
  const row = typeof data === "object" && data !== null ? (data as Record<string, unknown>) : {};
  const endpoints = Array.isArray(row.endpoints) ? row.endpoints : [];
  const providers = extractProvidersFromEndpoints(endpoints);
  if (providers.length > 0) {
    return providers;
  }

  // Fallback to model-owner slug if endpoint payload has no explicit provider info.
  const ownerSlug = normalizeProviderSlug(parsed.author);
  return ownerSlug ? [ownerSlug] : [];
}

export async function runOpenRouterInferenceStream(
  request: OpenRouterRequest,
  onChunk?: (chunk: StreamChunk) => void,
  onDebug?: (event: DebugEvent) => void,
): Promise<ModelResult> {
  const apiKey = request.apiKey.trim();
  if (!apiKey) {
    throw new Error("Missing OpenRouter API key. Run /auth and select openrouter api key.");
  }

  const client = new OpenAI({
    apiKey,
    baseURL: OPENROUTER_BASE_URL,
    defaultHeaders: {
      "HTTP-Referer": OPENROUTER_HTTP_REFERER,
      "X-Title": OPENROUTER_X_TITLE,
    },
  });

  const toolDeclarations = buildToolDeclarations();
  const startedAt = Date.now();
  let toolRound = 0;
  const systemInstruction = request.systemInstruction?.trim() || loafConfig.systemInstruction;

  const conversation: Array<Record<string, unknown>> = [
    {
      role: "system",
      content: systemInstruction,
    },
    ...request.messages.map((message) => ({
      role: message.role,
      content: message.text,
    })),
  ];

  while (true) {
    assertNotAborted(request.signal);

    const steeringMessages = request.drainSteeringMessages?.() ?? [];
    if (steeringMessages.length > 0) {
      conversation.push(
        ...steeringMessages.map((message) => ({
          role: message.role,
          content: message.text,
        })),
      );
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

    const requestPayload: Record<string, unknown> = {
      model: request.model,
      messages: conversation,
      tools: toolDeclarations.declarations,
      tool_choice: "auto",
      parallel_tool_calls: false,
      stream: false,
    };

    const reasoning = buildOpenRouterReasoning(request.thinkingLevel, request.includeThoughts);
    if (reasoning) {
      requestPayload.reasoning = reasoning;
      requestPayload.include_reasoning = request.includeThoughts;
    } else {
      requestPayload.include_reasoning = false;
    }

    const forcedProvider = normalizeForcedProvider(request.forcedProvider);
    if (forcedProvider) {
      requestPayload.provider = {
        order: [forcedProvider],
        allow_fallbacks: false,
      };
    }

    onDebug?.({
      stage: "request",
      data: {
        toolRound,
        payload: requestPayload,
      },
    });

    const response = await createCompletionWithRetry(client, requestPayload, toolRound, onDebug, request.signal);
    onDebug?.({
      stage: "response_raw",
      data: {
        toolRound,
        response,
      },
    });

    const choice = (response as { choices?: Array<Record<string, unknown>> }).choices?.[0];
    const message = (choice as { message?: Record<string, unknown> } | undefined)?.message ?? {};
    const thought = readReasoningFromMessage(message);
    if (thought) {
      onChunk?.({
        thoughts: [thought],
        answerText: "",
      });
    }

    const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    if (toolCalls.length > 0) {
      onDebug?.({
        stage: "tool_calls",
        data: {
          toolRound,
          functionCalls: toolCalls,
        },
      });

      const executed: Array<{
        name: string;
        ok: boolean;
        input?: Record<string, unknown>;
        result: unknown;
        error?: string;
      }> = [];

      conversation.push({
        role: "assistant",
        content: normalizeContentString(message.content),
        tool_calls: toolCalls,
      });

      for (let i = 0; i < toolCalls.length; i += 1) {
        assertNotAborted(request.signal);
        const call = (toolCalls[i] ?? {}) as Record<string, unknown>;
        const functionRecord =
          typeof call.function === "object" && call.function !== null ? (call.function as Record<string, unknown>) : {};
        const providerToolName = readTrimmedString(functionRecord.name);
        const runtimeToolName =
          toolDeclarations.providerToRuntimeName.get(providerToolName) ?? providerToolName;
        const callId = readTrimmedString(call.id) || `${providerToolName || "tool"}-${toolRound}-${i}`;
        const argumentsRaw = readTrimmedString(functionRecord.arguments) || "{}";
        const callArgs = safeParseObject(argumentsRaw);

        const toolResult = await defaultToolRuntime.execute(
          {
            id: callId,
            name: runtimeToolName,
            input: callArgs as never,
          },
          {
            now: new Date(),
            signal: request.signal,
          },
        );

        executed.push({
          name: runtimeToolName,
          ok: toolResult.ok,
          input: callArgs,
          result: toolResult.output,
          error: toolResult.error,
        });

        const responsePayload = toolResult.ok
          ? { output: toolResult.output }
          : { error: toolResult.error ?? "tool execution failed", output: toolResult.output };

        conversation.push({
          role: "tool",
          tool_call_id: callId,
          content: JSON.stringify(responsePayload),
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

    const answer = normalizeContentString(message.content).trim() || "(No response text returned)";
    onChunk?.({
      thoughts: [],
      answerText: answer,
    });

    onDebug?.({
      stage: "response_final",
      data: {
        toolRound,
        thoughtCount: thought ? 1 : 0,
        answerLength: answer.length,
        durationMs: Date.now() - startedAt,
        answerPreview: answer.slice(0, 400),
      },
    });

    return {
      thoughts: thought ? [thought] : [],
      answer,
    };
  }
}

function buildOpenRouterReasoning(
  thinkingLevel: ThinkingLevel,
  includeThoughts: boolean,
): Record<string, unknown> | null {
  if (thinkingLevel === "OFF") {
    return null;
  }

  return {
    effort: mapThinkingToOpenRouterEffort(thinkingLevel),
    exclude: !includeThoughts,
  };
}

function normalizeForcedProvider(value: string | null): string | null {
  const trimmed = (value ?? "").trim().toLowerCase();
  if (!trimmed || trimmed === "any") {
    return null;
  }
  return trimmed;
}

function buildToolDeclarations(): {
  declarations: Array<Record<string, unknown>>;
  providerToRuntimeName: Map<string, string>;
} {
  const declarations: Array<Record<string, unknown>> = [];
  const providerToRuntimeName = new Map<string, string>();
  const usedProviderNames = new Set<string>();

  for (const tool of defaultToolRegistry.list()) {
    const schema = tool.inputSchema ?? {
      type: "object",
      properties: {},
      required: [],
    };

    const providerName = toProviderToolName(tool.name, usedProviderNames);
    providerToRuntimeName.set(providerName, tool.name);
    declarations.push({
      type: "function",
      function: {
        name: providerName,
        description: tool.description,
        parameters: {
          type: schema.type,
          properties: schema.properties,
          required: schema.required ?? [],
          additionalProperties: false,
        },
      },
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

async function createCompletionWithRetry(
  client: OpenAI,
  requestPayload: Record<string, unknown>,
  toolRound: number,
  onDebug?: (event: DebugEvent) => void,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  let attempt = 0;
  while (true) {
    assertNotAborted(signal);
    attempt += 1;
    try {
      return (await client.chat.completions.create(
        requestPayload as never,
        signal ? ({ signal } as never) : undefined,
      )) as unknown as Record<string, unknown>;
    } catch (error) {
      if (isAbortError(error)) {
        throw error;
      }
      const retryable = isRetryable429Error(error);
      if (!retryable || attempt >= MAX_429_RETRY_ATTEMPTS) {
        throw error;
      }

      const delayMs = computeRetryDelayMs(attempt);
      onDebug?.({
        stage: "retry_429",
        data: {
          toolRound,
          attempt,
          maxAttempts: MAX_429_RETRY_ATTEMPTS,
          delayMs,
          error: summarizeError(error),
        },
      });
      await sleep(delayMs, signal);
    }
  }
}

function parseOpenRouterModels(
  payload: unknown,
  providerTagsByModel: Map<string, string[]>,
): OpenRouterModelCandidate[] {
  const data = (payload as { data?: unknown })?.data;
  if (!Array.isArray(data)) {
    throw new Error("Unexpected OpenRouter models response format.");
  }

  const byId = new Map<string, OpenRouterModelCandidate>();
  for (const rawItem of data) {
    const item = typeof rawItem === "object" && rawItem !== null ? (rawItem as Record<string, unknown>) : {};
    const id = readTrimmedString(item.id);
    if (!id) {
      continue;
    }

    const supportedParameters = toStringArray(item.supported_parameters).map((param) => param.toLowerCase());
    const modelProviders = providerTagsByModel.get(id.toLowerCase()) ?? [];

    byId.set(id, {
      id,
      label: readTrimmedString(item.name) || id.replace(/[-_]+/g, " "),
      description: readTrimmedString(item.description) || "server-advertised openrouter model",
      supportedParameters,
      providerTags: modelProviders,
    });
  }

  return Array.from(byId.values()).sort((a, b) => a.label.localeCompare(b.label));
}

function extractProviderTagsByModel(payload: unknown): Map<string, string[]> {
  const data = (payload as { data?: unknown })?.data;
  const map = new Map<string, string[]>();
  if (!Array.isArray(data)) {
    return map;
  }

  for (const rawItem of data) {
    const item = typeof rawItem === "object" && rawItem !== null ? (rawItem as Record<string, unknown>) : {};
    const modelId = readTrimmedString(item.id);
    if (!modelId) {
      continue;
    }

    const endpoints = Array.isArray(item.endpoints) ? item.endpoints : [];
    const tags = extractProvidersFromEndpoints(endpoints);
    map.set(modelId.toLowerCase(), tags);
  }
  return map;
}

function extractProvidersFromEndpoints(endpoints: unknown[]): string[] {
  const tags: string[] = [];
  for (const rawEndpoint of endpoints) {
    const endpoint =
      typeof rawEndpoint === "object" && rawEndpoint !== null ? (rawEndpoint as Record<string, unknown>) : {};
    const candidates = [
      readTrimmedString(endpoint.tag),
      readTrimmedString(endpoint.provider),
      readTrimmedString(endpoint.provider_slug),
      normalizeProviderSlug(readTrimmedString(endpoint.provider_name)),
    ];
    for (const candidateRaw of candidates) {
      const candidate = normalizeProviderSlug(candidateRaw);
      if (!candidate || tags.includes(candidate)) {
        continue;
      }
      tags.push(candidate);
    }
  }
  return tags.sort((a, b) => a.localeCompare(b));
}

function parseOpenRouterModelId(modelId: string): { author: string; slug: string } | null {
  const trimmed = modelId.trim();
  if (!trimmed) {
    return null;
  }
  const segments = trimmed.split("/").map((item) => item.trim()).filter(Boolean);
  if (segments.length < 2) {
    return null;
  }
  const author = segments[0]!;
  const slug = segments.slice(1).join("/");
  if (!author || !slug) {
    return null;
  }
  return { author, slug };
}

function normalizeProviderSlug(value: string): string {
  const trimmed = value.trim().toLowerCase();
  if (!trimmed) {
    return "";
  }
  return trimmed
    .replace(/\s+/g, "-")
    .replace(/_/g, "-")
    .replace(/[^a-z0-9/-]+/g, "")
    .replace(/--+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function mapThinkingToOpenRouterEffort(thinkingLevel: ThinkingLevel): string {
  switch (thinkingLevel) {
    case "OFF":
      return "low";
    case "MINIMAL":
      return "low";
    case "LOW":
      return "low";
    case "MEDIUM":
      return "medium";
    case "HIGH":
      return "high";
    case "XHIGH":
      return "high";
    default:
      return "medium";
  }
}

function readReasoningFromMessage(message: Record<string, unknown>): string {
  const direct = readTrimmedString(message.reasoning);
  if (direct) {
    return direct;
  }

  const details = Array.isArray(message.reasoning_details) ? message.reasoning_details : [];
  const chunks: string[] = [];
  for (const rawDetail of details) {
    const detail = typeof rawDetail === "object" && rawDetail !== null ? (rawDetail as Record<string, unknown>) : {};
    const summary = readTrimmedString(detail.summary);
    if (summary) {
      chunks.push(summary);
    }
  }
  return chunks.join("\n").trim();
}

function normalizeContentString(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }

  const chunks: string[] = [];
  for (const part of content) {
    if (typeof part === "string") {
      chunks.push(part);
      continue;
    }
    const record = typeof part === "object" && part !== null ? (part as Record<string, unknown>) : {};
    const text = readTrimmedString(record.text);
    if (text) {
      chunks.push(text);
    }
  }
  return chunks.join("\n").trim();
}

function safeParseObject(raw: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // noop
  }
  return {};
}

function readTrimmedString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildOpenRouterHeaders(apiKey: string): Record<string, string> {
  return {
    Authorization: `Bearer ${apiKey}`,
    Accept: "application/json",
    "HTTP-Referer": OPENROUTER_HTTP_REFERER,
    "X-Title": OPENROUTER_X_TITLE,
  };
}

function summarizeHttpError(payload: string): string {
  const text = payload.trim();
  if (!text) {
    return "empty response body";
  }
  return text.length <= 180 ? text : `${text.slice(0, 177)}...`;
}

function isRetryable429Error(error: unknown): boolean {
  const err = error as { status?: unknown; message?: unknown };
  if (typeof err.status === "number" && err.status === 429) {
    return true;
  }

  const text = summarizeError(error).toLowerCase();
  if (!text) {
    return false;
  }

  return (
    text.includes("too many requests") ||
    text.includes("rate limit") ||
    text.includes("\"status\":429") ||
    text.includes("\"code\":429")
  );
}

function computeRetryDelayMs(attempt: number): number {
  const exponential = RETRY_BASE_DELAY_MS * Math.pow(2, Math.max(0, attempt - 1));
  const capped = Math.min(RETRY_MAX_DELAY_MS, exponential);
  const jitter = Math.floor(Math.random() * 500);
  return Math.max(250, Math.floor(capped + jitter));
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  assertNotAborted(signal);
  return new Promise((resolve, reject) => {
    const handle = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(handle);
      cleanup();
      reject(createAbortError());
    };
    const cleanup = () => {
      signal?.removeEventListener("abort", onAbort);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
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

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function summarizeError(error: unknown): string {
  if (error instanceof Error) {
    return error.message || error.name;
  }
  if (typeof error === "string") {
    return error;
  }
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}
