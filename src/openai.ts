import OpenAI from "openai";
import { loafConfig, type ThinkingLevel } from "./config.js";
import { defaultToolRegistry, defaultToolRuntime } from "./tools/index.js";
import type { ChatMessage, DebugEvent, ModelResult, StreamChunk } from "./chat-types.js";

export type OpenAIRequest = {
  accessToken: string;
  chatgptAccountId: string | null;
  model: string;
  messages: ChatMessage[];
  thinkingLevel: ThinkingLevel;
  includeThoughts: boolean;
  systemInstruction?: string;
  signal?: AbortSignal;
  drainSteeringMessages?: () => ChatMessage[];
};

const CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex";
const CHATGPT_BACKEND_API_BASE_URL = "https://chatgpt.com/backend-api";
const OPENAI_MODELS_CLIENT_VERSION = "0.99.0";
const OPENAI_MODELS_ORIGINATOR = "codex_cli_rs";
const MAX_429_RETRY_ATTEMPTS = 8;
const RETRY_BASE_DELAY_MS = 1_250;
const RETRY_MAX_DELAY_MS = 20_000;
const SAFE_PROVIDER_TOOL_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

type OpenAiResponse = {
  id?: string;
  status?: string;
  output_text?: string;
  error?: {
    message?: string;
  } | null;
  output?: Array<{
    type?: string;
    id?: string;
    status?: string;
    call_id?: string;
    name?: string;
    arguments?: string;
    content?: Array<{
      type?: string;
      text?: string;
    }>;
  }>;
};

export type OpenAiCatalogModel = {
  id: string;
  label: string;
  description: string;
  priority?: number;
  supportedInApi?: boolean;
  visibility?: string;
  defaultReasoningLevel?: string;
  supportedReasoningLevels?: OpenAiCatalogReasoningLevel[];
};

export type OpenAiCatalogReasoningLevel = {
  effort: string;
  description: string;
};

export type OpenAiUsageWindow = {
  usedPercent: number;
  remainingPercent: number;
  limitWindowSeconds: number | null;
  resetAtEpochSeconds: number | null;
};

export type OpenAiUsageSnapshot = {
  planType: string | null;
  primary: OpenAiUsageWindow | null;
  secondary: OpenAiUsageWindow | null;
};

export function createChatgptCodexClient(
  accessToken: string,
  chatgptAccountId: string | null,
): OpenAI {
  const token = accessToken.trim();
  if (!token) {
    throw new Error("Missing OpenAI ChatGPT access token.");
  }

  const defaultHeaders: Record<string, string> = {
    version: OPENAI_MODELS_CLIENT_VERSION,
    originator: OPENAI_MODELS_ORIGINATOR,
  };
  const accountId = chatgptAccountId?.trim();
  if (accountId) {
    defaultHeaders["ChatGPT-Account-ID"] = accountId;
  }

  return new OpenAI({
    apiKey: token,
    baseURL: CHATGPT_CODEX_BASE_URL,
    defaultHeaders,
  });
}

export async function listOpenAiCatalogModels(
  accessToken: string,
  chatgptAccountId: string | null,
): Promise<OpenAiCatalogModel[]> {
  const token = accessToken.trim();
  if (!token) {
    throw new Error("Missing OpenAI ChatGPT access token.");
  }

  const query = new URLSearchParams({
    client_version: OPENAI_MODELS_CLIENT_VERSION,
  });
  const response = await fetch(`${CHATGPT_CODEX_BASE_URL}/models?${query.toString()}`, {
    method: "GET",
    headers: buildOpenAiCatalogHeaders(token, chatgptAccountId),
  });

  if (!response.ok) {
    const bodyText = await response.text().catch(() => "");
    throw new Error(
      `OpenAI models fetch failed (${response.status}): ${summarizeHttpError(bodyText)}`,
    );
  }

  const json = (await response.json()) as unknown;
  return parseOpenAiCatalogModels(json);
}

export async function fetchOpenAiUsageSnapshot(
  accessToken: string,
  chatgptAccountId: string | null,
): Promise<OpenAiUsageSnapshot> {
  const token = accessToken.trim();
  if (!token) {
    throw new Error("Missing OpenAI ChatGPT access token.");
  }

  const response = await fetch(`${CHATGPT_BACKEND_API_BASE_URL}/wham/usage`, {
    method: "GET",
    headers: buildOpenAiCatalogHeaders(token, chatgptAccountId),
  });

  if (!response.ok) {
    const bodyText = await response.text().catch(() => "");
    throw new Error(
      `OpenAI usage fetch failed (${response.status}): ${summarizeHttpError(bodyText)}`,
    );
  }

  const json = (await response.json()) as unknown;
  return parseOpenAiUsageSnapshot(json);
}

export async function runOpenAiInferenceStream(
  request: OpenAIRequest,
  onChunk?: (chunk: StreamChunk) => void,
  onDebug?: (event: DebugEvent) => void,
): Promise<ModelResult> {
  const accessToken = request.accessToken.trim();
  if (!accessToken) {
    throw new Error("Missing OpenAI ChatGPT access token. Run /auth and select openai oauth.");
  }

  const client = createChatgptCodexClient(accessToken, request.chatgptAccountId);
  const systemInstruction = request.systemInstruction?.trim() || loafConfig.systemInstruction;

  let conversationInput = messagesToResponsesInput(request.messages);
  const startedAt = Date.now();
  let toolRound = 0;

  while (true) {
    assertNotAborted(request.signal);

    const steeringMessages = request.drainSteeringMessages?.() ?? [];
    if (steeringMessages.length > 0) {
      conversationInput = [...conversationInput, ...chatMessagesToResponsesInput(steeringMessages)];
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
    const requestPayload: Record<string, unknown> = {
      model: request.model,
      instructions: systemInstruction,
      input: conversationInput,
      tools: toolDeclarations.declarations,
      tool_choice: "auto",
      parallel_tool_calls: false,
      store: false,
      stream: true,
    };

    requestPayload.reasoning = {
      effort: mapThinkingToOpenAiEffort(request.thinkingLevel),
    };

    onDebug?.({
      stage: "request",
      data: {
        toolRound,
        payload: requestPayload,
      },
    });

    let streamedAnswerTextThisRound = "";
    const handleChunkThisRound = (chunk: StreamChunk) => {
      const answerDelta = extractAnswerDeltaFromChunk(chunk);
      if (answerDelta) {
        streamedAnswerTextThisRound += answerDelta;
      }
      onChunk?.(chunk);
    };

    const response = await createResponseWithRetry(
      client,
      requestPayload,
      toolRound,
      onDebug,
      handleChunkThisRound,
      request.signal,
    );

    onDebug?.({
      stage: "response_raw",
      data: {
        toolRound,
        response,
      },
    });

    const functionCalls = selectActionableFunctionCalls(response.output);
    if (functionCalls.length > 0) {
      const preToolAnswer = extractResponseText(response).trim();
      const missingPreToolDelta = computeUnstreamedAnswerDelta(preToolAnswer, streamedAnswerTextThisRound);
      if (missingPreToolDelta) {
        onChunk?.({
          thoughts: [],
          answerText: missingPreToolDelta,
          segments: [{ kind: "answer", text: missingPreToolDelta }],
        });
      }

      onDebug?.({
        stage: "tool_calls",
        data: {
          toolRound,
          functionCalls,
        },
      });

      const executed: Array<{
        name: string;
        ok: boolean;
        input?: Record<string, unknown>;
        result: unknown;
        error?: string;
      }> = [];

      const functionOutputs: Array<{
        type: "function_call_output";
        call_id: string;
        output: string;
      }> = [];
      const replayFunctionCalls: Array<{
        type: "function_call";
        call_id: string;
        name: string;
        arguments: string;
      }> = [];

      for (let i = 0; i < functionCalls.length; i += 1) {
        assertNotAborted(request.signal);
        const call = functionCalls[i] ?? {};
        const providerToolName = String(call.name ?? "").trim();
        const runtimeToolName =
          toolDeclarations.providerToRuntimeName.get(providerToolName) ?? providerToolName;
        const callId = String(call.call_id ?? `${providerToolName || "tool"}-${toolRound}-${i}`);
        const callArgumentsRaw = typeof call.arguments === "string" && call.arguments.trim() ? call.arguments : "{}";
        const callArgs = safeParseObject(callArgumentsRaw);

        onDebug?.({
          stage: "tool_call_started",
          data: {
            toolRound,
            call: {
              name: runtimeToolName,
              input: callArgs,
              providerToolName,
              callId,
            },
          },
        });

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
        onDebug?.({
          stage: "tool_call_completed",
          data: {
            toolRound,
            executed: {
              name: runtimeToolName,
              ok: toolResult.ok,
              input: callArgs,
              result: toolResult.output,
              error: toolResult.error,
            },
          },
        });

        const responsePayload = toolResult.ok
          ? { output: toolResult.output }
          : { error: toolResult.error ?? "tool execution failed", output: toolResult.output };

        functionOutputs.push({
          type: "function_call_output",
          call_id: callId,
          output: JSON.stringify(responsePayload),
        });
        replayFunctionCalls.push({
          type: "function_call",
          call_id: callId,
          name: providerToolName,
          arguments: callArgumentsRaw,
        });
      }

      onDebug?.({
        stage: "tool_results",
        data: {
          toolRound,
          executed,
        },
      });

      const followUpInput = buildFunctionCallFollowUpInput(replayFunctionCalls, functionOutputs);
      conversationInput = [...conversationInput, ...followUpInput];
      continue;
    }

    const answer = extractResponseText(response).trim();
    const responseStatus = typeof response.status === "string" ? response.status.trim().toLowerCase() : "";
    if (!answer && responseStatus && responseStatus !== "completed") {
      onDebug?.({
        stage: "response_continue",
        data: {
          toolRound,
          responseStatus,
          reason: "empty assistant text",
        },
      });
      conversationInput = [
        ...conversationInput,
        {
          type: "message",
          role: "user",
          content: "continue",
        },
      ];
      continue;
    }
    const finalAnswer = answer || "(No response text returned)";

    const missingFinalAnswerDelta = computeUnstreamedAnswerDelta(finalAnswer, streamedAnswerTextThisRound);
    if (onChunk && missingFinalAnswerDelta) {
      onChunk({
        thoughts: [],
        answerText: missingFinalAnswerDelta,
        segments: [{ kind: "answer", text: missingFinalAnswerDelta }],
      });
    }

    onDebug?.({
      stage: "response_final",
      data: {
        toolRound,
        thoughtCount: 0,
        answerLength: finalAnswer.length,
        durationMs: Date.now() - startedAt,
        answerPreview: finalAnswer.slice(0, 400),
      },
    });

    return {
      thoughts: [],
      answer: finalAnswer,
    };
  }
}

function buildFunctionCallFollowUpInput(
  functionCalls: Array<{
    type: "function_call";
    call_id: string;
    name: string;
    arguments: string;
  }>,
  functionOutputs: Array<{
    type: "function_call_output";
    call_id: string;
    output: string;
  }>,
): Array<Record<string, unknown>> {
  return [...functionCalls, ...functionOutputs];
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
      name: providerName,
      description: tool.description,
      parameters: {
        type: schema.type,
        properties: schema.properties,
        required: schema.required ?? [],
        additionalProperties: false,
      },
      strict: false,
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

function messagesToResponsesInput(messages: ChatMessage[]): Array<Record<string, unknown>> {
  if (messages.length === 0) {
    return [
      {
        type: "message",
        role: "user",
        content: "Hello.",
      },
    ];
  }

  return messages.map((message) => ({
    type: "message",
    role: message.role,
    content: toOpenAiMessageContent(message),
  }));
}

function chatMessagesToResponsesInput(messages: ChatMessage[]): Array<Record<string, unknown>> {
  return messages.map((message) => ({
    type: "message",
    role: message.role,
    content: toOpenAiMessageContent(message),
  }));
}

function toOpenAiMessageContent(message: ChatMessage): string | Array<Record<string, unknown>> {
  if (message.role !== "user" || !Array.isArray(message.images) || message.images.length === 0) {
    return message.text;
  }

  const parts: Array<Record<string, unknown>> = [];
  const text = message.text.trim();
  if (text) {
    parts.push({
      type: "input_text",
      text,
    });
  }

  for (const image of message.images) {
    if (!image.dataUrl || !image.dataUrl.startsWith("data:")) {
      continue;
    }
    parts.push({
      type: "input_image",
      image_url: image.dataUrl,
    });
  }

  if (parts.length === 0) {
    return message.text;
  }
  return parts;
}

function safeParseObject(raw: unknown): Record<string, unknown> {
  if (typeof raw !== "string" || !raw.trim()) {
    return {};
  }

  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return {};
  } catch {
    return {};
  }
}

function extractResponseText(response: {
  output_text?: string;
  output?: Array<{
    type?: string;
    content?: Array<{
      type?: string;
      text?: string | { value?: string };
      value?: string;
    }>;
  }>;
}): string {
  const fromOutputText = typeof response.output_text === "string" ? response.output_text.trim() : "";
  if (fromOutputText) {
    return fromOutputText;
  }

  const parts: string[] = [];
  for (const item of response.output ?? []) {
    if (item?.type !== "message" || !Array.isArray(item.content)) {
      continue;
    }

    for (const contentItem of item.content) {
      const contentType = typeof contentItem?.type === "string" ? contentItem.type : "";
      if (contentType !== "output_text" && contentType !== "text") {
        continue;
      }
      const textField = contentItem.text;
      const directText = typeof textField === "string" ? textField : "";
      const nestedValue =
        textField && typeof textField === "object" && typeof textField.value === "string"
          ? textField.value
          : "";
      const valueField = typeof contentItem.value === "string" ? contentItem.value : "";
      const text = directText || nestedValue || valueField;
      if (text.trim()) {
        parts.push(text.trim());
      }
    }
  }

  return parts.join("\n\n").trim();
}

function extractAnswerDeltaFromChunk(chunk: StreamChunk): string {
  const segments = Array.isArray(chunk.segments) ? chunk.segments : [];
  if (segments.length > 0) {
    const pieces = segments
      .filter((segment) => segment.kind === "answer" && Boolean(segment.text))
      .map((segment) => segment.text);
    if (pieces.length > 0) {
      return pieces.join("");
    }
  }
  return chunk.answerText || "";
}

function computeUnstreamedAnswerDelta(expectedText: string, streamedText: string): string {
  if (!expectedText) {
    return "";
  }
  if (!streamedText) {
    return expectedText;
  }
  if (expectedText === streamedText) {
    return "";
  }
  if (expectedText.startsWith(streamedText)) {
    return expectedText.slice(streamedText.length);
  }
  if (streamedText.includes(expectedText)) {
    return "";
  }
  return "";
}

function selectActionableFunctionCalls(
  output: OpenAiResponse["output"] | undefined,
): Array<NonNullable<OpenAiResponse["output"]>[number]> {
  const calls = (output ?? []).filter((item) => item?.type === "function_call");
  if (calls.length === 0) {
    return [];
  }

  const seenSignatures = new Set<string>();
  const uniqueCalls: Array<NonNullable<OpenAiResponse["output"]>[number]> = [];
  for (const call of calls) {
    const status = typeof call.status === "string" ? call.status.trim().toLowerCase() : "";
    if (status && status !== "completed") {
      continue;
    }

    const callId = typeof call.call_id === "string" ? call.call_id.trim() : "";
    const name = typeof call.name === "string" ? call.name.trim() : "";
    const args = typeof call.arguments === "string" ? call.arguments.trim() : "";
    const signature = callId || `${name}:${args}`;
    if (signature && seenSignatures.has(signature)) {
      continue;
    }
    if (signature) {
      seenSignatures.add(signature);
    }
    uniqueCalls.push(call);
  }

  return uniqueCalls;
}

async function createResponseWithRetry(
  client: OpenAI,
  requestPayload: Record<string, unknown>,
  toolRound: number,
  onDebug?: (event: DebugEvent) => void,
  onChunk?: (chunk: StreamChunk) => void,
  signal?: AbortSignal,
): Promise<OpenAiResponse> {
  let attempt = 0;
  while (true) {
    assertNotAborted(signal);
    attempt += 1;
    try {
      return await createStreamedResponse(client, requestPayload, toolRound, onDebug, onChunk, signal);
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

async function createStreamedResponse(
  client: OpenAI,
  requestPayload: Record<string, unknown>,
  toolRound: number,
  onDebug?: (event: DebugEvent) => void,
  onChunk?: (chunk: StreamChunk) => void,
  signal?: AbortSignal,
): Promise<OpenAiResponse> {
  assertNotAborted(signal);
  const stream = signal
    ? client.responses.stream(requestPayload as never, { signal } as never)
    : client.responses.stream(requestPayload as never);
  let eventCount = 0;
  let outputDeltaChars = 0;
  let reasoningDeltaChars = 0;
  const reasoningSnapshots = new Map<string, string>();

  for await (const event of stream) {
    assertNotAborted(signal);
    eventCount += 1;

    if (event.type === "response.output_text.delta") {
      outputDeltaChars += event.delta.length;
      if (event.delta) {
        onChunk?.({
          thoughts: [],
          answerText: event.delta,
          segments: [{ kind: "answer", text: event.delta }],
        });
      }
      continue;
    }

    if (event.type === "response.reasoning_text.delta") {
      const snapshot = appendReasoningSnapshot(reasoningSnapshots, event, "content_index");
      if (snapshot && event.delta) {
        reasoningDeltaChars += event.delta.length;
        onChunk?.({
          thoughts: [snapshot],
          answerText: "",
          segments: [{ kind: "thought", text: event.delta }],
        });
      }
      continue;
    }

    if (event.type === "response.reasoning_summary_text.delta") {
      const snapshot = appendReasoningSnapshot(reasoningSnapshots, event, "summary_index");
      if (snapshot && event.delta) {
        reasoningDeltaChars += event.delta.length;
        onChunk?.({
          thoughts: [snapshot],
          answerText: "",
          segments: [{ kind: "thought", text: event.delta }],
        });
      }
      continue;
    }

    if (event.type === "error") {
      const message = event.message?.trim() || "unknown stream error";
      throw new Error(`OpenAI stream failed: ${message}`);
    }

    if (event.type === "response.failed") {
      const message = event.response.error?.message?.trim() || "response failed";
      throw new Error(`OpenAI response failed: ${message}`);
    }
  }

  const finalResponse = (await stream.finalResponse()) as unknown as OpenAiResponse;
  const finalError = finalResponse.error?.message?.trim();
  if (finalError) {
    throw new Error(`OpenAI response failed: ${finalError}`);
  }

  onDebug?.({
    stage: "response_stream_summary",
    data: {
      toolRound,
      eventCount,
      outputDeltaChars,
      reasoningDeltaChars,
      responseId: finalResponse.id ?? null,
    },
  });

  return finalResponse;
}

function appendReasoningSnapshot(
  snapshots: Map<string, string>,
  event: { delta?: string; item_id?: string; content_index?: number; summary_index?: number },
  indexKind: "content_index" | "summary_index",
): string | null {
  const delta = typeof event.delta === "string" ? event.delta : "";
  if (!delta) {
    return null;
  }

  const itemId = typeof event.item_id === "string" && event.item_id.trim() ? event.item_id.trim() : "unknown";
  const indexValueRaw = event[indexKind];
  const indexValue = typeof indexValueRaw === "number" ? indexValueRaw : 0;
  const key = `${itemId}:${indexKind}:${indexValue}`;
  const next = (snapshots.get(key) ?? "") + delta;
  snapshots.set(key, next);
  const snapshot = next.trim();
  return snapshot || null;
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

function buildOpenAiCatalogHeaders(
  accessToken: string,
  chatgptAccountId: string | null,
): Record<string, string> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${accessToken}`,
    Accept: "application/json",
    originator: OPENAI_MODELS_ORIGINATOR,
    version: OPENAI_MODELS_CLIENT_VERSION,
  };
  const accountId = chatgptAccountId?.trim();
  if (accountId) {
    headers["ChatGPT-Account-ID"] = accountId;
  }
  return headers;
}

function parseOpenAiCatalogModels(payload: unknown): OpenAiCatalogModel[] {
  const records = payload as {
    models?: unknown;
    data?: unknown;
  };

  if (Array.isArray(records.models)) {
    const parsed: OpenAiCatalogModel[] = [];
    for (const item of records.models) {
      const row = item as Record<string, unknown>;
      const id = normalizeCatalogId(
        typeof row.slug === "string"
          ? row.slug
          : typeof row.id === "string"
            ? row.id
            : undefined,
      );
      if (!id) {
        continue;
      }

      const label =
        (typeof row.display_name === "string" && row.display_name.trim()) ||
        id;
      const description =
        (typeof row.description === "string" && row.description.trim()) ||
        "server-advertised openai model";

      parsed.push({
        id,
        label,
        description,
        priority: typeof row.priority === "number" ? row.priority : undefined,
        supportedInApi:
          typeof row.supported_in_api === "boolean" ? row.supported_in_api : undefined,
        visibility: typeof row.visibility === "string" ? row.visibility : undefined,
        defaultReasoningLevel: parseDefaultReasoningLevel(row),
        supportedReasoningLevels: parseReasoningLevels(
          row.supported_reasoning_levels ?? row.supported_reasoning_efforts,
        ),
      });
    }
    return parsed;
  }

  if (Array.isArray(records.data)) {
    const parsed: OpenAiCatalogModel[] = [];
    for (const item of records.data) {
      const row = item as Record<string, unknown>;
      const id = normalizeCatalogId(typeof row.id === "string" ? row.id : undefined);
      if (!id) {
        continue;
      }
      parsed.push({
        id,
        label: id,
        description: "server-advertised openai model",
        defaultReasoningLevel: parseDefaultReasoningLevel(row),
        supportedReasoningLevels: parseReasoningLevels(
          row.supported_reasoning_levels ?? row.supported_reasoning_efforts,
        ),
      });
    }
    return parsed;
  }

  throw new Error("Unexpected OpenAI models response format.");
}

function parseOpenAiUsageSnapshot(payload: unknown): OpenAiUsageSnapshot {
  const root = asRecord(payload);
  if (!root) {
    throw new Error("Unexpected OpenAI usage response format.");
  }

  let primary: OpenAiUsageWindow | null = null;
  let secondary: OpenAiUsageWindow | null = null;

  const rootRateLimit = asRecord(root.rate_limit);
  if (rootRateLimit) {
    primary = parseOpenAiUsageWindow(rootRateLimit.primary_window);
    secondary = parseOpenAiUsageWindow(rootRateLimit.secondary_window);
  }

  if (!primary && !secondary) {
    const additional = Array.isArray(root.additional_rate_limits) ? root.additional_rate_limits : [];
    for (const entry of additional) {
      const details = asRecord(entry);
      if (!details) {
        continue;
      }
      const meteredFeature = typeof details.metered_feature === "string" ? details.metered_feature.trim().toLowerCase() : "";
      const limitName = typeof details.limit_name === "string" ? details.limit_name.trim().toLowerCase() : "";
      if (meteredFeature !== "codex" && limitName !== "codex") {
        continue;
      }
      const detailsRateLimit = asRecord(details.rate_limit);
      if (!detailsRateLimit) {
        continue;
      }
      primary = parseOpenAiUsageWindow(detailsRateLimit.primary_window);
      secondary = parseOpenAiUsageWindow(detailsRateLimit.secondary_window);
      break;
    }
  }

  const rawPlanType = typeof root.plan_type === "string" ? root.plan_type.trim() : "";
  return {
    planType: rawPlanType || null,
    primary,
    secondary,
  };
}

function parseOpenAiUsageWindow(value: unknown): OpenAiUsageWindow | null {
  const row = asRecord(value);
  if (!row) {
    return null;
  }

  const usedPercentRaw = toFiniteNumber(row.used_percent);
  if (usedPercentRaw === null) {
    return null;
  }
  const usedPercent = clampPercent(usedPercentRaw);
  const limitWindowSeconds = toPositiveIntegerOrNull(row.limit_window_seconds);
  const resetAtEpochSeconds = toPositiveIntegerOrNull(row.reset_at);

  return {
    usedPercent,
    remainingPercent: clampPercent(100 - usedPercent),
    limitWindowSeconds,
    resetAtEpochSeconds,
  };
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function toFiniteNumber(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return value;
}

function toPositiveIntegerOrNull(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  const floored = Math.floor(value);
  if (floored <= 0) {
    return null;
  }
  return floored;
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 100) {
    return 100;
  }
  return value;
}

function normalizeCatalogId(value: string | undefined): string {
  const id = (value ?? "").trim();
  return id;
}

function parseDefaultReasoningLevel(row: Record<string, unknown>): string | undefined {
  const defaultLevelRaw =
    (typeof row.default_reasoning_level === "string" && row.default_reasoning_level) ||
    (typeof row.default_reasoning_effort === "string" && row.default_reasoning_effort) ||
    "";
  const normalized = defaultLevelRaw.trim().toLowerCase();
  return normalized || undefined;
}

function parseReasoningLevels(value: unknown): OpenAiCatalogReasoningLevel[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const levels: OpenAiCatalogReasoningLevel[] = [];
  for (const item of value) {
    const row = item as Record<string, unknown>;
    const effort = typeof row.effort === "string" ? row.effort.trim().toLowerCase() : "";
    if (!effort) {
      continue;
    }
    const description =
      typeof row.description === "string" ? row.description.trim() : effort;
    levels.push({
      effort,
      description,
    });
  }
  return levels;
}

function mapThinkingToOpenAiEffort(thinkingLevel: ThinkingLevel): string {
  switch (thinkingLevel) {
    case "OFF":
      return "none";
    case "MINIMAL":
      return "minimal";
    case "LOW":
      return "low";
    case "MEDIUM":
      return "medium";
    case "HIGH":
      return "high";
    case "XHIGH":
      return "xhigh";
  }
}

function summarizeHttpError(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) {
    return "empty response";
  }
  if (trimmed.length <= 180) {
    return trimmed;
  }
  return `${trimmed.slice(0, 177)}...`;
}

export const __openAiInternals = {
  computeUnstreamedAnswerDelta,
  extractAnswerDeltaFromChunk,
  extractResponseText,
  selectActionableFunctionCalls,
};
