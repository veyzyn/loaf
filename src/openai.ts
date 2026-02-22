import OpenAI from "openai";
import { randomUUID } from "node:crypto";
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
  sessionId?: string;
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
  output?: OpenAiOutputItem[];
};

type OpenAiOutputItem = {
  type?: string;
  id?: string;
  status?: string;
  call_id?: string;
  name?: string;
  arguments?: string;
  content?: Array<{
    type?: string;
    text?: string | { value?: string };
    value?: string;
  }>;
};

type OpenAiFunctionCallOutputBody = string | Array<Record<string, unknown>>;

type OpenAiStreamResult = {
  response: OpenAiResponse;
  outputItemsDone: OpenAiOutputItem[];
  completedEventSeen: boolean;
  turnStateToken: string | null;
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
  contextWindowTokens?: number;
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
  const sessionId = request.sessionId?.trim() || randomUUID();
  let turnStateToken: string | null = null;

  let conversationInput: Array<Record<string, unknown>> = [];
  let pendingInput = messagesToResponsesInput(request.messages);
  const startedAt = Date.now();
  let toolRound = 0;

  while (true) {
    assertNotAborted(request.signal);

    const roundInput: Array<Record<string, unknown>> = [...pendingInput];
    pendingInput = [];

    const steeringMessages = request.drainSteeringMessages?.() ?? [];
    if (steeringMessages.length > 0) {
      roundInput.push(...chatMessagesToResponsesInput(steeringMessages));
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

    const fullInput = [...conversationInput, ...roundInput];

    toolRound += 1;
    const toolDeclarations = buildToolDeclarations();
    const requestPayload: Record<string, unknown> = {
      model: request.model,
      instructions: systemInstruction,
      tools: toolDeclarations.declarations,
      tool_choice: "auto",
      parallel_tool_calls: false,
      store: false,
      stream: true,
    };

    requestPayload.reasoning = {
      effort: mapThinkingToOpenAiEffort(request.thinkingLevel),
    };
    // ChatGPT Codex HTTP streaming currently rejects follow-up `previous_response_id`
    // requests for this interleaved tool path. Keep transport stateless and replay
    // ordered input items each round.
    requestPayload.input = fullInput;
    const requestHeaders: Record<string, string> = {
      session_id: sessionId,
    };
    if (turnStateToken) {
      requestHeaders["x-codex-turn-state"] = turnStateToken;
    }

    onDebug?.({
      stage: "request",
      data: {
        toolRound,
        payload: requestPayload,
        headers: requestHeaders,
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

    const streamResult = await createResponseWithRetry(
      client,
      requestPayload,
      requestHeaders,
      toolRound,
      onDebug,
      handleChunkThisRound,
      request.signal,
    );
    const response = streamResult.response;
    if (streamResult.turnStateToken) {
      turnStateToken = streamResult.turnStateToken;
    }
    const outputItems = pickOutputItemsForFollowUp(streamResult.outputItemsDone, response);
    conversationInput = fullInput;

    onDebug?.({
      stage: "response_raw",
      data: {
        toolRound,
        response,
        outputItemsDone: streamResult.outputItemsDone.length,
        completedEventSeen: streamResult.completedEventSeen,
      },
    });

    const functionCalls = selectActionableFunctionCalls(outputItems);
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
        output: OpenAiFunctionCallOutputBody;
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

        functionOutputs.push({
          type: "function_call_output",
          call_id: callId,
          output: toFunctionCallOutputBody(toolResult.output, toolResult.error),
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

      const followUpInputStateless = buildFunctionCallFollowUpInput({
        output: outputItems,
        functionCalls: replayFunctionCalls,
        functionOutputs,
      });
      pendingInput = [...followUpInputStateless];
      continue;
    }

    const answer = extractResponseText(response).trim();
    const responseStatus = typeof response.status === "string" ? response.status.trim().toLowerCase() : "";
    if (!streamResult.completedEventSeen && responseStatus !== "completed") {
      onDebug?.({
        stage: "response_continue",
        data: {
          toolRound,
          responseStatus,
          reason: "stream closed before response.completed",
        },
      });
      continue;
    }
    if (responseStatus === "failed" || responseStatus === "cancelled") {
      throw new Error(`OpenAI response did not complete successfully (status: ${responseStatus}).`);
    }
    if (responseStatus && responseStatus !== "completed") {
      onDebug?.({
        stage: "response_continue",
        data: {
          toolRound,
          responseStatus,
          reason: answer ? "non-completed status" : "empty assistant text",
        },
      });
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

function buildFunctionCallFollowUpInput(input: {
  output?: OpenAiOutputItem[];
  functionCalls: Array<{
    type: "function_call";
    call_id: string;
    name: string;
    arguments: string;
  }>;
  functionOutputs: Array<{
    type: "function_call_output";
    call_id: string;
    output: OpenAiFunctionCallOutputBody;
  }>;
}): Array<Record<string, unknown>> {
  const followUp: Array<Record<string, unknown>> = [];
  const callByCallId = new Map(
    input.functionCalls.map((call) => [call.call_id, call] as const),
  );
  const outputByCallId = new Map(
    input.functionOutputs.map((result) => [result.call_id, result] as const),
  );
  const replayedCallIds: string[] = [];
  const replayedCallIdSet = new Set<string>();

  for (const item of input.output ?? []) {
    if (item?.type === "message") {
      const assistantMessage = toAssistantFollowUpMessage(item);
      if (assistantMessage) {
        followUp.push(assistantMessage);
      }
      continue;
    }

    if (item?.type === "function_call") {
      const callId = typeof item.call_id === "string" ? item.call_id.trim() : "";
      if (!callId) {
        continue;
      }
      const replayCall = callByCallId.get(callId);
      if (!replayCall) {
        continue;
      }
      followUp.push(replayCall);
      replayedCallIds.push(callId);
      replayedCallIdSet.add(callId);
    }
  }

  for (const call of input.functionCalls) {
    if (replayedCallIdSet.has(call.call_id)) {
      continue;
    }
    followUp.push(call);
    replayedCallIds.push(call.call_id);
    replayedCallIdSet.add(call.call_id);
  }

  for (const callId of replayedCallIds) {
    const replayOutput = outputByCallId.get(callId);
    if (replayOutput) {
      followUp.push(replayOutput);
    }
  }

  return followUp;
}

function toAssistantFollowUpMessage(item: {
  content?: Array<{
    type?: string;
    text?: string | { value?: string };
    value?: string;
  }>;
}): Record<string, unknown> | null {
  if (!Array.isArray(item.content)) {
    return null;
  }

  const parts: string[] = [];
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

  const content = parts.join("\n\n").trim();
  if (!content) {
    return null;
  }

  return {
    type: "message",
    role: "assistant",
    content,
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

function computeIncrementalInput(input: {
  previousResponseId: string | null;
  requestSignature: string;
  lastRequestSignature: string | null;
  fullInput: Array<Record<string, unknown>>;
  lastRequestInput: Array<Record<string, unknown>>;
  lastResponseAddedInput: Array<Record<string, unknown>>;
}): Array<Record<string, unknown>> | null {
  if (!input.previousResponseId) {
    return null;
  }
  if (!input.lastRequestSignature || input.requestSignature !== input.lastRequestSignature) {
    return null;
  }
  if (input.lastRequestInput.length === 0) {
    return null;
  }

  const baseline = [...input.lastRequestInput, ...input.lastResponseAddedInput];
  if (!isStrictInputExtension(input.fullInput, baseline)) {
    return null;
  }

  return input.fullInput.slice(baseline.length);
}

function isStrictInputExtension(
  fullInput: Array<Record<string, unknown>>,
  baseline: Array<Record<string, unknown>>,
): boolean {
  if (baseline.length === 0 || baseline.length >= fullInput.length) {
    return false;
  }
  for (let i = 0; i < baseline.length; i += 1) {
    const baselineItem = baseline[i];
    const fullItem = fullInput[i];
    if (safeStableStringify(baselineItem) !== safeStableStringify(fullItem)) {
      return false;
    }
  }
  return true;
}

function toFunctionCallOutputBody(
  output: unknown,
  errorMessage?: string,
): OpenAiFunctionCallOutputBody {
  const contentItems = asFunctionCallOutputContentItems(output);
  if (contentItems) {
    return contentItems;
  }

  if (typeof output === "string") {
    return output;
  }

  if (output === undefined || output === null) {
    return errorMessage?.trim() || "";
  }

  if (output instanceof Error) {
    return output.message || String(output);
  }

  return safeStableStringify(output);
}

function asFunctionCallOutputContentItems(
  output: unknown,
): Array<Record<string, unknown>> | null {
  if (!Array.isArray(output) || output.length === 0) {
    return null;
  }

  const items: Array<Record<string, unknown>> = [];
  for (const entry of output) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      return null;
    }
    const row = entry as Record<string, unknown>;
    const type = typeof row.type === "string" ? row.type : "";
    if (type !== "input_text" && type !== "input_image") {
      return null;
    }
    if (type === "input_text" && typeof row.text !== "string") {
      return null;
    }
    if (type === "input_image" && typeof row.image_url !== "string") {
      return null;
    }
    items.push(row);
  }

  return items;
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

function safeStableStringify(value: unknown): string {
  if (value === undefined) {
    return "undefined";
  }
  try {
    return JSON.stringify(value, stableJsonReplacer);
  } catch {
    return String(value);
  }
}

function stableJsonReplacer(_key: string, value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }

  const record = value as Record<string, unknown>;
  const sorted: Record<string, unknown> = {};
  for (const key of Object.keys(record).sort()) {
    sorted[key] = record[key];
  }
  return sorted;
}

function readOutputItemFromStreamEvent(event: unknown): OpenAiOutputItem | null {
  const root = asRecord(event);
  if (!root) {
    return null;
  }
  const item = asRecord(root.item);
  if (!item) {
    return null;
  }

  return item as OpenAiOutputItem;
}

function pickOutputItemsForFollowUp(
  outputItemsDone: OpenAiOutputItem[],
  response: OpenAiResponse,
): OpenAiOutputItem[] {
  if (outputItemsDone.length > 0) {
    return outputItemsDone;
  }
  return Array.isArray(response.output) ? response.output : [];
}

function readTrimmedString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
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

function selectActionableFunctionCalls(output: OpenAiOutputItem[] | undefined): OpenAiOutputItem[] {
  const calls = (output ?? []).filter((item) => item?.type === "function_call");
  if (calls.length === 0) {
    return [];
  }

  const seenSignatures = new Set<string>();
  const uniqueCalls: OpenAiOutputItem[] = [];
  for (const call of calls) {
    const status = typeof call.status === "string" ? call.status.trim().toLowerCase() : "";
    if (status === "failed" || status === "cancelled") {
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
  requestHeaders: Record<string, string>,
  toolRound: number,
  onDebug?: (event: DebugEvent) => void,
  onChunk?: (chunk: StreamChunk) => void,
  signal?: AbortSignal,
): Promise<OpenAiStreamResult> {
  let attempt = 0;
  while (true) {
    assertNotAborted(signal);
    attempt += 1;
    try {
      return await createStreamedResponse(
        client,
        requestPayload,
        requestHeaders,
        toolRound,
        onDebug,
        onChunk,
        signal,
      );
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
  requestHeaders: Record<string, string>,
  toolRound: number,
  onDebug?: (event: DebugEvent) => void,
  onChunk?: (chunk: StreamChunk) => void,
  signal?: AbortSignal,
): Promise<OpenAiStreamResult> {
  assertNotAborted(signal);
  const requestOptions: Record<string, unknown> = {
    headers: requestHeaders,
    stream: true,
  };
  if (signal) {
    requestOptions.signal = signal;
  }
  const responsePromise = client.responses.create(requestPayload as never, requestOptions as never);
  const streamWithResponse = await responsePromise.withResponse();
  const stream = streamWithResponse.data as unknown as AsyncIterable<any>;
  const turnStateToken = readTrimmedString(
    streamWithResponse.response.headers.get("x-codex-turn-state"),
  );
  let eventCount = 0;
  let outputDeltaChars = 0;
  let reasoningDeltaChars = 0;
  const reasoningSnapshots = new Map<string, string>();
  const outputItemsDone: OpenAiOutputItem[] = [];
  let completedEventSeen = false;
  let finalResponse: OpenAiResponse | null = null;

  for await (const event of stream) {
    assertNotAborted(signal);
    eventCount += 1;
    if (event.type === "response.created") {
      const createdResponse = asRecord(event.response);
      if (createdResponse) {
        finalResponse = createdResponse as OpenAiResponse;
      }
      continue;
    }

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

    if (event.type === "response.output_item.done") {
      const outputItem = readOutputItemFromStreamEvent(event);
      if (outputItem) {
        outputItemsDone.push(outputItem);
      }
      continue;
    }

    if (event.type === "response.completed") {
      completedEventSeen = true;
      const completedResponse = asRecord(event.response);
      if (completedResponse) {
        finalResponse = completedResponse as OpenAiResponse;
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
  if (!finalResponse) {
    throw new Error("OpenAI stream ended without a response snapshot.");
  }
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
      turnStateToken: turnStateToken || null,
    },
  });

  return {
    response: finalResponse,
    outputItemsDone,
    completedEventSeen,
    turnStateToken: turnStateToken || null,
  };
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
        contextWindowTokens: parseContextWindowTokens(row),
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
        contextWindowTokens: parseContextWindowTokens(row),
      });
    }
    return parsed;
  }

  throw new Error("Unexpected OpenAI models response format.");
}

function parseContextWindowTokens(row: Record<string, unknown>): number | undefined {
  const directCandidates = [
    row.context_window,
    row.context_length,
    row.max_context_tokens,
    row.max_input_tokens,
    row.contextWindow,
    row.contextLength,
  ];

  for (const candidate of directCandidates) {
    const parsed = parsePositiveInteger(candidate);
    if (parsed) {
      return parsed;
    }
  }

  const architecture = asRecord(row.architecture);
  if (architecture) {
    const architectureCandidates = [
      architecture.context_window,
      architecture.context_length,
      architecture.max_context_tokens,
      architecture.max_input_tokens,
    ];
    for (const candidate of architectureCandidates) {
      const parsed = parsePositiveInteger(candidate);
      if (parsed) {
        return parsed;
      }
    }
  }

  return undefined;
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

function parsePositiveInteger(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    const floored = Math.floor(value);
    return floored > 0 ? floored : undefined;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return undefined;
    }
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) {
      const floored = Math.floor(numeric);
      return floored > 0 ? floored : undefined;
    }
  }
  return undefined;
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
  buildFunctionCallFollowUpInput,
  pickOutputItemsForFollowUp,
  computeIncrementalInput,
  toFunctionCallOutputBody,
  computeUnstreamedAnswerDelta,
  extractAnswerDeltaFromChunk,
  extractResponseText,
  selectActionableFunctionCalls,
};
