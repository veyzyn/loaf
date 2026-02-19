import OpenAI from "openai";
import { loafConfig, type ThinkingLevel } from "./config.js";
import { defaultToolRegistry, defaultToolRuntime } from "./tools/index.js";
import type { ChatMessage, DebugEvent, ModelResult, StreamChunk } from "./vertex.js";

export type OpenAIRequest = {
  accessToken: string;
  chatgptAccountId: string | null;
  model: string;
  messages: ChatMessage[];
  thinkingLevel: ThinkingLevel;
  includeThoughts: boolean;
  systemInstruction?: string;
};

const CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex";
const OPENAI_MODELS_CLIENT_VERSION = "0.1.0";
const OPENAI_MODELS_ORIGINATOR = "codex_cli_rs";
const MAX_429_RETRY_ATTEMPTS = 8;
const RETRY_BASE_DELAY_MS = 1_250;
const RETRY_MAX_DELAY_MS = 20_000;

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

export function createChatgptCodexClient(
  accessToken: string,
  chatgptAccountId: string | null,
): OpenAI {
  const token = accessToken.trim();
  if (!token) {
    throw new Error("Missing OpenAI ChatGPT access token.");
  }

  const defaultHeaders: Record<string, string> = {
    version: "0.1.0",
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

export async function listOpenAiModelIds(
  accessToken: string,
  chatgptAccountId: string | null,
): Promise<string[]> {
  const models = await listOpenAiCatalogModels(accessToken, chatgptAccountId);
  return models.map((model) => model.id);
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

  const tools = buildToolDeclarations();
  let conversationInput = messagesToResponsesInput(request.messages);
  const startedAt = Date.now();
  let toolRound = 0;

  while (true) {
    toolRound += 1;
    const requestPayload: Record<string, unknown> = {
      model: request.model,
      instructions: systemInstruction,
      input: conversationInput,
      tools,
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

    const response = await createResponseWithRetry(
      client,
      requestPayload,
      toolRound,
      onDebug,
      onChunk,
    );

    onDebug?.({
      stage: "response_raw",
      data: {
        toolRound,
        response,
      },
    });

    const functionCalls = (response.output ?? []).filter((item) => item?.type === "function_call");
    if (functionCalls.length > 0) {
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
        const call = functionCalls[i] ?? {};
        const callName = String(call.name ?? "").trim();
        const callId = String(call.call_id ?? `${callName || "tool"}-${toolRound}-${i}`);
        const callArgumentsRaw = typeof call.arguments === "string" && call.arguments.trim() ? call.arguments : "{}";
        const callArgs = safeParseObject(callArgumentsRaw);

        const toolResult = await defaultToolRuntime.execute(
          {
            id: callId,
            name: callName,
            input: callArgs as never,
          },
          {
            now: new Date(),
          },
        );

        executed.push({
          name: callName,
          ok: toolResult.ok,
          input: callArgs,
          result: toolResult.output,
          error: toolResult.error,
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
          name: callName,
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

    if (onChunk) {
      onChunk({
        thoughts: [],
        answerText: finalAnswer,
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

function buildToolDeclarations(): Array<Record<string, unknown>> {
  return defaultToolRegistry.list().map((tool) => {
    const schema = tool.inputSchema ?? {
      type: "object",
      properties: {},
      required: [],
    };

    return {
      type: "function",
      name: tool.name,
      description: tool.description,
      parameters: {
        type: schema.type,
        properties: schema.properties,
        required: schema.required ?? [],
        additionalProperties: false,
      },
      strict: false,
    };
  });
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
    content: message.text,
  }));
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
      text?: string;
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
      if (contentItem?.type !== "output_text") {
        continue;
      }
      if (typeof contentItem.text === "string" && contentItem.text.trim()) {
        parts.push(contentItem.text.trim());
      }
    }
  }

  return parts.join("\n\n").trim();
}

async function createResponseWithRetry(
  client: OpenAI,
  requestPayload: Record<string, unknown>,
  toolRound: number,
  onDebug?: (event: DebugEvent) => void,
  onChunk?: (chunk: StreamChunk) => void,
): Promise<OpenAiResponse> {
  let attempt = 0;
  while (true) {
    attempt += 1;
    try {
      return await createStreamedResponse(client, requestPayload, toolRound, onDebug, onChunk);
    } catch (error) {
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
      await sleep(delayMs);
    }
  }
}

async function createStreamedResponse(
  client: OpenAI,
  requestPayload: Record<string, unknown>,
  toolRound: number,
  onDebug?: (event: DebugEvent) => void,
  onChunk?: (chunk: StreamChunk) => void,
): Promise<OpenAiResponse> {
  const stream = client.responses.stream(requestPayload as never);
  let eventCount = 0;
  let outputDeltaChars = 0;
  let reasoningDeltaChars = 0;
  let reasoningBuffer = "";

  for await (const event of stream) {
    eventCount += 1;

    if (event.type === "response.output_text.delta") {
      outputDeltaChars += event.delta.length;
      if (event.delta) {
        onChunk?.({
          thoughts: [],
          answerText: event.delta,
        });
      }
      continue;
    }

    if (event.type === "response.reasoning_text.delta") {
      reasoningDeltaChars += event.delta.length;
      if (event.delta) {
        reasoningBuffer += event.delta;
        const snapshot = reasoningBuffer.trim();
        if (snapshot) {
          onChunk?.({
            thoughts: [snapshot],
            answerText: "",
          });
        }
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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
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
        typeof row.slug === "string" ? row.slug : undefined,
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
