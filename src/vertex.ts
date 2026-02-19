import { createPartFromFunctionResponse, GoogleGenAI } from "@google/genai";
import { loafConfig, type ThinkingLevel } from "./config.js";
import { defaultToolRegistry, defaultToolRuntime } from "./tools/index.js";

export type ChatMessage = {
  role: "user" | "assistant";
  text: string;
};

export type ModelResult = {
  thoughts: string[];
  answer: string;
};

export type StreamChunk = {
  thoughts: string[];
  answerText: string;
};

export type DebugEvent = {
  stage: string;
  data: unknown;
};

export type VertexRequest = {
  apiKey: string;
  model: string;
  messages: ChatMessage[];
  thinkingLevel: ThinkingLevel;
  includeThoughts: boolean;
};

export type VertexModelCandidate = {
  id: string;
  label: string;
  description: string;
};

const MAX_429_RETRY_ATTEMPTS = 8;
const RETRY_BASE_DELAY_MS = 1_250;
const RETRY_MAX_DELAY_MS = 20_000;

export async function listVertexModels(apiKey: string): Promise<VertexModelCandidate[]> {
  const token = apiKey.trim();
  if (!token) {
    throw new Error("Missing Vertex API key.");
  }

  const ai = new GoogleGenAI({
    vertexai: true,
    apiKey: token,
  });

  const pager = await ai.models.list({
    config: {
      queryBase: true,
      pageSize: 200,
    },
  });

  const byId = new Map<string, VertexModelCandidate>();
  for await (const model of pager) {
    const supportedActions = Array.isArray(model.supportedActions)
      ? model.supportedActions
          .filter((action): action is string => typeof action === "string")
          .map((action) => action.toLowerCase())
      : [];
    if (
      supportedActions.length > 0 &&
      !supportedActions.some((action) => action.includes("generatecontent"))
    ) {
      continue;
    }

    const rawName = typeof model.name === "string" ? model.name.trim() : "";
    const id = normalizeVertexModelIdentifier(rawName);
    if (!id) {
      continue;
    }

    const existing = byId.get(id);
    if (existing) {
      continue;
    }

    const displayName = typeof model.displayName === "string" ? model.displayName.trim() : "";
    const description = typeof model.description === "string" ? model.description.trim() : "";
    byId.set(id, {
      id,
      label: displayName || toDisplayLabelFromModelId(id),
      description: description || "server-advertised vertex model",
    });
  }

  return Array.from(byId.values()).sort((a, b) => a.label.localeCompare(b.label));
}

function toSdkContents(messages: ChatMessage[]) {
  return messages.map((message) => ({
    role: message.role,
    parts: [{ text: message.text }],
  }));
}

export async function runVertexInferenceStream(
  request: VertexRequest,
  onChunk?: (chunk: StreamChunk) => void,
  onDebug?: (event: DebugEvent) => void,
): Promise<ModelResult> {
  const apiKey = request.apiKey.trim();
  if (!apiKey) {
    throw new Error("Missing Vertex API key.");
  }

  const ai = new GoogleGenAI({
    vertexai: true,
    apiKey,
  });

  const startedAt = Date.now();
  const isProModel = normalizeModelForCapabilityChecks(request.model) === "gemini-3.1-pro-preview";
  if (
    isProModel &&
    (request.thinkingLevel === "OFF" ||
      request.thinkingLevel === "MINIMAL" ||
      request.thinkingLevel === "MEDIUM")
  ) {
    throw new Error("Gemini 3.1 Pro supports only LOW or HIGH thinking.");
  }

  const thinkingConfig: any =
    request.thinkingLevel === "OFF"
      ? {
          includeThoughts: false,
          thinkingBudget: 0,
        }
      : {
          includeThoughts: request.includeThoughts,
          thinkingLevel: request.thinkingLevel as any,
        };

  const toolDeclarations = buildToolDeclarations();
  const contents: any[] = toSdkContents(messagesToModelHistory(request.messages));

  let toolRound = 0;

  while (true) {
    toolRound += 1;

    const requestPayload: any = {
      model: request.model,
      contents,
      config: {
        systemInstruction: loafConfig.systemInstruction,
        thinkingConfig,
      },
    };

    if (toolDeclarations.length > 0) {
      requestPayload.config.tools = [{ functionDeclarations: toolDeclarations }];
      requestPayload.config.toolConfig = {
        functionCallingConfig: {
          mode: "AUTO",
        },
      };
    }

    onDebug?.({
      stage: "request",
      data: {
        toolRound,
        payload: requestPayload,
      },
    });

    const response = (await generateContentWithRetry(ai, requestPayload, toolRound, onDebug)) as any;
    onDebug?.({
      stage: "response_raw",
      data: {
        toolRound,
        response,
      },
    });

    const functionCalls = extractFunctionCalls(response);
    if (functionCalls.length > 0) {
      onDebug?.({
        stage: "tool_calls",
        data: {
          toolRound,
          functionCalls,
        },
      });

      const functionResponseParts: any[] = [];
      const executed: Array<{
        name: string;
        ok: boolean;
        input?: Record<string, unknown>;
        result: unknown;
        error?: string;
      }> = [];

      for (let i = 0; i < functionCalls.length; i += 1) {
        const call = functionCalls[i]!;
        const callName = String(call.name ?? "").trim();
        const callArgs = (call.args ?? {}) as Record<string, unknown>;
        const callId = String(call.id ?? `${callName || "tool"}-${toolRound}-${i}`);

        const toolResult = await defaultToolRuntime.execute(
          {
            id: callId,
            name: callName,
            input: callArgs as any,
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

        functionResponseParts.push(
          createPartFromFunctionResponse(callId, callName, responsePayload),
        );
      }

      onDebug?.({
        stage: "tool_results",
        data: {
          toolRound,
          executed,
        },
      });

      const modelContent = response.candidates?.[0]?.content ?? {
        role: "model",
        parts: functionCalls.map((call) => ({ functionCall: call })),
      };
      contents.push(modelContent);
      contents.push({
        role: "user",
        parts: functionResponseParts,
      });

      continue;
    }

    const parsed = extractResponseText(response);
    if (onChunk) {
      onChunk({
        thoughts: parsed.thoughts,
        answerText: parsed.answer,
      });
    }

    onDebug?.({
      stage: "response_final",
      data: {
        toolRound,
        thoughtCount: parsed.thoughts.length,
        answerLength: parsed.answer.length,
        durationMs: Date.now() - startedAt,
        answerPreview: parsed.answer.slice(0, 400),
      },
    });

    return {
      thoughts: parsed.thoughts,
      answer: parsed.answer,
    };
  }
}

export async function runVertexInference(request: VertexRequest): Promise<ModelResult> {
  return runVertexInferenceStream(request);
}

function normalizeThoughts(thoughts: string[]): string[] {
  const normalized: string[] = [];
  for (const item of thoughts) {
    const text = item.trim();
    if (!text) {
      continue;
    }
    const last = normalized.at(-1);
    if (!last) {
      normalized.push(text);
      continue;
    }
    if (text.startsWith(last)) {
      normalized[normalized.length - 1] = text;
      continue;
    }
    if (last.startsWith(text)) {
      continue;
    }
    normalized.push(text);
  }
  return normalized;
}

function messagesToModelHistory(messages: ChatMessage[]): ChatMessage[] {
  if (messages.length === 0) {
    return [{ role: "user", text: "Hello." }];
  }
  return messages;
}

function buildToolDeclarations(): Array<{
  name: string;
  description: string;
  parametersJsonSchema: Record<string, unknown>;
}> {
  return defaultToolRegistry.list().map((tool) => {
    const schema = tool.inputSchema ?? {
      type: "object",
      properties: {},
      required: [],
    };

    return {
      name: tool.name,
      description: tool.description,
      parametersJsonSchema: {
        type: schema.type,
        properties: schema.properties,
        required: schema.required ?? [],
        additionalProperties: false,
      },
    };
  });
}

function extractFunctionCalls(response: any): Array<{
  id?: string;
  name?: string;
  args?: Record<string, unknown>;
}> {
  if (Array.isArray(response.functionCalls) && response.functionCalls.length > 0) {
    return response.functionCalls;
  }

  const calls: Array<{ id?: string; name?: string; args?: Record<string, unknown> }> = [];
  for (const candidate of response.candidates ?? []) {
    for (const part of candidate.content?.parts ?? []) {
      if (part.functionCall && typeof part.functionCall === "object") {
        calls.push(part.functionCall);
      }
    }
  }
  return calls;
}

function extractResponseText(response: any): { thoughts: string[]; answer: string } {
  const thoughtSnapshots: string[] = [];
  const answerParts: string[] = [];

  for (const candidate of response.candidates ?? []) {
    for (const part of candidate.content?.parts ?? []) {
      const text = typeof part.text === "string" ? part.text.trim() : "";
      if (!text) {
        continue;
      }
      if (part.thought) {
        thoughtSnapshots.push(text);
      } else {
        answerParts.push(text);
      }
    }
  }

  const normalizedThoughts = normalizeThoughts(thoughtSnapshots);
  const answerFromParts = answerParts.join("\n\n").trim();
  const fallbackText = typeof response.text === "string" ? response.text.trim() : "";
  const answer = answerFromParts || fallbackText || "(No response text returned)";

  return {
    thoughts: normalizedThoughts,
    answer,
  };
}

async function generateContentWithRetry(
  ai: GoogleGenAI,
  requestPayload: unknown,
  toolRound: number,
  onDebug?: (event: DebugEvent) => void,
): Promise<unknown> {
  let attempt = 0;
  while (true) {
    attempt += 1;
    try {
      return await ai.models.generateContent(requestPayload as any);
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

function isRetryable429Error(error: unknown): boolean {
  const directCode = pickNumericCode(error);
  if (directCode === 429) {
    return true;
  }

  const text = summarizeError(error).toLowerCase();
  if (!text) {
    return false;
  }
  if (text.includes("resource_exhausted")) {
    return true;
  }
  if (text.includes("too many requests")) {
    return true;
  }
  if (text.includes("\"code\":429")) {
    return true;
  }
  if (/\b429\b/.test(text) && (text.includes("try again later") || text.includes("quota") || text.includes("rate"))) {
    return true;
  }
  return false;
}

function pickNumericCode(value: unknown): number | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  const candidates = [record.code, record.status, record.statusCode];
  for (const candidate of candidates) {
    const asNumber = toNumber(candidate);
    if (asNumber !== undefined) {
      return asNumber;
    }
  }
  if (record.error && typeof record.error === "object") {
    const nested = pickNumericCode(record.error);
    if (nested !== undefined) {
      return nested;
    }
  }
  if (record.cause && typeof record.cause === "object") {
    const nested = pickNumericCode(record.cause);
    if (nested !== undefined) {
      return nested;
    }
  }
  return undefined;
}

function toNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
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

function normalizeVertexModelIdentifier(rawName: string): string {
  const trimmed = rawName.trim();
  if (!trimmed) {
    return "";
  }

  const publisherModelMatch = trimmed.match(/publishers\/([^/]+)\/models\/([^/]+)/i);
  if (publisherModelMatch?.[1] && publisherModelMatch[2]) {
    return `${publisherModelMatch[1].toLowerCase()}/${publisherModelMatch[2]}`;
  }

  const modelsMatch = trimmed.match(/(?:^|\/)models\/([^/]+)$/i);
  if (modelsMatch?.[1]) {
    return modelsMatch[1];
  }

  if (trimmed.startsWith("models/")) {
    return trimmed.slice("models/".length);
  }

  return trimmed;
}

function normalizeModelForCapabilityChecks(modelId: string): string {
  const normalized = modelId.trim().toLowerCase();
  if (!normalized) {
    return "";
  }

  const modelsMatch = normalized.match(/(?:^|\/)models\/([^/]+)$/);
  if (modelsMatch?.[1]) {
    return modelsMatch[1];
  }

  const slashIndex = normalized.lastIndexOf("/");
  if (slashIndex >= 0) {
    return normalized.slice(slashIndex + 1);
  }

  return normalized;
}

function toDisplayLabelFromModelId(modelId: string): string {
  const slashIndex = modelId.lastIndexOf("/");
  const value = slashIndex >= 0 ? modelId.slice(slashIndex + 1) : modelId;
  return value.replace(/[-_]+/g, " ");
}
