import { randomUUID } from "node:crypto";
import { loafConfig, type ThinkingLevel } from "./config.js";
import type { ChatMessage, DebugEvent, ModelResult, StreamChunk } from "./chat-types.js";

const CLOUD_CODE_BASE_URL_STABLE = "https://daily-cloudcode-pa.googleapis.com";
const CLOUD_CODE_BASE_URL_GCP_TOS = "https://cloudcode-pa.googleapis.com";
const ANTIGRAVITY_IDE_VERSION = "1.107.0";
const CLOUD_CODE_TIMEOUT_MS = 60 * 1000;
const CLOUD_CODE_ERROR_SNIPPET_CHARS = 280;

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
        parts?: Array<{
          text?: string;
          thought?: boolean;
          inlineData?: {
            mimeType?: string;
            data?: string;
          };
        }>;
      };
    }>;
  };
  candidates?: Array<{
    content?: {
      parts?: Array<{
        text?: string;
        thought?: boolean;
        inlineData?: {
          mimeType?: string;
          data?: string;
        };
      }>;
    };
  }>;
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

  const systemInstruction = request.systemInstruction?.trim() || loafConfig.systemInstruction;
  let conversation = [...request.messages];

  const steeringMessages = request.drainSteeringMessages?.() ?? [];
  if (steeringMessages.length > 0) {
    conversation = [...conversation, ...steeringMessages];
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
  const payload = buildInferenceRequestBody({
    model: request.model,
    project,
    messages: conversation,
    systemInstruction,
  });

  onDebug?.({
    stage: "request",
    data: {
      provider: "antigravity",
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
      response,
    },
  });

  const parsed = extractInferenceText(response);
  const answer = parsed.answer.trim() || "(No response text returned)";

  onChunk?.({
    thoughts: parsed.thoughts,
    answerText: answer,
  });

  return {
    thoughts: parsed.thoughts,
    answer,
  };
}

function buildInferenceRequestBody(input: {
  model: string;
  project: string;
  messages: ChatMessage[];
  systemInstruction: string;
}): Record<string, unknown> {
  const contents = toGeminiContents(input.messages);

  const request: Record<string, unknown> = {
    model: input.model,
    contents,
  };
  // Antigravity desktop follows server-side defaults for thinking behavior.
  // Do not force client-side thinking config/budget to avoid Claude 4.x budget/max-token 400s.

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

function extractInferenceText(response: AntigravityInferenceResponse): {
  thoughts: string[];
  answer: string;
} {
  const raw = response.response ?? response;
  const firstCandidate = raw.candidates?.[0];
  const parts = firstCandidate?.content?.parts ?? [];

  const thoughts: string[] = [];
  const answerParts: string[] = [];

  for (const part of parts) {
    const text = (part.text ?? "").trim();
    if (text) {
      if (part.thought) {
        thoughts.push(text);
      } else {
        answerParts.push(text);
      }
      continue;
    }

    const imageData = part.inlineData?.data?.trim() ?? "";
    if (imageData) {
      const mimeType = part.inlineData?.mimeType?.trim() || "image/png";
      answerParts.push(`![image](data:${mimeType};base64,${imageData})`);
    }
  }

  return {
    thoughts,
    answer: answerParts.join("\n"),
  };
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
  return typeof value === "object" && value !== null;
}

function buildCloudCodeUserAgent(): string {
  const platform = process.platform === "win32" ? "windows" : process.platform;
  const arch = process.arch === "x64" ? "amd64" : process.arch === "ia32" ? "386" : process.arch;
  return `antigravity/${ANTIGRAVITY_IDE_VERSION} ${platform}/${arch}`;
}
