import type { ThinkingLevel } from "./config.js";

const CLOUD_CODE_BASE_URL_STABLE = "https://daily-cloudcode-pa.googleapis.com";
const CLOUD_CODE_BASE_URL_GCP_TOS = "https://cloudcode-pa.googleapis.com";
const CLOUD_CODE_TIMEOUT_MS = 20 * 1000;
const CLOUD_CODE_ERROR_SNIPPET_CHARS = 4_000;
const ANTIGRAVITY_IDE_VERSION = "1.107.0";

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

type FetchAvailableModelsResponse = {
  models?: Record<string, AntigravityModelDetails>;
  agentModelSorts?: Array<{
    groups?: Array<{
      modelIds?: string[];
    }>;
  }>;
};

type AntigravityModelDetails = {
  displayName?: string;
  description?: string;
  supportsThinking?: boolean;
  supportsImages?: boolean;
  maxTokens?: number;
  maxOutputTokens?: number;
  thinkingBudget?: number;
  minThinkingBudget?: number;
  quotaInfo?: {
    remainingFraction?: number;
    resetTime?: string;
  };
  disabled?: boolean;
  beta?: boolean;
  preview?: boolean;
  tagTitle?: string;
  tagDescription?: string;
};

export type AntigravityDiscoveredModel = {
  id: string;
  label: string;
  description: string;
  supportsThinking: boolean;
  supportsImages: boolean;
  maxTokens?: number;
  maxOutputTokens?: number;
  contextWindowTokens?: number;
  thinkingBudget?: number;
  minThinkingBudget?: number;
};

export type AntigravityModelsDiscoveryResult = {
  models: AntigravityDiscoveredModel[];
  source: "remote";
};

type AntigravityUsageModelQuota = {
  name: string;
  remainingPercent: number;
  resetTime: string | null;
};

export type AntigravityUsageSnapshot = {
  models: AntigravityUsageModelQuota[];
};

export async function discoverAntigravityModelOptions(request: {
  accessToken: string;
}): Promise<AntigravityModelsDiscoveryResult> {
  const accessToken = request.accessToken.trim();
  if (!accessToken) {
    return { models: [], source: "remote" };
  }

  const metadata = buildCloudCodeMetadata();
  const loadCodeAssist = await callCloudCode<LoadCodeAssistResponse>({
    accessToken,
    baseUrl: CLOUD_CODE_BASE_URL_STABLE,
    path: "v1internal:loadCodeAssist",
    body: { metadata },
  });
  const project = loadCodeAssist.cloudaicompanionProject?.trim() ?? "";
  const isGcpTos = loadCodeAssist.currentTier?.usesGcpTos === true;
  const baseUrl = isGcpTos ? CLOUD_CODE_BASE_URL_GCP_TOS : CLOUD_CODE_BASE_URL_STABLE;

  const fetchResponse = await callCloudCode<FetchAvailableModelsResponse>({
    accessToken,
    baseUrl,
    path: "v1internal:fetchAvailableModels",
    body: { project },
  });

  return {
    models: normalizeAntigravityModels(fetchResponse),
    source: "remote",
  };
}

export async function fetchAntigravityUsageSnapshot(request: {
  accessToken: string;
}): Promise<AntigravityUsageSnapshot> {
  const accessToken = request.accessToken.trim();
  if (!accessToken) {
    throw new Error("Missing Antigravity OAuth token.");
  }

  const metadata = buildCloudCodeMetadata();
  const loadCodeAssist = await callCloudCode<LoadCodeAssistResponse>({
    accessToken,
    baseUrl: CLOUD_CODE_BASE_URL_STABLE,
    path: "v1internal:loadCodeAssist",
    body: { metadata },
  });

  const project = loadCodeAssist.cloudaicompanionProject?.trim() ?? "";
  if (!project) {
    throw new Error("antigravity project resolution failed. try /auth antigravity oauth again.");
  }

  const isGcpTos = loadCodeAssist.currentTier?.usesGcpTos === true;
  const baseUrl = isGcpTos ? CLOUD_CODE_BASE_URL_GCP_TOS : CLOUD_CODE_BASE_URL_STABLE;
  const fetchResponse = await callCloudCode<FetchAvailableModelsResponse>({
    accessToken,
    baseUrl,
    path: "v1internal:fetchAvailableModels",
    body: { project },
  });

  return {
    models: normalizeAntigravityUsageModels(fetchResponse),
  };
}

export function antigravityModelToThinkingLevels(
  model: AntigravityDiscoveredModel,
): {
  supportedThinkingLevels: ThinkingLevel[];
  defaultThinkingLevel: ThinkingLevel;
} {
  if (!model.supportsThinking) {
    return {
      supportedThinkingLevels: ["OFF"],
      defaultThinkingLevel: "OFF",
    };
  }

  return {
    supportedThinkingLevels: ["OFF", "MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH"],
    defaultThinkingLevel: inferDefaultThinkingLevel(model),
  };
}

function normalizeAntigravityModels(response: FetchAvailableModelsResponse): AntigravityDiscoveredModel[] {
  const modelMap = response.models ?? {};
  const orderedIds = getOrderedModelIds(response, modelMap);
  const rows: AntigravityDiscoveredModel[] = [];

  for (const id of orderedIds) {
    const details = modelMap[id];
    const modelId = id.trim();
    if (!modelId || !details || details.disabled) {
      continue;
    }

    const label = (details.displayName ?? "").trim() || modelId;
    const detailsBits = [
      "antigravity catalog model",
      details.description?.trim() || "",
      details.beta ? "beta" : "",
      details.preview ? "preview" : "",
      details.tagTitle?.trim() || "",
      details.tagDescription?.trim() || "",
    ].filter(Boolean);

    rows.push({
      id: modelId,
      label,
      description: detailsBits.join(" | "),
      supportsThinking: details.supportsThinking !== false,
      supportsImages: details.supportsImages === true,
      maxTokens: normalizePositiveInteger(details.maxTokens),
      maxOutputTokens: normalizePositiveInteger(details.maxOutputTokens),
      contextWindowTokens: inferContextWindowTokens(details),
      thinkingBudget: toFiniteNumber(details.thinkingBudget),
      minThinkingBudget: toFiniteNumber(details.minThinkingBudget),
    });
  }

  return rows;
}

function normalizeAntigravityUsageModels(response: FetchAvailableModelsResponse): AntigravityUsageModelQuota[] {
  const modelMap = response.models ?? {};
  const orderedIds = getOrderedModelIds(response, modelMap);
  const rows: AntigravityUsageModelQuota[] = [];

  for (const id of orderedIds) {
    const details = modelMap[id];
    const name = id.trim();
    if (!name || !details) {
      continue;
    }
    if (
      !name.includes("claude") &&
      !name.includes("gemini") &&
      !name.includes("image") &&
      !name.includes("imagen")
    ) {
      continue;
    }

    const quotaInfo = details.quotaInfo;
    const remainingFraction =
      typeof quotaInfo?.remainingFraction === "number" && Number.isFinite(quotaInfo.remainingFraction)
        ? quotaInfo.remainingFraction
        : null;
    if (remainingFraction === null) {
      continue;
    }

    rows.push({
      name,
      remainingPercent: clampPercent(Math.round(remainingFraction * 100)),
      resetTime: typeof quotaInfo?.resetTime === "string" && quotaInfo.resetTime.trim() ? quotaInfo.resetTime : null,
    });
  }

  return rows;
}

function getOrderedModelIds(
  response: FetchAvailableModelsResponse,
  modelMap: Record<string, AntigravityModelDetails>,
): string[] {
  const ordered = new Set<string>();

  for (const sort of response.agentModelSorts ?? []) {
    for (const group of sort.groups ?? []) {
      for (const modelId of group.modelIds ?? []) {
        if (typeof modelId !== "string" || !Object.hasOwn(modelMap, modelId)) {
          continue;
        }
        ordered.add(modelId);
      }
    }
  }

  for (const modelId of Object.keys(modelMap).sort((left, right) => left.localeCompare(right))) {
    ordered.add(modelId);
  }

  return [...ordered];
}

function inferDefaultThinkingLevel(model: AntigravityDiscoveredModel): ThinkingLevel {
  const normalizedId = model.id.toLowerCase();
  const normalizedLabel = model.label.toLowerCase();
  if (normalizedId.includes("high") || normalizedLabel.includes("high")) {
    return "HIGH";
  }
  if (normalizedId.includes("low") || normalizedLabel.includes("low")) {
    return "LOW";
  }

  const budget = model.thinkingBudget;
  if (budget === -1) {
    return "MEDIUM";
  }
  if (typeof budget === "number") {
    if (budget <= 0) {
      return "OFF";
    }
    if (budget <= 128) {
      return "LOW";
    }
    if (budget <= 1024) {
      return "MINIMAL";
    }
    if (budget <= 4096) {
      return "LOW";
    }
    if (budget <= 10000) {
      return "MEDIUM";
    }
    if (budget <= 20000) {
      return "HIGH";
    }
    return "XHIGH";
  }

  return "MEDIUM";
}

function toFiniteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizePositiveInteger(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  const floored = Math.floor(value);
  return floored > 0 ? floored : undefined;
}

function inferContextWindowTokens(details: AntigravityModelDetails): number | undefined {
  const maxTokens = normalizePositiveInteger(details.maxTokens);
  const maxOutputTokens = normalizePositiveInteger(details.maxOutputTokens);
  if (maxTokens && maxOutputTokens && maxTokens > maxOutputTokens) {
    return maxTokens - maxOutputTokens;
  }
  return maxTokens;
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value <= 0) {
    return 0;
  }
  if (value >= 100) {
    return 100;
  }
  return value;
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
}): Promise<T> {
  const abortController = new AbortController();
  const timeoutHandle = setTimeout(() => {
    abortController.abort();
  }, CLOUD_CODE_TIMEOUT_MS);

  try {
    const headers: Record<string, string> = {
      Authorization: `Bearer ${request.accessToken}`,
      "Content-Type": "application/json",
      "User-Agent": buildCloudCodeUserAgent(),
    };

    const response = await fetch(`${request.baseUrl}/${request.path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(request.body),
      signal: abortController.signal,
    });

    const text = await response.text();
    if (!response.ok) {
      const compact = text.replace(/\s+/g, " ").trim();
      throw new Error(`[cloudcode ${response.status}] ${compact.slice(0, CLOUD_CODE_ERROR_SNIPPET_CHARS)}`);
    }

    if (!text.trim()) {
      return {} as T;
    }
    return JSON.parse(text) as T;
  } finally {
    clearTimeout(timeoutHandle);
  }
}

function buildCloudCodeUserAgent(): string {
  const platform = process.platform === "win32" ? "windows" : process.platform;
  const arch = process.arch === "x64" ? "amd64" : process.arch === "ia32" ? "386" : process.arch;
  return `antigravity/${ANTIGRAVITY_IDE_VERSION} ${platform}/${arch}`;
}
