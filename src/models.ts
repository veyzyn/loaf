import fs from "node:fs";
import path from "node:path";
import type { AuthProvider, ThinkingLevel } from "./config.js";
import { listOpenAiCatalogModels, type OpenAiCatalogModel } from "./openai.js";
import { listOpenRouterModels } from "./openrouter.js";
import { getLoafDataDir } from "./persistence.js";

export type ModelOption = {
  id: string;
  provider: AuthProvider;
  displayProvider?: string;
  label: string;
  description: string;
  supportedThinkingLevels?: ThinkingLevel[];
  defaultThinkingLevel?: ThinkingLevel;
  routingProviders?: string[];
  contextWindowTokens?: number;
};

export type ModelDiscoverySource = "remote" | "cache" | "fallback";

export type ModelDiscoveryResult = {
  models: ModelOption[];
  source: ModelDiscoverySource;
};

type CachedProviderModels = {
  fetchedAt: string;
  models: Array<{
    id: string;
    label: string;
    description: string;
    supportedThinkingLevels?: ThinkingLevel[];
    defaultThinkingLevel?: ThinkingLevel;
    routingProviders?: string[];
    contextWindowTokens?: number;
  }>;
};

type ModelsCacheFile = {
  version: 1;
  openai?: CachedProviderModels;
  openrouter?: CachedProviderModels;
  antigravity?: CachedProviderModels;
};

const MODEL_CACHE_FILE_PATH = path.join(getLoafDataDir(), "models-cache.json");
const MODEL_CACHE_TTL_MS = 5 * 60 * 1_000;
const MAX_OPENAI_MODELS = 120;

const OPENAI_REASONING_GENERAL: ThinkingLevel[] = ["MINIMAL", "LOW", "MEDIUM", "HIGH"];
const OPENAI_REASONING_CODEX: ThinkingLevel[] = ["LOW", "MEDIUM", "HIGH", "XHIGH"];
const OPENAI_REASONING_CODEX_MINI: ThinkingLevel[] = ["MEDIUM", "HIGH"];
const OPENROUTER_REASONING_ENABLED: ThinkingLevel[] = ["OFF", "MINIMAL", "LOW", "MEDIUM", "HIGH"];
const OPENROUTER_REASONING_DISABLED: ThinkingLevel[] = ["OFF"];

const DEFAULT_OPENAI_MODEL_OPTIONS: ModelOption[] = [];

const DEFAULT_OPENROUTER_MODEL_OPTIONS: ModelOption[] = [];

const DEFAULT_ANTIGRAVITY_MODEL_OPTIONS: ModelOption[] = [];

export function getDefaultModelOptionsForProvider(provider: AuthProvider): ModelOption[] {
  const defaults =
    provider === "openai"
      ? DEFAULT_OPENAI_MODEL_OPTIONS
      : provider === "openrouter"
        ? DEFAULT_OPENROUTER_MODEL_OPTIONS
        : DEFAULT_ANTIGRAVITY_MODEL_OPTIONS;
  return defaults.map((item) => ({ ...item }));
}

export async function discoverOpenAiModelOptions(request: {
  accessToken: string;
  chatgptAccountId: string | null;
}): Promise<ModelDiscoveryResult> {
  // Always prefer fresh remote data for OpenAI models to quickly surface catalog updates.
  try {
    const catalog = await listOpenAiCatalogModels(request.accessToken, request.chatgptAccountId);
    const remote = normalizeOpenAiModelOptions(catalog);
    if (remote.length > 0) {
      writeCachedProviderModels("openai", remote);
      return {
        models: mergeWithFallbacks("openai", remote),
        source: "remote",
      };
    }
  } catch {
    // fall back to cache/default options
  }

  const freshCached = readCachedProviderModels("openai", true);
  if (freshCached.length > 0) {
    return {
      models: mergeWithFallbacks("openai", freshCached),
      source: "cache",
    };
  }

  const staleCached = readCachedProviderModels("openai", false);
  if (staleCached.length > 0) {
    return {
      models: mergeWithFallbacks("openai", staleCached),
      source: "cache",
    };
  }

  return {
    models: getDefaultModelOptionsForProvider("openai"),
    source: "fallback",
  };
}

export async function discoverOpenRouterModelOptions(request: {
  apiKey: string;
}): Promise<ModelDiscoveryResult> {
  try {
    const discovered = await listOpenRouterModels(request.apiKey);
    const remote = normalizeOpenRouterModelOptions(discovered);
    if (remote.length > 0) {
      writeCachedProviderModels("openrouter", remote);
      return {
        models: mergeWithFallbacks("openrouter", remote),
        source: "remote",
      };
    }
  } catch {
    // fall back to cached/default options
  }

  const freshCached = readCachedProviderModels("openrouter", true);
  if (freshCached.length > 0) {
    return {
      models: mergeWithFallbacks("openrouter", freshCached),
      source: "cache",
    };
  }

  const staleCached = readCachedProviderModels("openrouter", false);
  if (staleCached.length > 0) {
    return {
      models: mergeWithFallbacks("openrouter", staleCached),
      source: "cache",
    };
  }

  return {
    models: getDefaultModelOptionsForProvider("openrouter"),
    source: "fallback",
  };
}

export function modelIdToLabel(modelId: string): string {
  const slug = modelIdToSlug(modelId);
  return slug.replace(/[-_]+/g, " ");
}

export function modelIdToSlug(modelId: string): string {
  const trimmed = modelId.trim();
  if (!trimmed) {
    return "";
  }

  const modelsMatch = trimmed.match(/(?:^|\/)models\/([^/]+)$/i);
  if (modelsMatch?.[1]) {
    return modelsMatch[1];
  }

  const slashIndex = trimmed.lastIndexOf("/");
  if (slashIndex >= 0) {
    return trimmed.slice(slashIndex + 1);
  }

  return trimmed;
}

function normalizeOpenAiModelOptions(catalog: OpenAiCatalogModel[]): ModelOption[] {
  const byId = new Map<string, ModelOption>();
  for (const item of catalog) {
    const id = item.id.trim();
    if (!id || !isLikelyOpenAiTextModel(id)) {
      continue;
    }

    const slug = modelIdToSlug(id);
    const supportedThinkingLevels = normalizeOpenAiReasoningLevels(item, slug);
    const defaultThinkingLevel = normalizeDefaultThinkingLevel(
      item.defaultReasoningLevel,
      supportedThinkingLevels,
    );
    byId.set(id, {
      id,
      provider: "openai",
      label: item.label.trim() || modelIdToLabel(id),
      description: item.description.trim() || "server-advertised openai model",
      supportedThinkingLevels,
      defaultThinkingLevel,
      contextWindowTokens: normalizeContextWindowTokens(item.contextWindowTokens),
    });
  }

  const rows = Array.from(byId.values());
  rows.sort((a, b) => {
    const leftPriority = catalog.find((item) => item.id === a.id)?.priority;
    const rightPriority = catalog.find((item) => item.id === b.id)?.priority;
    if (typeof leftPriority === "number" && typeof rightPriority === "number" && leftPriority !== rightPriority) {
      return leftPriority - rightPriority;
    }
    return compareOpenAiModels(a.id, b.id);
  });
  return rows.slice(0, MAX_OPENAI_MODELS);
}

function normalizeOpenRouterModelOptions(
  discovered: Array<{
    id: string;
    label: string;
    description: string;
    supportedParameters: string[];
    providerTags: string[];
    contextWindowTokens?: number;
  }>,
): ModelOption[] {
  const byId = new Map<string, ModelOption>();
  for (const item of discovered) {
    const id = item.id.trim();
    if (!id) {
      continue;
    }
    const slug = modelIdToSlug(id);
    const normalizedParams = item.supportedParameters.map((param) => param.toLowerCase());
    const supportsReasoning = normalizedParams.some((param) => param.includes("reasoning"));
    const supportedThinkingLevels = supportsReasoning
      ? OPENROUTER_REASONING_ENABLED
      : OPENROUTER_REASONING_DISABLED;
    const defaultThinkingLevel: ThinkingLevel = supportsReasoning ? "MEDIUM" : "OFF";
    byId.set(id, {
      id,
      provider: "openrouter",
      label: item.label.trim() || modelIdToLabel(id),
      description: item.description.trim() || "server-advertised openrouter model",
      supportedThinkingLevels,
      defaultThinkingLevel,
      routingProviders: dedupeStringArray(item.providerTags),
      contextWindowTokens: normalizeContextWindowTokens(item.contextWindowTokens),
    });
  }

  return Array.from(byId.values())
    .sort((a, b) => a.label.localeCompare(b.label));
}

function normalizeOpenAiReasoningLevels(model: OpenAiCatalogModel, slug: string): ThinkingLevel[] {
  const fromServer = (model.supportedReasoningLevels ?? [])
    .map((item) => mapOpenAiReasoningEffort(item.effort))
    .filter((value): value is ThinkingLevel => value !== undefined);
  const dedupedServer = dedupeThinkingLevels(fromServer);
  if (dedupedServer.length > 0) {
    return dedupedServer;
  }

  const normalizedSlug = slug.toLowerCase();
  if (normalizedSlug.includes("codex-mini")) {
    return OPENAI_REASONING_CODEX_MINI;
  }
  if (normalizedSlug.includes("codex")) {
    return OPENAI_REASONING_CODEX;
  }
  if (normalizedSlug.startsWith("gpt-5")) {
    return OPENAI_REASONING_GENERAL;
  }
  return ["OFF", ...OPENAI_REASONING_GENERAL];
}

function mapOpenAiReasoningEffort(effort: string): ThinkingLevel | undefined {
  const normalized = effort.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }

  if (normalized === "none" || normalized === "off") {
    return "OFF";
  }
  if (normalized === "minimal") {
    return "MINIMAL";
  }
  if (normalized === "low") {
    return "LOW";
  }
  if (normalized === "medium") {
    return "MEDIUM";
  }
  if (normalized === "high") {
    return "HIGH";
  }
  if (normalized === "xhigh" || normalized === "extra_high" || normalized === "extra-high") {
    return "XHIGH";
  }
  return undefined;
}

function normalizeDefaultThinkingLevel(
  value: string | undefined,
  supportedThinkingLevels: ThinkingLevel[],
): ThinkingLevel | undefined {
  const mapped = value ? mapOpenAiReasoningEffort(value) : undefined;
  if (mapped && supportedThinkingLevels.includes(mapped)) {
    return mapped;
  }
  return supportedThinkingLevels[Math.floor(Math.max(0, (supportedThinkingLevels.length - 1) / 2))];
}

function dedupeThinkingLevels(levels: ThinkingLevel[]): ThinkingLevel[] {
  const ordered: ThinkingLevel[] = [];
  for (const level of levels) {
    if (!ordered.includes(level)) {
      ordered.push(level);
    }
  }
  return ordered;
}

function isLikelyOpenAiTextModel(modelId: string): boolean {
  const normalized = modelId.trim().toLowerCase();
  if (!normalized) {
    return false;
  }

  if (
    normalized.startsWith("gpt-") ||
    normalized.startsWith("o1") ||
    normalized.startsWith("o3") ||
    normalized.startsWith("o4") ||
    normalized.includes("codex")
  ) {
    return true;
  }

  if (
    normalized.includes("embedding") ||
    normalized.includes("whisper") ||
    normalized.includes("moderation") ||
    normalized.includes("tts") ||
    normalized.includes("dall")
  ) {
    return false;
  }

  return false;
}

function compareOpenAiModels(leftId: string, rightId: string): number {
  const leftRank = openAiModelRank(leftId);
  const rightRank = openAiModelRank(rightId);
  if (leftRank !== rightRank) {
    return leftRank - rightRank;
  }
  return leftId.localeCompare(rightId);
}

function openAiModelRank(modelId: string): number {
  const normalized = modelId.trim().toLowerCase();
  if (normalized.includes("codex")) {
    return 0;
  }
  if (normalized === "gpt-5" || normalized.startsWith("gpt-5-")) {
    return 1;
  }
  if (normalized === "gpt-4.1" || normalized.startsWith("gpt-4.1-")) {
    return 2;
  }
  if (normalized === "gpt-4o" || normalized.startsWith("gpt-4o-")) {
    return 3;
  }
  if (normalized.startsWith("o")) {
    return 4;
  }
  return 10;
}

function mergeWithFallbacks(provider: AuthProvider, discovered: ModelOption[]): ModelOption[] {
  const merged = new Map<string, ModelOption>();
  for (const option of discovered) {
    merged.set(option.id, option);
  }
  for (const fallback of getDefaultModelOptionsForProvider(provider)) {
    const existing = merged.get(fallback.id);
    if (!existing) {
      merged.set(fallback.id, fallback);
      continue;
    }
    merged.set(fallback.id, {
      ...existing,
      supportedThinkingLevels:
        existing.supportedThinkingLevels && existing.supportedThinkingLevels.length > 0
          ? existing.supportedThinkingLevels
          : fallback.supportedThinkingLevels,
      defaultThinkingLevel: existing.defaultThinkingLevel ?? fallback.defaultThinkingLevel,
      routingProviders:
        existing.routingProviders && existing.routingProviders.length > 0
          ? existing.routingProviders
          : fallback.routingProviders,
      contextWindowTokens: existing.contextWindowTokens ?? fallback.contextWindowTokens,
    });
  }

  return Array.from(merged.values());
}

function readCachedProviderModels(provider: AuthProvider, requireFresh: boolean): ModelOption[] {
  const cache = readModelsCacheFile();
  if (!cache) {
    return [];
  }

  const entry =
    provider === "openai"
      ? cache.openai
      : provider === "openrouter"
        ? cache.openrouter
        : cache.antigravity;
  if (!entry || !Array.isArray(entry.models) || !entry.fetchedAt) {
    return [];
  }

  const fetchedAtMs = Date.parse(entry.fetchedAt);
  if (!Number.isFinite(fetchedAtMs)) {
    return [];
  }

  if (requireFresh && Date.now() - fetchedAtMs > MODEL_CACHE_TTL_MS) {
    return [];
  }

  const models: ModelOption[] = [];
  for (const model of entry.models) {
    if (!model || typeof model !== "object") {
      continue;
    }
    const id = typeof model.id === "string" ? model.id.trim() : "";
    if (!id) {
      continue;
    }
    const labelValue = typeof model.label === "string" ? model.label.trim() : "";
    const descriptionValue = typeof model.description === "string" ? model.description.trim() : "";
    const supportedThinkingLevels = Array.isArray(model.supportedThinkingLevels)
      ? dedupeThinkingLevels(
          model.supportedThinkingLevels.filter((item): item is ThinkingLevel => isThinkingLevel(item)),
        )
      : undefined;
    const defaultThinkingLevel =
      typeof model.defaultThinkingLevel === "string" && isThinkingLevel(model.defaultThinkingLevel)
        ? model.defaultThinkingLevel
        : undefined;
    const routingProviders = dedupeStringArray(
      Array.isArray(model.routingProviders)
        ? model.routingProviders.filter((item): item is string => typeof item === "string")
        : [],
    );
    const contextWindowTokens = normalizeContextWindowTokens(model.contextWindowTokens);
    models.push({
      id,
      provider,
      label: labelValue || modelIdToLabel(id),
      description: descriptionValue || "server-advertised model",
      supportedThinkingLevels: supportedThinkingLevels?.length ? supportedThinkingLevels : undefined,
      defaultThinkingLevel:
        defaultThinkingLevel && supportedThinkingLevels?.includes(defaultThinkingLevel)
          ? defaultThinkingLevel
          : supportedThinkingLevels?.[Math.floor(Math.max(0, (supportedThinkingLevels.length - 1) / 2))],
      routingProviders: routingProviders.length > 0 ? routingProviders : undefined,
      contextWindowTokens,
    });
  }

  return models;
}

function writeCachedProviderModels(provider: AuthProvider, models: ModelOption[]): void {
  const existing = readModelsCacheFile() ?? { version: 1 as const };
  const sanitizedModels = models
    .map((model) => ({
      id: model.id.trim(),
      label: model.label.trim(),
      description: model.description.trim(),
      supportedThinkingLevels: dedupeThinkingLevels(
        (model.supportedThinkingLevels ?? []).filter((item): item is ThinkingLevel => isThinkingLevel(item)),
      ),
      defaultThinkingLevel:
        model.defaultThinkingLevel && isThinkingLevel(model.defaultThinkingLevel)
          ? model.defaultThinkingLevel
          : undefined,
      routingProviders: dedupeStringArray(
        (model.routingProviders ?? [])
          .filter((item): item is string => typeof item === "string")
          .map((item) => item.trim().toLowerCase())
          .filter(Boolean),
      ),
      contextWindowTokens: normalizeContextWindowTokens(model.contextWindowTokens),
    }))
    .filter((model) => model.id)
    .map((model) => ({
      ...model,
      supportedThinkingLevels: model.supportedThinkingLevels.length
        ? model.supportedThinkingLevels
        : undefined,
      defaultThinkingLevel:
        model.defaultThinkingLevel && model.supportedThinkingLevels?.includes(model.defaultThinkingLevel)
          ? model.defaultThinkingLevel
          : undefined,
      routingProviders: model.routingProviders.length > 0 ? model.routingProviders : undefined,
      contextWindowTokens: normalizeContextWindowTokens(model.contextWindowTokens),
    }));
  const limitedModels = provider === "openai" ? sanitizedModels.slice(0, MAX_OPENAI_MODELS) : sanitizedModels;

  const nextEntry: CachedProviderModels = {
    fetchedAt: new Date().toISOString(),
    models: limitedModels,
  };

  const nextCache: ModelsCacheFile = {
    version: 1,
    openai: existing.openai,
    openrouter: existing.openrouter,
    antigravity: existing.antigravity,
  };
  if (provider === "openai") {
    nextCache.openai = nextEntry;
  } else if (provider === "openrouter") {
    nextCache.openrouter = nextEntry;
  } else {
    nextCache.antigravity = nextEntry;
  }

  writeModelsCacheFile(nextCache);
}

function readModelsCacheFile(): ModelsCacheFile | null {
  try {
    if (!fs.existsSync(MODEL_CACHE_FILE_PATH)) {
      return null;
    }
    const raw = fs.readFileSync(MODEL_CACHE_FILE_PATH, "utf8");
    const parsed = JSON.parse(raw) as Partial<ModelsCacheFile>;
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    if (parsed.version !== 1) {
      return null;
    }
    return parsed as ModelsCacheFile;
  } catch {
    return null;
  }
}

function writeModelsCacheFile(payload: ModelsCacheFile): void {
  const dir = path.dirname(MODEL_CACHE_FILE_PATH);
  const tmpPath = `${MODEL_CACHE_FILE_PATH}.tmp`;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(tmpPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
    fs.renameSync(tmpPath, MODEL_CACHE_FILE_PATH);
  } catch {
    try {
      if (fs.existsSync(tmpPath)) {
        fs.unlinkSync(tmpPath);
      }
    } catch {
      // ignore cleanup failure
    }
  }
}

function isThinkingLevel(value: unknown): value is ThinkingLevel {
  return (
    value === "OFF" ||
    value === "MINIMAL" ||
    value === "LOW" ||
    value === "MEDIUM" ||
    value === "HIGH" ||
    value === "XHIGH"
  );
}

function dedupeStringArray(values: string[]): string[] {
  const ordered: string[] = [];
  for (const rawValue of values) {
    const value = rawValue.trim();
    if (!value || ordered.includes(value)) {
      continue;
    }
    ordered.push(value);
  }
  return ordered;
}

function normalizeContextWindowTokens(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  const floored = Math.floor(value);
  return floored > 0 ? floored : undefined;
}
