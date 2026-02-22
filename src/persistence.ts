import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { type AuthProvider, type ThinkingLevel } from "./config.js";
import {
  SECRET_ACCOUNT_ANTIGRAVITY_OAUTH_TOKEN_INFO,
  SECRET_ACCOUNT_EXA_API_KEY,
  SECRET_ACCOUNT_OPENAI_CHATGPT_AUTH,
  SECRET_ACCOUNT_OPENROUTER_API_KEY,
} from "./secret-accounts.js";
import {
  deleteSecureValue,
  getSecureValue,
  setSecureValue,
} from "./secure-store.js";

export type LoafPersistedState = {
  version: 1;
  authProviders?: AuthProvider[];
  authProvider?: AuthProvider;
  selectedModel?: string;
  selectedThinking?: ThinkingLevel;
  // Legacy plaintext secret fields. Kept for migration only.
  openRouterApiKey?: string;
  exaApiKey?: string;
  selectedOpenRouterProvider?: string;
  onboardingCompleted?: boolean;
  inputHistory?: string[];
  updatedAt?: string;
};

export type LoafRuntimeSecrets = {
  openRouterApiKey: string;
  exaApiKey: string;
};

const CURRENT_LOAF_DATA_DIR = path.join(os.homedir(), ".loaf");
const LEGACY_LOAF_DATA_DIR = getLegacyLoafDataDir();
let migrationChecked = false;
const STATE_FILE_PATH = getStateFilePath();
const LEGACY_STATE_FILE_PATH = path.resolve(process.cwd(), ".loaf-state.json");
const MAX_HISTORY_ITEMS = 200;

export function loadPersistedState(): LoafPersistedState | null {
  const current = readStateFromPath(STATE_FILE_PATH);
  if (current) {
    return current;
  }

  const legacy = readStateFromPath(LEGACY_STATE_FILE_PATH);
  if (!legacy) {
    return null;
  }

  writeStateToPath(STATE_FILE_PATH, legacy);
  try {
    if (fs.existsSync(LEGACY_STATE_FILE_PATH)) {
      fs.unlinkSync(LEGACY_STATE_FILE_PATH);
    }
  } catch {
    // best-effort cleanup
  }
  return legacy;
}

export function savePersistedState(next: {
  authProviders: AuthProvider[];
  selectedModel: string;
  selectedThinking: ThinkingLevel;
  selectedOpenRouterProvider?: string;
  onboardingCompleted: boolean;
  inputHistory: string[];
}): void {
  const authProviders = dedupeAuthProviders(next.authProviders);
  const primaryAuthProvider = authProviders[0];

  const payload: LoafPersistedState = {
    version: 1,
    authProviders: authProviders.length > 0 ? authProviders : undefined,
    authProvider: primaryAuthProvider,
    selectedModel: next.selectedModel.trim(),
    selectedThinking: next.selectedThinking,
    selectedOpenRouterProvider:
      typeof next.selectedOpenRouterProvider === "string" && next.selectedOpenRouterProvider.trim()
        ? next.selectedOpenRouterProvider.trim()
        : undefined,
    onboardingCompleted: next.onboardingCompleted === true,
    inputHistory: next.inputHistory
      .filter((item) => typeof item === "string")
      .map((item) => item.trim())
      .filter(Boolean)
      .slice(-MAX_HISTORY_ITEMS),
    updatedAt: new Date().toISOString(),
  };

  writeStateToPath(STATE_FILE_PATH, payload);
}

function readStateFromPath(stateFilePath: string): LoafPersistedState | null {
  try {
    if (!fs.existsSync(stateFilePath)) {
      return null;
    }

    const raw = fs.readFileSync(stateFilePath, "utf8");
    const parsed = JSON.parse(raw) as Partial<LoafPersistedState>;
    if (!parsed || typeof parsed !== "object") {
      return null;
    }

    // Backward-compat: map legacy "vertex" persisted provider values to openrouter.
    const authProvider =
      parsed.authProvider === "openai"
        ? "openai"
        : parsed.authProvider === "antigravity"
          ? "antigravity"
        : parsed.authProvider === "openrouter" || parsed.authProvider === "vertex"
          ? "openrouter"
          : undefined;
    const parsedProviders = Array.isArray(parsed.authProviders)
      ? (parsed.authProviders as string[])
      : [];
    const authProviders = dedupeAuthProviders(
      parsedProviders.length > 0
        ? parsedProviders
            .map((provider) => (provider === "vertex" ? "openrouter" : provider))
            .filter(
              (provider): provider is AuthProvider =>
                provider === "openai" || provider === "openrouter" || provider === "antigravity",
            )
        : authProvider
          ? [authProvider]
          : [],
    );

    const selectedModel =
      typeof parsed.selectedModel === "string" && parsed.selectedModel.trim()
        ? parsed.selectedModel.trim()
        : undefined;

    const selectedThinking = isThinkingLevel(parsed.selectedThinking)
      ? parsed.selectedThinking
      : undefined;

    const inputHistory = Array.isArray(parsed.inputHistory)
      ? parsed.inputHistory
          .filter((item): item is string => typeof item === "string")
          .map((item) => item.trim())
          .filter(Boolean)
          .slice(-MAX_HISTORY_ITEMS)
      : [];

    return {
      version: 1,
      authProviders: authProviders.length > 0 ? authProviders : undefined,
      authProvider,
      selectedModel,
      selectedThinking,
      openRouterApiKey:
        typeof parsed.openRouterApiKey === "string" && parsed.openRouterApiKey.trim()
          ? parsed.openRouterApiKey.trim()
          : undefined,
      exaApiKey:
        typeof parsed.exaApiKey === "string" && parsed.exaApiKey.trim()
          ? parsed.exaApiKey.trim()
          : undefined,
      selectedOpenRouterProvider:
        typeof parsed.selectedOpenRouterProvider === "string" && parsed.selectedOpenRouterProvider.trim()
          ? parsed.selectedOpenRouterProvider.trim()
          : undefined,
      onboardingCompleted: parsed.onboardingCompleted === true,
      inputHistory,
      updatedAt: typeof parsed.updatedAt === "string" ? parsed.updatedAt : undefined,
    };
  } catch {
    return null;
  }
}

function dedupeAuthProviders(providers: AuthProvider[]): AuthProvider[] {
  const ordered: AuthProvider[] = [];
  for (const provider of providers) {
    if (
      (provider !== "openai" && provider !== "openrouter" && provider !== "antigravity") ||
      ordered.includes(provider)
    ) {
      continue;
    }
    ordered.push(provider);
  }
  return ordered;
}

function writeStateToPath(stateFilePath: string, payload: LoafPersistedState): void {
  const stateDir = path.dirname(stateFilePath);
  const tmpPath = `${stateFilePath}.tmp`;
  try {
    fs.mkdirSync(stateDir, { recursive: true });
    fs.writeFileSync(tmpPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
    fs.renameSync(tmpPath, stateFilePath);
  } catch {
    try {
      if (fs.existsSync(tmpPath)) {
        fs.unlinkSync(tmpPath);
      }
    } catch {
      // ignore cleanup failures
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

function getStateFilePath(): string {
  return path.join(getLoafDataDir(), "state.json");
}

export function getLoafDataDir(): string {
  ensureLoafDataDirMigration();
  return CURRENT_LOAF_DATA_DIR;
}

export function clearPersistedConfig(): void {
  const dataDir = getLoafDataDir();
  const targetFiles = [
    path.join(dataDir, "state.json"),
    path.join(dataDir, "auth.json"),
    path.join(dataDir, "models-cache.json"),
  ];

  for (const filePath of targetFiles) {
    try {
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    } catch {
      // best-effort cleanup
    }
  }

  try {
    if (fs.existsSync(LEGACY_STATE_FILE_PATH)) {
      fs.unlinkSync(LEGACY_STATE_FILE_PATH);
    }
  } catch {
    // best-effort cleanup
  }

  void clearPersistedSecrets();
}

export async function loadPersistedRuntimeSecrets(
  legacyState: LoafPersistedState | null,
): Promise<LoafRuntimeSecrets> {
  const legacyOpenRouterApiKey = legacyState?.openRouterApiKey?.trim() ?? "";
  const legacyExaApiKey = legacyState?.exaApiKey?.trim() ?? "";

  let openRouterApiKey = await getSecureValue(SECRET_ACCOUNT_OPENROUTER_API_KEY);
  let exaApiKey = await getSecureValue(SECRET_ACCOUNT_EXA_API_KEY);

  if (!openRouterApiKey && legacyOpenRouterApiKey) {
    const stored = await setSecureValue(SECRET_ACCOUNT_OPENROUTER_API_KEY, legacyOpenRouterApiKey);
    if (stored) {
      openRouterApiKey = legacyOpenRouterApiKey;
    }
  }
  if (!exaApiKey && legacyExaApiKey) {
    const stored = await setSecureValue(SECRET_ACCOUNT_EXA_API_KEY, legacyExaApiKey);
    if (stored) {
      exaApiKey = legacyExaApiKey;
    }
  }

  const canScrubOpenRouter = !legacyOpenRouterApiKey || Boolean(openRouterApiKey);
  const canScrubExa = !legacyExaApiKey || Boolean(exaApiKey);
  if (canScrubOpenRouter && canScrubExa && (legacyOpenRouterApiKey || legacyExaApiKey)) {
    scrubLegacySecretsFromStateFiles();
  }

  return {
    openRouterApiKey,
    exaApiKey,
  };
}

export async function persistRuntimeSecrets(next: LoafRuntimeSecrets): Promise<void> {
  const openRouterApiKey = next.openRouterApiKey.trim();
  const exaApiKey = next.exaApiKey.trim();

  if (openRouterApiKey) {
    await setSecureValue(SECRET_ACCOUNT_OPENROUTER_API_KEY, openRouterApiKey);
  } else {
    await deleteSecureValue(SECRET_ACCOUNT_OPENROUTER_API_KEY);
  }

  if (exaApiKey) {
    await setSecureValue(SECRET_ACCOUNT_EXA_API_KEY, exaApiKey);
  } else {
    await deleteSecureValue(SECRET_ACCOUNT_EXA_API_KEY);
  }

  scrubLegacySecretsFromStateFiles();
}

function getLegacyStateBaseDir(): string {
  if (process.platform === "win32") {
    const appData = process.env.APPDATA?.trim();
    if (appData) {
      return appData;
    }

    const localAppData = process.env.LOCALAPPDATA?.trim();
    if (localAppData) {
      return localAppData;
    }
  }

  const xdgStateHome = process.env.XDG_STATE_HOME?.trim();
  if (xdgStateHome) {
    return xdgStateHome;
  }

  const home = os.homedir();
  if (process.platform === "darwin") {
    return path.join(home, "Library", "Application Support");
  }

  return path.join(home, ".local", "state");
}

function getLegacyLoafDataDir(): string {
  return path.join(getLegacyStateBaseDir(), "loaf");
}

function ensureLoafDataDirMigration(): void {
  if (migrationChecked) {
    return;
  }
  migrationChecked = true;

  try {
    if (fs.existsSync(CURRENT_LOAF_DATA_DIR)) {
      return;
    }
    if (!fs.existsSync(LEGACY_LOAF_DATA_DIR)) {
      return;
    }

    try {
      fs.renameSync(LEGACY_LOAF_DATA_DIR, CURRENT_LOAF_DATA_DIR);
      return;
    } catch {
      // fall back to copy-on-failure, e.g. cross-device rename
    }

    fs.mkdirSync(CURRENT_LOAF_DATA_DIR, { recursive: true });
    fs.cpSync(LEGACY_LOAF_DATA_DIR, CURRENT_LOAF_DATA_DIR, { recursive: true });
    try {
      fs.rmSync(LEGACY_LOAF_DATA_DIR, { recursive: true, force: true });
    } catch {
      // best-effort cleanup
    }
  } catch {
    // best-effort migration
  }
}

function scrubLegacySecretsFromStateFiles(): void {
  scrubLegacySecretsFromStateFile(STATE_FILE_PATH);
  scrubLegacySecretsFromStateFile(LEGACY_STATE_FILE_PATH);
}

function scrubLegacySecretsFromStateFile(stateFilePath: string): void {
  const state = readStateFromPath(stateFilePath);
  if (!state) {
    return;
  }
  if (!state.openRouterApiKey && !state.exaApiKey) {
    return;
  }

  const sanitized: LoafPersistedState = {
    ...state,
    openRouterApiKey: undefined,
    exaApiKey: undefined,
  };
  writeStateToPath(stateFilePath, sanitized);
}

async function clearPersistedSecrets(): Promise<void> {
  await Promise.all([
    deleteSecureValue(SECRET_ACCOUNT_ANTIGRAVITY_OAUTH_TOKEN_INFO),
    deleteSecureValue(SECRET_ACCOUNT_OPENROUTER_API_KEY),
    deleteSecureValue(SECRET_ACCOUNT_EXA_API_KEY),
    deleteSecureValue(SECRET_ACCOUNT_OPENAI_CHATGPT_AUTH),
  ]);
}
