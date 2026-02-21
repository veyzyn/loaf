import fs from "node:fs";
import os from "node:os";
import path from "node:path";

type KeytarModule = {
  getPassword: (service: string, account: string) => Promise<string | null>;
  setPassword: (service: string, account: string, password: string) => Promise<void>;
  deletePassword: (service: string, account: string) => Promise<boolean>;
};

const LOAF_SECRET_SERVICE_NAME = "loaf";
const FALLBACK_SECRETS_FILE_PATH = path.join(os.homedir(), ".loaf", "secrets-fallback.json");
let keytarPromise: Promise<KeytarModule | null> | null = null;

export async function isSecureStoreAvailable(): Promise<boolean> {
  return Boolean(await getKeytar());
}

export async function getSecureValue(account: string): Promise<string> {
  const keytar = await getKeytar();
  if (keytar) {
    try {
      const value = await keytar.getPassword(LOAF_SECRET_SERVICE_NAME, account);
      const normalized = value?.trim() ?? "";
      if (normalized) {
        return normalized;
      }

      // If keytar becomes available after fallback use, migrate account lazily.
      const fallbackSecrets = readFallbackSecrets();
      const fallbackValue = fallbackSecrets[account];
      const normalizedFallback = typeof fallbackValue === "string" ? fallbackValue.trim() : "";
      if (!normalizedFallback) {
        return "";
      }
      await keytar.setPassword(LOAF_SECRET_SERVICE_NAME, account, normalizedFallback);
      delete fallbackSecrets[account];
      writeFallbackSecrets(fallbackSecrets);
      return normalizedFallback;
    } catch {
      return "";
    }
  }

  const fallback = readFallbackSecrets();
  const value = fallback[account];
  return typeof value === "string" ? value.trim() : "";
}

export async function setSecureValue(account: string, value: string): Promise<boolean> {
  const keytar = await getKeytar();
  const normalized = value.trim();
  if (!normalized) {
    return deleteSecureValue(account);
  }

  if (keytar) {
    try {
      await keytar.setPassword(LOAF_SECRET_SERVICE_NAME, account, normalized);
      const fallbackSecrets = readFallbackSecrets();
      if (account in fallbackSecrets) {
        delete fallbackSecrets[account];
        writeFallbackSecrets(fallbackSecrets);
      }
      return true;
    } catch {
      return false;
    }
  }

  try {
    const next = readFallbackSecrets();
    next[account] = normalized;
    writeFallbackSecrets(next);
    return true;
  } catch {
    return false;
  }
}

export async function deleteSecureValue(account: string): Promise<boolean> {
  const keytar = await getKeytar();
  if (keytar) {
    try {
      await keytar.deletePassword(LOAF_SECRET_SERVICE_NAME, account);
      const fallbackSecrets = readFallbackSecrets();
      if (account in fallbackSecrets) {
        delete fallbackSecrets[account];
        writeFallbackSecrets(fallbackSecrets);
      }
      return true;
    } catch {
      return false;
    }
  }

  try {
    const next = readFallbackSecrets();
    if (!(account in next)) {
      return true;
    }
    delete next[account];
    writeFallbackSecrets(next);
    return true;
  } catch {
    return false;
  }
}

async function getKeytar(): Promise<KeytarModule | null> {
  if (!keytarPromise) {
    keytarPromise = importKeytar();
  }
  return keytarPromise;
}

async function importKeytar(): Promise<KeytarModule | null> {
  try {
    const imported = (await import("keytar")) as { default?: unknown };
    const candidate = (imported.default ?? imported) as Partial<KeytarModule>;
    if (
      typeof candidate.getPassword === "function" &&
      typeof candidate.setPassword === "function" &&
      typeof candidate.deletePassword === "function"
    ) {
      return candidate as KeytarModule;
    }
  } catch {
    // secure store unavailable in this environment
  }
  return null;
}

function readFallbackSecrets(): Record<string, string> {
  try {
    if (!fs.existsSync(FALLBACK_SECRETS_FILE_PATH)) {
      return {};
    }

    const raw = fs.readFileSync(FALLBACK_SECRETS_FILE_PATH, "utf8");
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object") {
      return {};
    }

    const entries = Object.entries(parsed as Record<string, unknown>);
    const next: Record<string, string> = {};
    for (const [key, value] of entries) {
      if (typeof key !== "string" || typeof value !== "string") {
        continue;
      }
      const normalizedKey = key.trim();
      const normalizedValue = value.trim();
      if (!normalizedKey || !normalizedValue) {
        continue;
      }
      next[normalizedKey] = normalizedValue;
    }
    return next;
  } catch {
    return {};
  }
}

function writeFallbackSecrets(payload: Record<string, string>): void {
  const entries = Object.entries(payload).filter(([key, value]) => key.trim() && value.trim());
  const normalized = Object.fromEntries(entries);
  const fileDir = path.dirname(FALLBACK_SECRETS_FILE_PATH);
  const tmpPath = `${FALLBACK_SECRETS_FILE_PATH}.tmp`;

  fs.mkdirSync(fileDir, { recursive: true });
  fs.writeFileSync(tmpPath, `${JSON.stringify(normalized, null, 2)}\n`, "utf8");
  fs.renameSync(tmpPath, FALLBACK_SECRETS_FILE_PATH);
}
