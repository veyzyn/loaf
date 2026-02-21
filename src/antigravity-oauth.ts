import { randomUUID } from "node:crypto";
import http from "node:http";
import { spawn } from "node:child_process";
import { OAuth2Client } from "google-auth-library";
import { SECRET_ACCOUNT_ANTIGRAVITY_OAUTH_TOKEN_INFO } from "./secret-accounts.js";
import { getSecureValue, setSecureValue } from "./secure-store.js";

const ANTIGRAVITY_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
const ANTIGRAVITY_CLIENT_SECRET = "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";
const ANTIGRAVITY_PROFILE_URL = "https://www.googleapis.com/oauth2/v2/userinfo";
const DEFAULT_CALLBACK_PATH = "/oauth-callback";
const DEFAULT_LOCAL_SUCCESS_PATH = "/oauth-success";
const ANTIGRAVITY_SCOPES = [
  "https://www.googleapis.com/auth/cloud-platform",
  "https://www.googleapis.com/auth/userinfo.email",
  "https://www.googleapis.com/auth/userinfo.profile",
  "https://www.googleapis.com/auth/cclog",
  "https://www.googleapis.com/auth/experimentsandconfigs",
] as const;
const TOKEN_REFRESH_WINDOW_SECONDS = 5 * 60;
const OAUTH_TIMEOUT_MS = 10 * 60 * 1000;
const PROFILE_FETCH_TIMEOUT_MS = 5 * 1000;
const TOKEN_EXCHANGE_TIMEOUT_MS = 30 * 1000;
const SERVER_CLOSE_TIMEOUT_MS = 1_000;

type StoredAntigravityOauthSecrets = {
  version: 1;
  accessToken: string;
  refreshToken: string;
  expiryDateSeconds: number;
  tokenType: string;
};

export type AntigravityOauthTokenInfo = {
  accessToken: string;
  refreshToken: string;
  expiryDateSeconds: number;
  tokenType: string;
};

export type AntigravityOauthProfile = {
  email: string;
  name: string;
  picture?: string;
};

export type AntigravityOauthLoginResult = {
  authUrl: string;
  tokenInfo: AntigravityOauthTokenInfo;
  profile: AntigravityOauthProfile | null;
};

export async function loadPersistedAntigravityOauthTokenInfo(): Promise<AntigravityOauthTokenInfo | null> {
  const raw = await getSecureValue(SECRET_ACCOUNT_ANTIGRAVITY_OAUTH_TOKEN_INFO);
  if (!raw) {
    return null;
  }

  let parsed: Partial<StoredAntigravityOauthSecrets>;
  try {
    parsed = JSON.parse(raw) as Partial<StoredAntigravityOauthSecrets>;
  } catch {
    return null;
  }

  if (!parsed || typeof parsed !== "object" || parsed.version !== 1) {
    return null;
  }

  const normalized = normalizeTokenInfo(parsed);
  if (!normalized) {
    return null;
  }

  return refreshAntigravityOauthTokenIfNeeded(normalized);
}

export async function runAntigravityOauthLogin(options?: {
  openBrowser?: boolean;
  timeoutMs?: number;
}): Promise<AntigravityOauthLoginResult> {
  const { server, port } = await listenOnPort(0);
  const redirectUri = `http://localhost:${port}${DEFAULT_CALLBACK_PATH}`;
  const callbackPath = DEFAULT_CALLBACK_PATH;
  const successPath = resolveLocalSuccessPath(callbackPath);
  const oauthClient = new OAuth2Client(ANTIGRAVITY_CLIENT_ID, ANTIGRAVITY_CLIENT_SECRET, redirectUri);
  const state = randomUUID();

  const authUrl = oauthClient.generateAuthUrl({
    access_type: "offline",
    scope: [...ANTIGRAVITY_SCOPES],
    state,
    prompt: "consent",
  });

  if (options?.openBrowser !== false) {
    openExternalUrl(authUrl);
  }

  try {
    const tokenInfo = await waitForOauthCallback({
      server,
      oauthClient,
      timeoutMs: options?.timeoutMs ?? OAUTH_TIMEOUT_MS,
      callbackPath,
      successPath,
      expectedState: state,
    });
    await persistAntigravityOauthTokenInfo(tokenInfo);
    const profile = await fetchAntigravityProfileData(tokenInfo.accessToken, PROFILE_FETCH_TIMEOUT_MS);
    return {
      authUrl,
      tokenInfo,
      profile,
    };
  } finally {
    await closeServer(server);
  }
}

export async function refreshAntigravityOauthTokenIfNeeded(
  tokenInfo: AntigravityOauthTokenInfo,
): Promise<AntigravityOauthTokenInfo | null> {
  if (!tokenInfo.accessToken.trim()) {
    return null;
  }

  const nowSeconds = Date.now() / 1000;
  if (tokenInfo.expiryDateSeconds - nowSeconds >= TOKEN_REFRESH_WINDOW_SECONDS) {
    return tokenInfo;
  }

  const oauthClient = new OAuth2Client(ANTIGRAVITY_CLIENT_ID, ANTIGRAVITY_CLIENT_SECRET);
  oauthClient.setCredentials({
    refresh_token: tokenInfo.refreshToken,
    access_token: tokenInfo.accessToken,
    expiry_date: tokenInfo.expiryDateSeconds * 1000,
    token_type: tokenInfo.tokenType,
  });

  try {
    const refreshed = await oauthClient.refreshAccessToken();
    const next: AntigravityOauthTokenInfo = {
      accessToken: refreshed.credentials.access_token ?? "",
      refreshToken: refreshed.credentials.refresh_token ?? "",
      expiryDateSeconds: Math.floor((refreshed.credentials.expiry_date ?? 0) / 1000),
      tokenType: refreshed.credentials.token_type ?? "",
    };
    if (!next.accessToken.trim()) {
      return null;
    }

    await persistAntigravityOauthTokenInfo(next);
    return next;
  } catch {
    return null;
  }
}

export async function fetchAntigravityProfileData(
  accessToken: string,
  timeoutMs = PROFILE_FETCH_TIMEOUT_MS,
): Promise<AntigravityOauthProfile | null> {
  const token = accessToken.trim();
  if (!token) {
    return null;
  }

  const abortController = new AbortController();
  const timeoutHandle = setTimeout(() => {
    abortController.abort();
  }, timeoutMs);

  try {
    const response = await fetch(ANTIGRAVITY_PROFILE_URL, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${token}`,
      },
      signal: abortController.signal,
    });
    if (!response.ok) {
      return null;
    }

    const parsed = (await response.json()) as Partial<{
      email: string;
      name: string;
      picture: string;
    }>;
    const email = parsed.email?.trim() ?? "";
    const name = parsed.name?.trim() ?? "";
    if (!email || !name) {
      return null;
    }

    return {
      email,
      name,
      picture: parsed.picture?.trim() || undefined,
    };
  } catch {
    return null;
  } finally {
    clearTimeout(timeoutHandle);
  }
}

function normalizeTokenInfo(value: Partial<StoredAntigravityOauthSecrets>): AntigravityOauthTokenInfo | null {
  const accessToken = typeof value.accessToken === "string" ? value.accessToken.trim() : "";
  if (!accessToken) {
    return null;
  }

  const refreshToken = typeof value.refreshToken === "string" ? value.refreshToken.trim() : "";
  const expiryDateSecondsRaw = value.expiryDateSeconds;
  const expiryDateSeconds =
    typeof expiryDateSecondsRaw === "number" && Number.isFinite(expiryDateSecondsRaw)
      ? Math.floor(expiryDateSecondsRaw)
      : 0;
  const tokenType = typeof value.tokenType === "string" ? value.tokenType.trim() : "";

  return {
    accessToken,
    refreshToken,
    expiryDateSeconds,
    tokenType,
  };
}

async function persistAntigravityOauthTokenInfo(tokenInfo: AntigravityOauthTokenInfo): Promise<boolean> {
  const payload: StoredAntigravityOauthSecrets = {
    version: 1,
    accessToken: tokenInfo.accessToken.trim(),
    refreshToken: tokenInfo.refreshToken.trim(),
    expiryDateSeconds: Math.floor(tokenInfo.expiryDateSeconds),
    tokenType: tokenInfo.tokenType.trim(),
  };
  if (!payload.accessToken) {
    return false;
  }
  return setSecureValue(SECRET_ACCOUNT_ANTIGRAVITY_OAUTH_TOKEN_INFO, JSON.stringify(payload));
}

async function waitForOauthCallback(options: {
  server: http.Server;
  oauthClient: OAuth2Client;
  timeoutMs: number;
  callbackPath: string;
  successPath: string;
  expectedState: string;
}): Promise<AntigravityOauthTokenInfo> {
  const { server, oauthClient, timeoutMs, callbackPath, successPath, expectedState } = options;

  return new Promise<AntigravityOauthTokenInfo>((resolve, reject) => {
    let settled = false;
    let timeoutHandle: NodeJS.Timeout | null = null;

    const finish = (outcome: { ok: true; value: AntigravityOauthTokenInfo } | { ok: false; error: unknown }) => {
      if (settled) {
        return;
      }
      settled = true;
      if (timeoutHandle) {
        clearTimeout(timeoutHandle);
        timeoutHandle = null;
      }
      server.removeListener("request", onRequest);
      server.removeListener("error", onError);
      if (outcome.ok) {
        resolve(outcome.value);
        return;
      }
      reject(outcome.error);
    };

    const onError = (error: unknown) => {
      finish({ ok: false, error });
    };

    const onRequest = (req: http.IncomingMessage, res: http.ServerResponse) => {
      void (async () => {
        try {
          // Match Antigravity's permissive localhost callback handler behavior.
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
          res.setHeader("Access-Control-Allow-Headers", "Content-Type");

          if (req.method === "OPTIONS") {
            res.statusCode = 200;
            res.end();
            return;
          }

          const requestUrl = new URL(req.url ?? "/", "http://localhost");
          const isCallbackPath = requestUrl.pathname === callbackPath;
          const isSuccessPath = requestUrl.pathname === successPath;
          if (!isCallbackPath && !isSuccessPath) {
            writeText(res, 404, "Not Found");
            return;
          }

          const oauthError = requestUrl.searchParams.get("error");
          if (oauthError) {
            const message = requestUrl.searchParams.get("error_description") ?? oauthError;
            writeText(res, 400, `Authentication failed: ${message}`);
            finish({ ok: false, error: new Error(`OAuth authorization failed: ${message}`) });
            return;
          }

          const code = requestUrl.searchParams.get("code")?.trim() ?? "";
          const state = requestUrl.searchParams.get("state")?.trim() ?? "";
          if (!code) {
            if (isSuccessPath) {
              writeText(res, 200, "Authentication complete. You can close this tab and return to loaf.");
              return;
            }
            writeText(res, 400, "Bad Request: Missing code parameter");
            return;
          }

          if (state && state !== expectedState) {
            writeText(res, 400, "Bad Request: Invalid state parameter");
            finish({ ok: false, error: new Error("OAuth login rejected due to state mismatch.") });
            return;
          }

          if (isCallbackPath) {
            res.statusCode = 302;
            res.setHeader("Location", successPath);
            res.end();
          }

          const tokenResponse = await withTimeout(
            oauthClient.getToken(code),
            TOKEN_EXCHANGE_TIMEOUT_MS,
            "OAuth token exchange timed out.",
          );
          const value: AntigravityOauthTokenInfo = {
            accessToken: tokenResponse.tokens.access_token ?? "",
            refreshToken: tokenResponse.tokens.refresh_token ?? "",
            expiryDateSeconds: Math.floor((tokenResponse.tokens.expiry_date ?? 0) / 1000),
            tokenType: tokenResponse.tokens.token_type ?? "",
          };
          if (!value.accessToken.trim()) {
            throw new Error("OAuth token exchange returned an empty access token.");
          }

          if (isSuccessPath) {
            writeText(res, 200, "Authentication complete. You can close this tab and return to loaf.");
          }

          finish({ ok: true, value });
        } catch (error) {
          finish({ ok: false, error });
        }
      })();
    };

    timeoutHandle = setTimeout(() => {
      finish({ ok: false, error: new Error("OAuth login cancelled.") });
    }, timeoutMs);

    server.on("request", onRequest);
    server.once("error", onError);
  });
}

async function listenOnPort(port: number): Promise<{ server: http.Server; port: number }> {
  const server = http.createServer();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, () => {
      server.off("error", reject);
      resolve();
    });
  });

  const address = server.address();
  if (!address || typeof address === "string") {
    server.close();
    throw new Error("Failed to determine OAuth callback port.");
  }

  return {
    server,
    port: address.port,
  };
}

async function closeServer(server: http.Server): Promise<void> {
  if (!server.listening) {
    return;
  }

  try {
    server.closeIdleConnections?.();
  } catch {
    // best-effort
  }

  await new Promise<void>((resolve) => {
    const timeoutHandle = setTimeout(() => {
      try {
        server.closeAllConnections?.();
      } catch {
        // best-effort
      }
      resolve();
    }, SERVER_CLOSE_TIMEOUT_MS);

    server.close(() => {
      clearTimeout(timeoutHandle);
      resolve();
    });
  });
}

function openExternalUrl(url: string): void {
  if (process.platform === "win32") {
    spawn("rundll32", ["url.dll,FileProtocolHandler", url], {
      detached: true,
      stdio: "ignore",
      windowsHide: true,
    }).unref();
    return;
  }

  if (process.platform === "darwin") {
    spawn("open", [url], {
      detached: true,
      stdio: "ignore",
    }).unref();
    return;
  }

  spawn("xdg-open", [url], {
    detached: true,
    stdio: "ignore",
  }).unref();
}

function writeText(res: http.ServerResponse, status: number, body: string): void {
  res.statusCode = status;
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.end(body);
}

function resolveLocalSuccessPath(callbackPath: string): string {
  if (callbackPath !== DEFAULT_LOCAL_SUCCESS_PATH) {
    return DEFAULT_LOCAL_SUCCESS_PATH;
  }
  return "/oauth-complete";
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  let timeoutHandle: NodeJS.Timeout | null = null;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timeoutHandle = setTimeout(() => {
          reject(new Error(message));
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timeoutHandle) {
      clearTimeout(timeoutHandle);
    }
  }
}
