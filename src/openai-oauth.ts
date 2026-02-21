import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { spawn } from "node:child_process";
import { getLoafDataDir } from "./persistence.js";
import { SECRET_ACCOUNT_OPENAI_CHATGPT_AUTH } from "./secret-accounts.js";
import {
  getSecureValue,
  setSecureValue,
} from "./secure-store.js";

const DEFAULT_ISSUER = "https://auth.openai.com";
const DEFAULT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const DEFAULT_PORT = 1455;
const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000;
const DEFAULT_ORIGINATOR = "codex_cli_rs";
const DEFAULT_DEVICE_CODE_INTERVAL_SECONDS = 5;
const DEFAULT_DEVICE_CODE_EXPIRY_SECONDS = 15 * 60;

type OpenAiTokenBundle = {
  id_token: string;
  access_token: string;
  refresh_token: string;
};

export type OpenAiAuthJson = {
  auth_mode?: "chatgpt";
  OPENAI_API_KEY?: string;
  tokens?: {
    id_token?: string;
    access_token?: string;
    refresh_token?: string;
    account_id?: string;
  };
  last_refresh?: string;
};

export type OpenAiChatgptAuth = {
  accessToken: string;
  accountId: string | null;
  refreshToken?: string;
  idToken?: string;
  apiKey?: string;
};

export type OpenAiOauthLoginOptions = {
  issuer?: string;
  clientId?: string;
  port?: number;
  timeoutMs?: number;
  openBrowser?: boolean;
  originator?: string;
  mode?: "auto" | "browser" | "device_code";
  onDeviceCode?: (info: OpenAiDeviceCodeInfo) => void;
};

export type OpenAiOauthLoginResult = {
  authUrl: string;
  authFilePath: string;
  chatgptAuth: OpenAiChatgptAuth;
  loginMethod: "browser" | "device_code";
};

type StoredOpenAiChatgptSecrets = {
  version: 1;
  accessToken: string;
  accountId?: string | null;
  refreshToken?: string;
  idToken?: string;
  apiKey?: string;
};

export type OpenAiDeviceCodeInfo = {
  verificationUrl: string;
  userCode: string;
  intervalSeconds: number;
  expiresInSeconds: number;
};

export function getOpenAiAuthFilePath(): string {
  return path.join(getLoafDataDir(), "auth.json");
}

export function loadOpenAiAuthFromDisk(): OpenAiAuthJson | null {
  const authFilePath = getOpenAiAuthFilePath();
  try {
    if (!fs.existsSync(authFilePath)) {
      return null;
    }

    const raw = fs.readFileSync(authFilePath, "utf8");
    const parsed = JSON.parse(raw) as Partial<OpenAiAuthJson>;
    if (!parsed || typeof parsed !== "object") {
      return null;
    }

    const apiKey = typeof parsed.OPENAI_API_KEY === "string" ? parsed.OPENAI_API_KEY.trim() : "";
    const tokens = parsed.tokens;
    const normalizedTokens =
      tokens && typeof tokens === "object"
        ? {
            id_token: typeof tokens.id_token === "string" ? tokens.id_token : undefined,
            access_token: typeof tokens.access_token === "string" ? tokens.access_token : undefined,
            refresh_token: typeof tokens.refresh_token === "string" ? tokens.refresh_token : undefined,
            account_id: typeof tokens.account_id === "string" ? tokens.account_id : undefined,
          }
        : undefined;

    return {
      auth_mode: parsed.auth_mode === "chatgpt" ? "chatgpt" : undefined,
      OPENAI_API_KEY: apiKey || undefined,
      tokens: normalizedTokens,
      last_refresh: typeof parsed.last_refresh === "string" ? parsed.last_refresh : undefined,
    };
  } catch {
    return null;
  }
}

export function getPersistedOpenAiChatgptAuth(): OpenAiChatgptAuth | null {
  return normalizePersistedOpenAiAuth(loadOpenAiAuthFromDisk());
}

export async function loadPersistedOpenAiChatgptAuth(): Promise<OpenAiChatgptAuth | null> {
  const fromSecureStore = await loadOpenAiChatgptAuthFromSecureStore();
  if (fromSecureStore) {
    const authFromDisk = loadOpenAiAuthFromDisk();
    if (hasLegacySecretsInAuthFile(authFromDisk)) {
      writeOpenAiAuthMetadata({
        accountId: fromSecureStore.accountId,
        lastRefresh: authFromDisk?.last_refresh,
      });
    }
    return fromSecureStore;
  }

  const authFromDisk = loadOpenAiAuthFromDisk();
  const fromLegacyDisk = normalizePersistedOpenAiAuth(authFromDisk);
  if (!fromLegacyDisk) {
    return null;
  }

  const migrated = await persistOpenAiAuthToSecureStore(fromLegacyDisk);
  if (migrated) {
    writeOpenAiAuthMetadata({
      accountId: fromLegacyDisk.accountId,
      lastRefresh: authFromDisk?.last_refresh,
    });
  }

  return fromLegacyDisk;
}

export async function runOpenAiOauthLogin(
  options: OpenAiOauthLoginOptions = {},
): Promise<OpenAiOauthLoginResult> {
  const issuer = (options.issuer ?? DEFAULT_ISSUER).replace(/\/+$/, "");
  const clientId = options.clientId ?? DEFAULT_CLIENT_ID;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const mode = options.mode ?? "auto";
  const openBrowser = options.openBrowser ?? !isLikelyHeadlessEnvironment();
  const originator = resolveOriginator(options.originator);
  const shouldUseDeviceCode = mode === "device_code" || (mode === "auto" && isLikelyHeadlessEnvironment());

  if (shouldUseDeviceCode) {
    try {
      const result = await runOpenAiDeviceCodeLogin({
        issuer,
        clientId,
        timeoutMs,
        openBrowser,
        onDeviceCode: options.onDeviceCode,
      });
      return {
        authUrl: result.deviceCode.verificationUrl,
        authFilePath: result.authFilePath,
        chatgptAuth: result.chatgptAuth,
        loginMethod: "device_code",
      };
    } catch (error) {
      if (!(mode === "auto" && isDeviceCodeUnsupportedError(error))) {
        throw error;
      }
    }
  }

  const { server, port } = await listenWithFallback(options.port ?? DEFAULT_PORT);
  const pkce = generatePkce();
  const state = randomBase64Url(32);
  const redirectUri = `http://localhost:${port}/auth/callback`;
  const authUrl = buildAuthorizeUrl({
    issuer,
    clientId,
    redirectUri,
    state,
    codeChallenge: pkce.codeChallenge,
    originator,
  });

  if (openBrowser) {
    openExternalUrl(authUrl);
  }

  const result = await waitForOAuthCompletion({
    server,
    clientId,
    tokenEndpoint: `${issuer}/oauth/token`,
    state,
    redirectUri,
    codeVerifier: pkce.codeVerifier,
    timeoutMs,
  });

  return {
    authUrl,
    authFilePath: result.authFilePath,
    chatgptAuth: result.chatgptAuth,
    loginMethod: "browser",
  };
}

function generatePkce(): { codeVerifier: string; codeChallenge: string } {
  const codeVerifier = randomBase64Url(64);
  const digest = crypto.createHash("sha256").update(codeVerifier).digest();
  const codeChallenge = toBase64Url(digest);
  return { codeVerifier, codeChallenge };
}

function randomBase64Url(bytes: number): string {
  return toBase64Url(crypto.randomBytes(bytes));
}

function toBase64Url(value: Buffer): string {
  return value.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function buildAuthorizeUrl(params: {
  issuer: string;
  clientId: string;
  redirectUri: string;
  state: string;
  codeChallenge: string;
  originator: string;
}): string {
  const query = new URLSearchParams({
    response_type: "code",
    client_id: params.clientId,
    redirect_uri: params.redirectUri,
    scope: "openid profile email offline_access",
    code_challenge: params.codeChallenge,
    code_challenge_method: "S256",
    id_token_add_organizations: "true",
    codex_cli_simplified_flow: "true",
    originator: params.originator,
    state: params.state,
  });

  return `${params.issuer}/oauth/authorize?${query.toString()}`;
}

function resolveOriginator(candidate: string | undefined): string {
  const fromOption = candidate?.trim();
  if (fromOption) {
    return fromOption;
  }

  return DEFAULT_ORIGINATOR;
}

export function isLikelyHeadlessEnvironment(): boolean {
  if (hasNonEmptyEnv("CI") || hasNonEmptyEnv("SSH_CONNECTION") || hasNonEmptyEnv("SSH_CLIENT") || hasNonEmptyEnv("SSH_TTY")) {
    return true;
  }

  if (process.platform === "linux" && !hasNonEmptyEnv("DISPLAY") && !hasNonEmptyEnv("WAYLAND_DISPLAY")) {
    return true;
  }

  return false;
}

function hasNonEmptyEnv(key: string): boolean {
  const value = process.env[key];
  return typeof value === "string" && value.trim().length > 0;
}

class DeviceCodeUnsupportedError extends Error {}

function isDeviceCodeUnsupportedError(error: unknown): boolean {
  return error instanceof DeviceCodeUnsupportedError;
}

async function runOpenAiDeviceCodeLogin(options: {
  issuer: string;
  clientId: string;
  timeoutMs: number;
  openBrowser: boolean;
  onDeviceCode?: (info: OpenAiDeviceCodeInfo) => void;
}): Promise<{ authFilePath: string; chatgptAuth: OpenAiChatgptAuth; deviceCode: OpenAiDeviceCodeInfo }> {
  const deviceCode = await requestOpenAiDeviceCode(options.issuer, options.clientId);
  options.onDeviceCode?.({
    verificationUrl: deviceCode.verificationUrl,
    userCode: deviceCode.userCode,
    intervalSeconds: deviceCode.intervalSeconds,
    expiresInSeconds: deviceCode.expiresInSeconds,
  });

  if (options.openBrowser) {
    openExternalUrl(deviceCode.verificationUrl);
  }

  const authorization = await waitForOpenAiDeviceAuthorization({
    issuer: options.issuer,
    deviceAuthId: deviceCode.deviceAuthId,
    userCode: deviceCode.userCode,
    intervalSeconds: deviceCode.intervalSeconds,
    timeoutMs: options.timeoutMs,
  });

  const tokens = await exchangeAuthorizationCode({
    tokenEndpoint: `${options.issuer}/oauth/token`,
    clientId: options.clientId,
    redirectUri: `${options.issuer}/deviceauth/callback`,
    codeVerifier: authorization.codeVerifier,
    code: authorization.authorizationCode,
  });

  let apiKey: string | undefined;
  try {
    apiKey = await exchangeIdTokenForApiKey({
      tokenEndpoint: `${options.issuer}/oauth/token`,
      clientId: options.clientId,
      idToken: tokens.id_token,
    });
  } catch {
    // Keep ChatGPT-account auth usable even if API key exchange is unavailable.
  }

  const accountId = readChatGptAccountId(tokens.id_token);
  const authFilePath = await persistOpenAiAuth({
    apiKey,
    accountId,
    idToken: tokens.id_token,
    accessToken: tokens.access_token,
    refreshToken: tokens.refresh_token,
  });

  return {
    authFilePath,
    chatgptAuth: {
      accessToken: tokens.access_token,
      accountId,
      refreshToken: tokens.refresh_token,
      idToken: tokens.id_token,
      apiKey,
    },
    deviceCode: {
      verificationUrl: deviceCode.verificationUrl,
      userCode: deviceCode.userCode,
      intervalSeconds: deviceCode.intervalSeconds,
      expiresInSeconds: deviceCode.expiresInSeconds,
    },
  };
}

type OpenAiDeviceCodeState = {
  deviceAuthId: string;
  userCode: string;
  verificationUrl: string;
  intervalSeconds: number;
  expiresInSeconds: number;
};

async function requestOpenAiDeviceCode(issuer: string, clientId: string): Promise<OpenAiDeviceCodeState> {
  const response = await fetch(`${issuer}/api/accounts/deviceauth/usercode`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      client_id: clientId,
    }),
  });

  if (!response.ok) {
    if (response.status === 404) {
      throw new DeviceCodeUnsupportedError("Device code login is not enabled for this OAuth issuer.");
    }

    const text = await response.text();
    throw new Error(`Device code request failed (${response.status}): ${summarizeHttpError(text)}`);
  }

  const payload = (await response.json()) as Partial<{
    device_auth_id: string;
    user_code: string;
    usercode: string;
    interval: string | number;
    expires_in: string | number;
  }>;
  const deviceAuthId = payload.device_auth_id?.trim();
  const userCode = (payload.user_code ?? payload.usercode ?? "").trim();
  if (!deviceAuthId || !userCode) {
    throw new Error("Device code request returned an incomplete payload.");
  }

  return {
    deviceAuthId,
    userCode,
    verificationUrl: `${issuer}/codex/device`,
    intervalSeconds: parsePositiveInteger(payload.interval, DEFAULT_DEVICE_CODE_INTERVAL_SECONDS),
    expiresInSeconds: parsePositiveInteger(payload.expires_in, DEFAULT_DEVICE_CODE_EXPIRY_SECONDS),
  };
}

async function waitForOpenAiDeviceAuthorization(options: {
  issuer: string;
  deviceAuthId: string;
  userCode: string;
  intervalSeconds: number;
  timeoutMs: number;
}): Promise<{ authorizationCode: string; codeVerifier: string }> {
  const startedAt = Date.now();
  const pollIntervalMs = Math.max(1, options.intervalSeconds) * 1000;
  while (Date.now() - startedAt < options.timeoutMs) {
    const response = await fetch(`${options.issuer}/api/accounts/deviceauth/token`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        device_auth_id: options.deviceAuthId,
        user_code: options.userCode,
      }),
    });

    if (response.ok) {
      const payload = (await response.json()) as Partial<{
        authorization_code: string;
        code_verifier: string;
      }>;
      const authorizationCode = payload.authorization_code?.trim();
      const codeVerifier = payload.code_verifier?.trim();
      if (!authorizationCode || !codeVerifier) {
        throw new Error("Device code poll returned an incomplete authorization payload.");
      }
      return {
        authorizationCode,
        codeVerifier,
      };
    }

    if (response.status !== 403 && response.status !== 404) {
      const text = await response.text();
      throw new Error(`Device code poll failed (${response.status}): ${summarizeHttpError(text)}`);
    }

    const elapsedMs = Date.now() - startedAt;
    const remainingMs = options.timeoutMs - elapsedMs;
    if (remainingMs <= 0) {
      break;
    }
    await sleep(Math.min(pollIntervalMs, remainingMs));
  }

  throw new Error("OAuth device login timed out after 15 minutes.");
}

function parsePositiveInteger(value: string | number | undefined, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return Math.floor(value);
  }
  if (typeof value === "string") {
    const parsed = Number.parseInt(value.trim(), 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }
  return fallback;
}

async function sleep(ms: number): Promise<void> {
  await new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function listenWithFallback(requestedPort: number): Promise<{ server: http.Server; port: number }> {
  try {
    return await listenOnPort(requestedPort);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== "EADDRINUSE" || requestedPort === 0) {
      throw error;
    }
    return listenOnPort(0);
  }
}

async function listenOnPort(port: number): Promise<{ server: http.Server; port: number }> {
  const server = http.createServer();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => {
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

async function waitForOAuthCompletion(options: {
  server: http.Server;
  clientId: string;
  tokenEndpoint: string;
  state: string;
  redirectUri: string;
  codeVerifier: string;
  timeoutMs: number;
}): Promise<{ authFilePath: string; chatgptAuth: OpenAiChatgptAuth }> {
  const { server } = options;

  let settled = false;
  let timeoutHandle: NodeJS.Timeout | undefined;

  const finish = async (
    resolve: (value: { authFilePath: string; chatgptAuth: OpenAiChatgptAuth }) => void,
    reject: (reason?: unknown) => void,
    outcome:
      | { ok: true; value: { authFilePath: string; chatgptAuth: OpenAiChatgptAuth } }
      | { ok: false; error: unknown },
  ) => {
    if (settled) {
      return;
    }
    settled = true;
    if (timeoutHandle) {
      clearTimeout(timeoutHandle);
      timeoutHandle = undefined;
    }
    await closeServer(server);
    if (outcome.ok) {
      resolve(outcome.value);
    } else {
      reject(outcome.error);
    }
  };

  const done = new Promise<{ authFilePath: string; chatgptAuth: OpenAiChatgptAuth }>((resolve, reject) => {
    timeoutHandle = setTimeout(() => {
      void finish(resolve, reject, {
        ok: false,
        error: new Error("OAuth login timed out after 15 minutes."),
      });
    }, options.timeoutMs);

    server.on("request", (req, res) => {
      void handleOAuthRequest(options, req, res)
        .then(async (result) => {
          if (!result) {
            return;
          }
          await finish(resolve, reject, {
            ok: true,
            value: result,
          });
        })
        .catch(async (error) => {
          await finish(resolve, reject, {
            ok: false,
            error,
          });
        });
    });

    server.once("error", (error) => {
      void finish(resolve, reject, {
        ok: false,
        error,
      });
    });
  });

  return done;
}

async function handleOAuthRequest(
  options: {
    clientId: string;
    tokenEndpoint: string;
    state: string;
    redirectUri: string;
    codeVerifier: string;
  },
  req: http.IncomingMessage,
  res: http.ServerResponse,
): Promise<{ authFilePath: string; chatgptAuth: OpenAiChatgptAuth } | null> {
  const requestUrl = new URL(req.url ?? "/", "http://localhost");

  if (requestUrl.pathname === "/success") {
    writeHtml(res, 200, successHtml());
    return null;
  }

  if (requestUrl.pathname === "/cancel") {
    writeText(res, 200, "Login cancelled.");
    throw new Error("OAuth login cancelled.");
  }

  if (requestUrl.pathname !== "/auth/callback") {
    writeText(res, 404, "Not found.");
    return null;
  }

  const returnedState = requestUrl.searchParams.get("state") ?? "";
  if (!returnedState || returnedState !== options.state) {
    writeText(res, 400, "State mismatch.");
    throw new Error("OAuth login failed: state mismatch.");
  }

  const code = (requestUrl.searchParams.get("code") ?? "").trim();
  if (!code) {
    const errorMessage = requestUrl.searchParams.get("error_description") ?? "Missing authorization code.";
    writeText(res, 400, errorMessage);
    throw new Error(`OAuth login failed: ${errorMessage}`);
  }

  const tokens = await exchangeAuthorizationCode({
    tokenEndpoint: options.tokenEndpoint,
    clientId: options.clientId,
    redirectUri: options.redirectUri,
    codeVerifier: options.codeVerifier,
    code,
  });

  let apiKey: string | undefined;
  try {
    apiKey = await exchangeIdTokenForApiKey({
      tokenEndpoint: options.tokenEndpoint,
      clientId: options.clientId,
      idToken: tokens.id_token,
    });
  } catch {
    // Keep ChatGPT-account auth usable even if API key exchange is unavailable.
  }

  const accountId = readChatGptAccountId(tokens.id_token);
  const authFilePath = await persistOpenAiAuth({
    apiKey,
    accountId,
    idToken: tokens.id_token,
    accessToken: tokens.access_token,
    refreshToken: tokens.refresh_token,
  });

  const successUrl = new URL(options.redirectUri.replace("/auth/callback", "/success"));
  res.statusCode = 302;
  res.setHeader("Location", successUrl.toString());
  res.end();

  return {
    authFilePath,
    chatgptAuth: {
      accessToken: tokens.access_token,
      accountId,
      refreshToken: tokens.refresh_token,
      idToken: tokens.id_token,
      apiKey,
    },
  };
}

async function exchangeAuthorizationCode(params: {
  tokenEndpoint: string;
  clientId: string;
  redirectUri: string;
  codeVerifier: string;
  code: string;
}): Promise<OpenAiTokenBundle> {
  const body = new URLSearchParams({
    grant_type: "authorization_code",
    code: params.code,
    redirect_uri: params.redirectUri,
    client_id: params.clientId,
    code_verifier: params.codeVerifier,
  });

  const response = await fetch(params.tokenEndpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OAuth token exchange failed (${response.status}): ${summarizeHttpError(text)}`);
  }

  const json = (await response.json()) as Partial<OpenAiTokenBundle>;
  if (!json.id_token || !json.access_token || !json.refresh_token) {
    throw new Error("OAuth token exchange returned an incomplete token payload.");
  }

  return {
    id_token: json.id_token,
    access_token: json.access_token,
    refresh_token: json.refresh_token,
  };
}

async function exchangeIdTokenForApiKey(params: {
  tokenEndpoint: string;
  clientId: string;
  idToken: string;
}): Promise<string> {
  const body = new URLSearchParams({
    grant_type: "urn:ietf:params:oauth:grant-type:token-exchange",
    client_id: params.clientId,
    requested_token: "openai-api-key",
    subject_token: params.idToken,
    subject_token_type: "urn:ietf:params:oauth:token-type:id_token",
  });

  const response = await fetch(params.tokenEndpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OpenAI API key exchange failed (${response.status}): ${summarizeHttpError(text)}`);
  }

  const json = (await response.json()) as Partial<{ access_token: string }>;
  const apiKey = json.access_token?.trim();
  if (!apiKey) {
    throw new Error("OpenAI API key exchange did not return an access_token.");
  }
  return apiKey;
}

async function loadOpenAiChatgptAuthFromSecureStore(): Promise<OpenAiChatgptAuth | null> {
  const raw = await getSecureValue(SECRET_ACCOUNT_OPENAI_CHATGPT_AUTH);
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as Partial<StoredOpenAiChatgptSecrets>;
    if (!parsed || typeof parsed !== "object" || parsed.version !== 1) {
      return null;
    }

    const accessToken = typeof parsed.accessToken === "string" ? parsed.accessToken.trim() : "";
    if (!accessToken) {
      return null;
    }

    const idToken = typeof parsed.idToken === "string" ? parsed.idToken : undefined;
    const accountId =
      typeof parsed.accountId === "string"
        ? parsed.accountId
        : (parsed.accountId ?? readChatGptAccountId(idToken ?? "")) || null;
    const refreshToken = typeof parsed.refreshToken === "string" ? parsed.refreshToken : undefined;
    const apiKey = typeof parsed.apiKey === "string" ? parsed.apiKey : undefined;

    return {
      accessToken,
      accountId,
      refreshToken,
      idToken,
      apiKey,
    };
  } catch {
    return null;
  }
}

function normalizePersistedOpenAiAuth(auth: OpenAiAuthJson | null): OpenAiChatgptAuth | null {
  if (!auth) {
    return null;
  }

  const accessToken = auth.tokens?.access_token?.trim();
  if (!accessToken) {
    return null;
  }

  const idToken = auth.tokens?.id_token;
  const accountId = auth.tokens?.account_id?.trim() || readChatGptAccountId(idToken ?? "") || null;
  const refreshToken = auth.tokens?.refresh_token?.trim() || undefined;
  const apiKey = auth.OPENAI_API_KEY?.trim() || undefined;

  return {
    accessToken,
    accountId,
    refreshToken,
    idToken,
    apiKey,
  };
}

function hasLegacySecretsInAuthFile(auth: OpenAiAuthJson | null): boolean {
  if (!auth) {
    return false;
  }
  return Boolean(
    auth.OPENAI_API_KEY?.trim() ||
      auth.tokens?.access_token?.trim() ||
      auth.tokens?.refresh_token?.trim() ||
      auth.tokens?.id_token?.trim(),
  );
}

async function persistOpenAiAuthToSecureStore(auth: OpenAiChatgptAuth): Promise<boolean> {
  const payload: StoredOpenAiChatgptSecrets = {
    version: 1,
    accessToken: auth.accessToken.trim(),
    accountId: auth.accountId,
    refreshToken: auth.refreshToken?.trim() || undefined,
    idToken: auth.idToken?.trim() || undefined,
    apiKey: auth.apiKey?.trim() || undefined,
  };
  if (!payload.accessToken) {
    return false;
  }

  return setSecureValue(SECRET_ACCOUNT_OPENAI_CHATGPT_AUTH, JSON.stringify(payload));
}

function writeOpenAiAuthMetadata(params: { accountId: string | null; lastRefresh?: string }): string {
  const payload: OpenAiAuthJson = {
    auth_mode: "chatgpt",
    tokens: {
      account_id: params.accountId || undefined,
    },
    last_refresh: params.lastRefresh || new Date().toISOString(),
  };

  const authFilePath = getOpenAiAuthFilePath();
  fs.mkdirSync(path.dirname(authFilePath), { recursive: true });
  fs.writeFileSync(authFilePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  return authFilePath;
}

async function persistOpenAiAuth(params: {
  apiKey?: string;
  accountId: string | null;
  idToken: string;
  accessToken: string;
  refreshToken: string;
}): Promise<string> {
  const auth: OpenAiChatgptAuth = {
    accessToken: params.accessToken,
    accountId: params.accountId,
    refreshToken: params.refreshToken,
    idToken: params.idToken,
    apiKey: params.apiKey,
  };

  const secureStored = await persistOpenAiAuthToSecureStore(auth);
  const lastRefresh = new Date().toISOString();
  if (secureStored) {
    return writeOpenAiAuthMetadata({
      accountId: params.accountId,
      lastRefresh,
    });
  }

  const payload: OpenAiAuthJson = {
    auth_mode: "chatgpt",
    OPENAI_API_KEY: params.apiKey,
    tokens: {
      id_token: params.idToken,
      access_token: params.accessToken,
      refresh_token: params.refreshToken,
      account_id: params.accountId || undefined,
    },
    last_refresh: lastRefresh,
  };

  const authFilePath = getOpenAiAuthFilePath();
  fs.mkdirSync(path.dirname(authFilePath), { recursive: true });
  fs.writeFileSync(authFilePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  return authFilePath;
}

function readChatGptAccountId(idToken: string): string | null {
  try {
    const parts = idToken.split(".");
    if (parts.length < 2) {
      return null;
    }
    const payloadRaw = decodeBase64Url(parts[1] ?? "");
    const parsed = JSON.parse(payloadRaw) as Record<string, unknown>;
    const authClaims = parsed["https://api.openai.com/auth"] as Record<string, unknown> | undefined;
    const accountId = authClaims?.chatgpt_account_id;
    if (typeof accountId !== "string") {
      return null;
    }
    return accountId;
  } catch {
    return null;
  }
}

function decodeBase64Url(value: string): string {
  const base64 = value.replace(/-/g, "+").replace(/_/g, "/");
  const pad = base64.length % 4;
  const padded = pad === 0 ? base64 : `${base64}${"=".repeat(4 - pad)}`;
  return Buffer.from(padded, "base64").toString("utf8");
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

function writeText(res: http.ServerResponse, status: number, body: string): void {
  res.statusCode = status;
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.end(body);
}

function writeHtml(res: http.ServerResponse, status: number, body: string): void {
  res.statusCode = status;
  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.end(body);
}

function successHtml(): string {
  return [
    "<!doctype html>",
    "<html>",
    "<head><meta charset=\"utf-8\" /><title>Loaf Login Success</title></head>",
    "<body style=\"font-family: system-ui, sans-serif; padding: 28px;\">",
    "<h1>OpenAI login complete</h1>",
    "<p>You can return to the terminal and continue using loaf.</p>",
    "</body>",
    "</html>",
  ].join("");
}

function openExternalUrl(url: string): void {
  if (process.platform === "win32") {
    // Avoid cmd parsing (`&`, `^`, etc.) that can truncate OAuth query params.
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

async function closeServer(server: http.Server): Promise<void> {
  if (!server.listening) {
    return;
  }

  await new Promise<void>((resolve) => {
    server.close(() => resolve());
  });
}
