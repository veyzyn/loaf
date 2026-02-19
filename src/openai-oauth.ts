import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { spawn } from "node:child_process";
import { getLoafDataDir } from "./persistence.js";

const DEFAULT_ISSUER = "https://auth.openai.com";
const DEFAULT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const DEFAULT_PORT = 1455;
const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000;
const DEFAULT_ORIGINATOR = "codex_cli_rs";

type OpenAiTokenBundle = {
  id_token: string;
  access_token: string;
  refresh_token: string;
};

export type OpenAiAuthJson = {
  auth_mode?: "chatgpt";
  OPENAI_API_KEY?: string;
  tokens?: {
    id_token: string;
    access_token: string;
    refresh_token: string;
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
};

export type OpenAiOauthLoginResult = {
  authUrl: string;
  authFilePath: string;
  chatgptAuth: OpenAiChatgptAuth;
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
      tokens &&
      typeof tokens === "object" &&
      typeof tokens.id_token === "string" &&
      typeof tokens.access_token === "string" &&
      typeof tokens.refresh_token === "string"
        ? {
            id_token: tokens.id_token,
            access_token: tokens.access_token,
            refresh_token: tokens.refresh_token,
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
  const auth = loadOpenAiAuthFromDisk();
  if (!auth) {
    return null;
  }

  const accessToken = auth?.tokens?.access_token?.trim();
  if (!accessToken) {
    return null;
  }

  const accountId =
    auth.tokens?.account_id?.trim() ||
    readChatGptAccountId(auth.tokens?.id_token ?? "") ||
    null;

  return {
    accessToken,
    accountId,
    refreshToken: auth.tokens?.refresh_token,
    idToken: auth.tokens?.id_token,
    apiKey: auth.OPENAI_API_KEY,
  };
}

export async function runOpenAiOauthLogin(
  options: OpenAiOauthLoginOptions = {},
): Promise<OpenAiOauthLoginResult> {
  const issuer = (options.issuer ?? DEFAULT_ISSUER).replace(/\/+$/, "");
  const clientId = options.clientId ?? DEFAULT_CLIENT_ID;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const openBrowser = options.openBrowser !== false;
  const originator = resolveOriginator(options.originator);
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
    issuer,
    clientId,
    state,
    redirectUri,
    codeVerifier: pkce.codeVerifier,
    timeoutMs,
  });

  return {
    authUrl,
    authFilePath: result.authFilePath,
    chatgptAuth: result.chatgptAuth,
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
  issuer: string;
  clientId: string;
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
    issuer: string;
    clientId: string;
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
    issuer: options.issuer,
    clientId: options.clientId,
    redirectUri: options.redirectUri,
    codeVerifier: options.codeVerifier,
    code,
  });

  let apiKey: string | undefined;
  try {
    apiKey = await exchangeIdTokenForApiKey({
      issuer: options.issuer,
      clientId: options.clientId,
      idToken: tokens.id_token,
    });
  } catch {
    // Keep ChatGPT-account auth usable even if API key exchange is unavailable.
  }

  const accountId = readChatGptAccountId(tokens.id_token);
  const authFilePath = persistOpenAiAuth({
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
  issuer: string;
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

  const response = await fetch(`${params.issuer}/oauth/token`, {
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
  issuer: string;
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

  const response = await fetch(`${params.issuer}/oauth/token`, {
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

function persistOpenAiAuth(params: {
  apiKey?: string;
  accountId: string | null;
  idToken: string;
  accessToken: string;
  refreshToken: string;
}): string {
  const payload: OpenAiAuthJson = {
    auth_mode: "chatgpt",
    OPENAI_API_KEY: params.apiKey,
    tokens: {
      id_token: params.idToken,
      access_token: params.accessToken,
      refresh_token: params.refreshToken,
      account_id: params.accountId || undefined,
    },
    last_refresh: new Date().toISOString(),
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
