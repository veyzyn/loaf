import { randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { AuthProvider, ThinkingLevel } from "./config.js";
import { getLoafDataDir } from "./persistence.js";
import type { ChatMessage } from "./vertex.js";

const SESSIONS_DIR = path.join(getLoafDataDir(), "sessions");
const SESSION_FILE_PREFIX = "rollout-";
const SESSION_FILE_SUFFIX = ".jsonl";
const MAX_SESSION_TITLE_LENGTH = 80;

type SessionMetaLine = {
  type: "session_meta";
  id: string;
  title: string;
  provider: AuthProvider;
  model: string;
  thinkingLevel: ThinkingLevel;
  openRouterProvider?: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
};

type SessionMessageLine = {
  type: "message";
  index: number;
  role: ChatMessage["role"];
  text: string;
};

export type ChatSessionSummary = {
  id: string;
  title: string;
  provider: AuthProvider;
  model: string;
  thinkingLevel: ThinkingLevel;
  openRouterProvider?: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
  rolloutPath: string;
};

export type ChatSessionRecord = ChatSessionSummary & {
  messages: ChatMessage[];
};

export function createChatSession(params: {
  provider: AuthProvider;
  model: string;
  thinkingLevel: ThinkingLevel;
  openRouterProvider?: string;
  titleHint?: string;
}): ChatSessionSummary {
  const now = new Date();
  const id = randomUUID();
  const rolloutPath = buildRolloutPath(now, id);
  const createdAt = now.toISOString();
  return {
    id,
    title: deriveSessionTitle([], params.titleHint),
    provider: params.provider,
    model: params.model.trim(),
    thinkingLevel: params.thinkingLevel,
    openRouterProvider: normalizeOpenRouterProvider(params.openRouterProvider),
    createdAt,
    updatedAt: createdAt,
    messageCount: 0,
    rolloutPath,
  };
}

export function writeChatSession(params: {
  session: ChatSessionSummary;
  messages: ChatMessage[];
  provider: AuthProvider;
  model: string;
  thinkingLevel: ThinkingLevel;
  openRouterProvider?: string;
  titleHint?: string;
}): ChatSessionSummary {
  const nowIso = new Date().toISOString();
  const normalizedMessages = params.messages
    .map((message) => normalizeChatMessage(message))
    .filter((message): message is ChatMessage => message !== null);
  const title = deriveSessionTitle(normalizedMessages, params.titleHint || params.session.title);
  const provider = params.provider;
  const model = params.model.trim();
  const metaLine: SessionMetaLine = {
    type: "session_meta",
    id: params.session.id,
    title,
    provider,
    model,
    thinkingLevel: params.thinkingLevel,
    openRouterProvider: provider === "openrouter" ? normalizeOpenRouterProvider(params.openRouterProvider) : undefined,
    createdAt: params.session.createdAt,
    updatedAt: nowIso,
    messageCount: normalizedMessages.length,
  };

  const lines: Array<SessionMetaLine | SessionMessageLine> = [metaLine];
  for (let i = 0; i < normalizedMessages.length; i += 1) {
    lines.push({
      type: "message",
      index: i,
      role: normalizedMessages[i]!.role,
      text: normalizedMessages[i]!.text,
    });
  }
  writeRolloutFile(params.session.rolloutPath, lines);

  return {
    id: metaLine.id,
    title: metaLine.title,
    provider: metaLine.provider,
    model: metaLine.model,
    thinkingLevel: metaLine.thinkingLevel,
    openRouterProvider: metaLine.openRouterProvider,
    createdAt: metaLine.createdAt,
    updatedAt: metaLine.updatedAt,
    messageCount: metaLine.messageCount,
    rolloutPath: params.session.rolloutPath,
  };
}

export function listChatSessions(params?: { limit?: number }): ChatSessionSummary[] {
  const rolloutPaths = listRolloutFiles(SESSIONS_DIR);
  const sessions: ChatSessionSummary[] = [];
  for (const rolloutPath of rolloutPaths) {
    const summary = readChatSessionSummary(rolloutPath);
    if (!summary) {
      continue;
    }
    sessions.push(summary);
  }

  sessions.sort((left, right) => {
    const leftTime = Date.parse(left.updatedAt);
    const rightTime = Date.parse(right.updatedAt);
    if (Number.isFinite(leftTime) && Number.isFinite(rightTime)) {
      return rightTime - leftTime;
    }
    return right.updatedAt.localeCompare(left.updatedAt);
  });

  const limit = params?.limit;
  if (typeof limit === "number" && Number.isFinite(limit) && limit > 0) {
    return sessions.slice(0, Math.floor(limit));
  }
  return sessions;
}

export function loadLatestChatSession(): ChatSessionRecord | null {
  const latest = listChatSessions({ limit: 1 })[0];
  if (!latest) {
    return null;
  }
  return loadChatSession(latest.rolloutPath);
}

export function loadChatSessionById(sessionId: string): ChatSessionRecord | null {
  const needle = sessionId.trim().toLowerCase();
  if (!needle) {
    return null;
  }

  const sessions = listChatSessions();
  const exact = sessions.find((session) => session.id.toLowerCase() === needle);
  if (exact) {
    return loadChatSession(exact.rolloutPath);
  }

  const prefixMatches = sessions.filter((session) => session.id.toLowerCase().startsWith(needle));
  if (prefixMatches.length === 1) {
    return loadChatSession(prefixMatches[0]!.rolloutPath);
  }
  return null;
}

export function loadChatSession(rolloutPath: string): ChatSessionRecord | null {
  try {
    const raw = fs.readFileSync(rolloutPath, "utf8");
    const lines = raw
      .replace(/\r\n/g, "\n")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (lines.length === 0) {
      return null;
    }

    let meta: SessionMetaLine | null = null;
    const messages: ChatMessage[] = [];

    for (const line of lines) {
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch {
        continue;
      }
      if (!isRecord(parsed)) {
        continue;
      }

      const typeValue = readTrimmedString(parsed.type);
      if (typeValue === "session_meta") {
        const normalizedMeta = parseSessionMetaLine(parsed);
        if (normalizedMeta) {
          meta = normalizedMeta;
        }
        continue;
      }

      if (typeValue === "message") {
        const role = parsed.role;
        const text = readTrimmedString(parsed.text);
        if ((role === "user" || role === "assistant") && text) {
          messages.push({
            role,
            text,
          });
        }
      }
    }

    if (!meta) {
      return null;
    }

    const messageCount = messages.length > 0 ? messages.length : meta.messageCount;
    return {
      id: meta.id,
      title: meta.title,
      provider: meta.provider,
      model: meta.model,
      thinkingLevel: meta.thinkingLevel,
      openRouterProvider: meta.openRouterProvider,
      createdAt: meta.createdAt,
      updatedAt: meta.updatedAt,
      messageCount,
      rolloutPath,
      messages,
    };
  } catch {
    return null;
  }
}

function readChatSessionSummary(rolloutPath: string): ChatSessionSummary | null {
  const session = loadChatSession(rolloutPath);
  if (!session) {
    return null;
  }
  return {
    id: session.id,
    title: session.title,
    provider: session.provider,
    model: session.model,
    thinkingLevel: session.thinkingLevel,
    openRouterProvider: session.openRouterProvider,
    createdAt: session.createdAt,
    updatedAt: session.updatedAt,
    messageCount: session.messageCount,
    rolloutPath: session.rolloutPath,
  };
}

function buildRolloutPath(date: Date, sessionId: string): string {
  const year = String(date.getUTCFullYear());
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  const timestamp = [
    year,
    month,
    day,
    "-",
    String(date.getUTCHours()).padStart(2, "0"),
    String(date.getUTCMinutes()).padStart(2, "0"),
    String(date.getUTCSeconds()).padStart(2, "0"),
    String(date.getUTCMilliseconds()).padStart(3, "0"),
  ].join("");
  return path.join(
    SESSIONS_DIR,
    year,
    month,
    day,
    `${SESSION_FILE_PREFIX}${timestamp}-${sessionId}${SESSION_FILE_SUFFIX}`,
  );
}

function writeRolloutFile(rolloutPath: string, lines: Array<SessionMetaLine | SessionMessageLine>): void {
  const dirPath = path.dirname(rolloutPath);
  const tmpPath = `${rolloutPath}.tmp`;
  const payload = `${lines.map((line) => JSON.stringify(line)).join("\n")}\n`;

  fs.mkdirSync(dirPath, { recursive: true });
  fs.writeFileSync(tmpPath, payload, "utf8");
  fs.renameSync(tmpPath, rolloutPath);
}

function listRolloutFiles(rootDir: string): string[] {
  if (!fs.existsSync(rootDir)) {
    return [];
  }

  const files: string[] = [];
  const stack: string[] = [rootDir];
  while (stack.length > 0) {
    const nextDir = stack.pop();
    if (!nextDir) {
      continue;
    }

    let entries: fs.Dirent[] = [];
    try {
      entries = fs.readdirSync(nextDir, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      const fullPath = path.join(nextDir, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }
      if (!entry.isFile()) {
        continue;
      }
      if (
        entry.name.startsWith(SESSION_FILE_PREFIX) &&
        entry.name.endsWith(SESSION_FILE_SUFFIX)
      ) {
        files.push(fullPath);
      }
    }
  }

  return files;
}

function parseSessionMetaLine(value: Record<string, unknown>): SessionMetaLine | null {
  const id = readTrimmedString(value.id);
  const title = sanitizeSessionTitle(readTrimmedString(value.title));
  const providerRaw = readTrimmedString(value.provider);
  const model = readTrimmedString(value.model);
  const thinkingRaw = readTrimmedString(value.thinkingLevel);
  const createdAtRaw = readTrimmedString(value.createdAt);
  const updatedAtRaw = readTrimmedString(value.updatedAt);
  const messageCountRaw = value.messageCount;

  if (!id || !title || !isAuthProvider(providerRaw) || !model || !isThinkingLevel(thinkingRaw)) {
    return null;
  }

  const createdAt = parseIsoDateOrNow(createdAtRaw);
  const updatedAt = parseIsoDateOrNow(updatedAtRaw || createdAtRaw);
  const messageCount =
    typeof messageCountRaw === "number" && Number.isFinite(messageCountRaw) && messageCountRaw >= 0
      ? Math.floor(messageCountRaw)
      : 0;
  const openRouterProvider =
    providerRaw === "openrouter"
      ? normalizeOpenRouterProvider(readTrimmedString(value.openRouterProvider))
      : undefined;

  return {
    type: "session_meta",
    id,
    title,
    provider: providerRaw,
    model,
    thinkingLevel: thinkingRaw,
    openRouterProvider,
    createdAt,
    updatedAt,
    messageCount,
  };
}

function normalizeChatMessage(message: ChatMessage): ChatMessage | null {
  if (message.role !== "user" && message.role !== "assistant") {
    return null;
  }
  const text = message.text.trim();
  if (!text) {
    return null;
  }
  return {
    role: message.role,
    text,
  };
}

function deriveSessionTitle(messages: ChatMessage[], titleHint?: string): string {
  const hint = sanitizeSessionTitle(titleHint ?? "");
  if (hint) {
    return hint;
  }

  const firstUserMessage = messages.find((message) => message.role === "user")?.text ?? "";
  const fromMessage = sanitizeSessionTitle(firstUserMessage);
  if (fromMessage) {
    return fromMessage;
  }

  return "untitled chat";
}

function sanitizeSessionTitle(value: string): string {
  const compact = value.replace(/\s+/g, " ").trim();
  if (!compact) {
    return "";
  }
  if (compact.length <= MAX_SESSION_TITLE_LENGTH) {
    return compact;
  }
  return `${compact.slice(0, MAX_SESSION_TITLE_LENGTH - 3).trim()}...`;
}

function normalizeOpenRouterProvider(value: string | undefined): string | undefined {
  const normalized = value?.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }
  return normalized;
}

function parseIsoDateOrNow(value: string): string {
  const time = Date.parse(value);
  if (Number.isFinite(time)) {
    return new Date(time).toISOString();
  }
  return new Date().toISOString();
}

function readTrimmedString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isAuthProvider(value: string): value is AuthProvider {
  return value === "openai" || value === "openrouter";
}

function isThinkingLevel(value: string): value is ThinkingLevel {
  return (
    value === "OFF" ||
    value === "MINIMAL" ||
    value === "LOW" ||
    value === "MEDIUM" ||
    value === "HIGH" ||
    value === "XHIGH"
  );
}
