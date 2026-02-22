import React, { useEffect, useMemo, useRef, useState } from "react";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { Box, Newline, render, Text, useApp, useInput } from "ink";
import TextInput from "ink-text-input";
import { loafConfig, type AuthProvider, type ThinkingLevel } from "./config.js";
import {
  loadPersistedRuntimeSecrets,
  clearPersistedConfig,
  loadPersistedState,
  persistRuntimeSecrets,
  savePersistedState,
  type LoafPersistedState,
} from "./persistence.js";
import { loadPersistedOpenAiChatgptAuth, runOpenAiOauthLogin } from "./openai-oauth.js";
import {
  fetchAntigravityProfileData,
  loadPersistedAntigravityOauthTokenInfo,
  runAntigravityOauthLogin,
  type AntigravityOauthProfile,
  type AntigravityOauthTokenInfo,
} from "./antigravity-oauth.js";
import {
  antigravityModelToThinkingLevels,
  discoverAntigravityModelOptions,
  fetchAntigravityUsageSnapshot,
  type AntigravityDiscoveredModel,
  type AntigravityUsageSnapshot,
} from "./antigravity-models.js";
import { runAntigravityInferenceStream } from "./antigravity.js";
import {
  createChatSession,
  listChatSessions,
  loadChatSession,
  loadChatSessionById,
  loadLatestChatSession,
  type ChatSessionRecord,
  type ChatSessionSummary,
  writeChatSession,
} from "./chat-history.js";
import {
  discoverOpenAiModelOptions,
  discoverOpenRouterModelOptions,
  getDefaultModelOptionsForProvider,
  modelIdToLabel,
  modelIdToSlug,
  type ModelOption,
} from "./models.js";
import { fetchOpenAiUsageSnapshot, runOpenAiInferenceStream, type OpenAiUsageSnapshot } from "./openai.js";
import { listOpenRouterProvidersForModel, runOpenRouterInferenceStream } from "./openrouter.js";
import { configureBuiltinTools, defaultToolRegistry, loadCustomTools } from "./tools/index.js";
import type { ChatImageAttachment, ChatMessage, DebugEvent, StreamChunk } from "./chat-types.js";
import { buildToolReplacement, consumeAssistantBoundary, parseToolCallPreview } from "./interleaving.js";
import {
  buildSkillPromptContext,
  loadSkillsCatalog,
  mapMessagesForModel,
  type SkillDefinition,
} from "./skills/index.js";
import { createInProcessRpcClient, type InProcessRpcClient } from "./rpc/inprocess-client.js";
import type { RuntimeEvent, RuntimeSessionState, RuntimeSnapshot } from "./core/runtime.js";

type UiMessage = {
  id: number;
  kind: "user" | "assistant" | "system";
  text: string;
  images?: ChatImageAttachment[];
};

type AuthOption = {
  id: AuthSelection;
  label: string;
  description: string;
};

type AuthSelection = AuthProvider | "antigravity";

type ThinkingOption = {
  id: string;
  label: string;
  description: string;
};

type OpenRouterProviderOption = {
  id: string;
  label: string;
  description: string;
};

type ProviderSwitchConfirmOption = {
  id: "switch_confirm" | "switch_cancel";
  label: string;
  description: string;
};

type CommandOption = {
  name: string;
  description: string;
};

type HistoryOption = {
  id: string;
  label: string;
  description: string;
  session: ChatSessionSummary;
};

type OnboardingOption = {
  id: "auth_openai" | "auth_openrouter" | "auth_continue";
  label: string;
  description: string;
};

type SelectorState =
  | {
    kind: "onboarding";
    title: string;
    index: number;
    options: OnboardingOption[];
  }
  | {
    kind: "auth";
    title: string;
    index: number;
    options: AuthOption[];
    returnToOnboarding?: boolean;
  }
  | {
    kind: "openrouter_api_key";
    title: string;
    returnToOnboarding?: boolean;
  }
  | {
    kind: "exa_api_key";
    title: string;
    returnToOnboarding?: boolean;
  }
  | {
    kind: "model";
    title: string;
    index: number;
    options: ModelOption[];
  }
  | {
    kind: "thinking";
    title: string;
    index: number;
    modelId: string;
    modelLabel: string;
    modelProvider: AuthProvider;
    options: ThinkingOption[];
  }
  | {
    kind: "openrouter_provider";
    title: string;
    index: number;
    modelId: string;
    modelLabel: string;
    modelProvider: AuthProvider;
    thinkingLevel: ThinkingLevel;
    options: OpenRouterProviderOption[];
  }
  | {
    kind: "history";
    title: string;
    index: number;
    options: HistoryOption[];
  }
  | {
    kind: "provider_switch_confirm";
    title: string;
    index: number;
    options: ProviderSwitchConfirmOption[];
    modelId: string;
    modelLabel: string;
    modelProvider: AuthProvider;
    thinkingLevel: ThinkingLevel;
    openRouterProvider?: string;
  };

type RpcModelListResponse = {
  providers: AuthProvider[];
  models: ModelOption[];
  selected_model: string;
  selected_thinking: ThinkingLevel;
  selected_provider: AuthProvider | null;
  selected_openrouter_provider: string;
};

type RpcLimitsResponse = {
  openai:
    | {
        ok: true;
        snapshot: OpenAiUsageSnapshot;
      }
    | {
        ok: false;
        message: string;
      }
    | null;
  antigravity:
    | {
        ok: true;
        snapshot: AntigravityUsageSnapshot;
      }
    | {
        ok: false;
        message: string;
      }
    | null;
};

type RpcHistoryListResponse = {
  total: number;
  cursor: number;
  limit: number;
  sessions: ChatSessionSummary[];
  next_cursor: number | null;
};

const AUTH_PROVIDER_ORDER: AuthProvider[] = ["openai", "openrouter"];
const TUI_RPC_DOGFOOD = true;

const THINKING_OPTION_DETAILS: Record<ThinkingLevel, { label: string; description: string }> = {
  OFF: { label: "off", description: "disable reasoning effort" },
  MINIMAL: { label: "minimal", description: "very light reasoning, fastest with thinking" },
  LOW: { label: "low", description: "fast with lighter reasoning" },
  MEDIUM: { label: "medium", description: "balanced depth and speed" },
  HIGH: { label: "high", description: "maximum reasoning depth" },
  XHIGH: { label: "xhigh", description: "extra high reasoning depth for hardest tasks" },
};

const THINKING_OPTIONS_OPENAI_DEFAULT: ThinkingOption[] = toThinkingOptions([
  "OFF",
  "MINIMAL",
  "LOW",
  "MEDIUM",
  "HIGH",
  "XHIGH",
]);
const THINKING_OPTIONS_OPENROUTER_DEFAULT: ThinkingOption[] = toThinkingOptions([
  "OFF",
  "MINIMAL",
  "LOW",
  "MEDIUM",
  "HIGH",
]);
const OPENROUTER_PROVIDER_ANY_ID = "any";

const COMMAND_OPTIONS: CommandOption[] = [
  { name: "/auth", description: "add auth provider" },
  { name: "/onboarding", description: "open setup flow (auth + exa key)" },
  { name: "/forgeteverything", description: "wipe local config and restart onboarding" },
  { name: "/model", description: "choose model and thinking level" },
  { name: "/limits", description: "show oauth usage limits" },
  { name: "/history", description: "resume a saved chat (/history, /history last, /history <id>)" },
  { name: "/skills", description: "list available skills from repo .agents/skills, ~/.loaf/skills, and ~/.agents/skills" },
  { name: "/tools", description: "list registered tools" },
  { name: "/clear", description: "clear conversation messages" },
  { name: "/quit", description: "exit loaf" },
  { name: "/help", description: "show available commands" },
  { name: "/exit", description: "exit loaf" },
];

const SUPER_DEBUG_COMMAND = "/superdebug-69";
const MAX_INPUT_HISTORY = 200;
const MAX_VISIBLE_MESSAGES = 14;
const MIN_VISIBLE_MESSAGE_ROWS = 2;
const BASE_LAYOUT_ROWS = 8;
const SELECTOR_WINDOW_SIZE = 10;
const MAX_PENDING_IMAGES = 4;
const MAX_IMAGE_FILE_BYTES = 8 * 1024 * 1024;
const CLIPBOARD_IMAGE_CAPTURE_MAX_BUFFER = MAX_IMAGE_FILE_BYTES + 512 * 1024;
const CLIPBOARD_OSASCRIPT_MAX_BUFFER = MAX_IMAGE_FILE_BYTES * 3;
const IMAGE_MIME_BY_EXT: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
};
// Keep user prefix ASCII to avoid terminal width ambiguity clipping tail characters.
const GLYPH_USER = "> ";
const GLYPH_ASSISTANT = "\u27E3 ";
const GLYPH_SYSTEM = "\u2301 ";
const SEARCH_WEB_PROMPT_EXTENSION = [
  "for facts that may be stale/uncertain (dates, releases, pricing, availability, docs), proactively use search_web.",
  "prefer at least one search_web pass before answering factual questions from memory.",
  "if search results are weak or conflicting, refine the query and search_web again before switching tools.",
  "for factual web lookups, call search_web first and use returned highlights before writing custom scrapers.",
].join("\n");
const OS_PROMPT_EXTENSION = buildOsPromptExtension();

function App() {
  const { exit } = useApp();
  const exitShortcutLabel = getExitShortcutLabel();
  const initialModelOptionsByProvider = useMemo(
    () =>
      ({
        openai: getDefaultModelOptionsForProvider("openai"),
        openrouter: getDefaultModelOptionsForProvider("openrouter"),
      }) satisfies Record<AuthProvider, ModelOption[]>,
    [],
  );
  const persistedState = useMemo(() => loadPersistedState(), []);
  const initialEnabledProviders = resolveInitialEnabledProviders({
    persistedProviders: persistedState?.authProviders,
    legacyProvider: persistedState?.authProvider,
    hasOpenAiToken: false,
    hasOpenRouterKey: Boolean((persistedState?.openRouterApiKey ?? loafConfig.openrouterApiKey).trim()),
  });
  const initialModel = resolveInitialModel(
    initialEnabledProviders,
    persistedState?.selectedModel,
    initialModelOptionsByProvider,
  );
  const initialModelProvider =
    findProviderForModel(
      initialModel,
      getModelOptionsForProviders(initialEnabledProviders, initialModelOptionsByProvider),
    ) ?? null;
  const initialThinking = normalizeThinkingForModel(
    initialModel,
    initialModelProvider,
    persistedState?.selectedThinking ?? loafConfig.thinkingLevel,
    initialModelOptionsByProvider,
  );
  const initialInputHistory = persistedState?.inputHistory ?? [];
  const initialExaApiKey = persistedState?.exaApiKey ?? loafConfig.exaApiKey;
  const initialOnboardingCompleted = resolveInitialOnboardingCompleted(persistedState);
  const initialSkillCatalog = useMemo(() => loadSkillsCatalog(), []);
  const [input, setInput] = useState("");
  const [textInputResetKey, setTextInputResetKey] = useState(0);
  const [autocompletedSkillPrefix, setAutocompletedSkillPrefix] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [statusLabel, setStatusLabel] = useState("ready");
  const [startupStatusLabel, setStartupStatusLabel] = useState("initializing...");
  const [superDebug, setSuperDebug] = useState(false);
  const [secretsHydrated, setSecretsHydrated] = useState(false);
  const [onboardingCompleted, setOnboardingCompleted] = useState<boolean>(initialOnboardingCompleted);
  const [enabledProviders, setEnabledProviders] = useState<AuthProvider[]>(initialEnabledProviders);
  const [openAiAccessToken, setOpenAiAccessToken] = useState("");
  const [openAiAccountId, setOpenAiAccountId] = useState<string | null>(null);
  const [antigravityOauthTokenInfo, setAntigravityOauthTokenInfo] =
    useState<AntigravityOauthTokenInfo | null>(null);
  const [antigravityOauthProfile, setAntigravityOauthProfile] =
    useState<AntigravityOauthProfile | null>(null);
  const [antigravityOpenAiModelOptions, setAntigravityOpenAiModelOptions] = useState<ModelOption[]>([]);
  const [openRouterApiKey, setOpenRouterApiKey] = useState(
    persistedState?.openRouterApiKey ?? loafConfig.openrouterApiKey,
  );
  const [exaApiKey, setExaApiKey] = useState(initialExaApiKey);
  const [selectedOpenRouterProvider, setSelectedOpenRouterProvider] = useState(
    persistedState?.selectedOpenRouterProvider ?? OPENROUTER_PROVIDER_ANY_ID,
  );
  const [modelOptionsByProvider, setModelOptionsByProvider] = useState<Record<AuthProvider, ModelOption[]>>(
    initialModelOptionsByProvider,
  );
  const [selectedModel, setSelectedModel] = useState(initialModel);
  const [selectedThinking, setSelectedThinking] = useState<ThinkingLevel>(initialThinking);
  const [conversationProvider, setConversationProvider] = useState<AuthProvider | null>(null);
  const [selector, setSelector] = useState<SelectorState | null>(null);
  const [commandIndex, setCommandIndex] = useState(0);
  const [skillIndex, setSkillIndex] = useState(0);
  const [availableSkills, setAvailableSkills] = useState<SkillDefinition[]>(initialSkillCatalog.skills);
  const [pendingImages, setPendingImages] = useState<ChatImageAttachment[]>([]);
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [activeSession, setActiveSession] = useState<ChatSessionSummary | null>(null);
  const [inputHistory, setInputHistory] = useState<string[]>(initialInputHistory);
  const [inputHistoryIndex, setInputHistoryIndex] = useState<number | null>(null);
  const [inputHistoryDraft, setInputHistoryDraft] = useState("");
  const [messages, setMessages] = useState<UiMessage[]>([]);
  // Keep local-only UI rows in a negative id range so they never collide with
  // runtime-emitted message ids (which are positive, server-generated ids).
  const nextIdRef = useRef(-1);
  const activeInferenceAbortControllerRef = useRef<AbortController | null>(null);
  const steeringQueueRef = useRef<ChatMessage[]>([]);
  const queuedPromptsRef = useRef<string[]>([]);
  const [queuedPromptsVersion, setQueuedPromptsVersion] = useState(0);
  const suppressNextSubmitRef = useRef(false);
  const recentSlashCommandRef = useRef<{ payload: string; at: number } | null>(null);
  const statusCommandInFlightRef = useRef(false);
  const previousAntigravityModelIdsRef = useRef<Set<string>>(new Set());
  const skipNextAntigravitySyncTokenRef = useRef<string | null>(null);
  const suppressCtrlVEchoRef = useRef<{
    active: boolean;
    previousInput: string;
    timeout: ReturnType<typeof setTimeout> | null;
  }>({
    active: false,
    previousInput: "",
    timeout: null,
  });
  const rpcClientRef = useRef<InProcessRpcClient | null>(null);
  const rpcSessionIdRef = useRef("");
  const rpcEventUnsubscribeRef = useRef<(() => void) | null>(null);
  const streamingAssistantIdRef = useRef<number | null>(null);
  const streamingAssistantTextRef = useRef("");
  const pendingToolCallMessageIdsRef = useRef<number[]>([]);

  const nextMessageId = () => {
    const id = nextIdRef.current;
    nextIdRef.current -= 1;
    return id;
  };

  const requireRpcClient = (): InProcessRpcClient => {
    const client = rpcClientRef.current;
    if (!client) {
      throw new Error("rpc client is not ready");
    }
    return client;
  };

  const getRpcSessionId = (): string => {
    const sessionId = rpcSessionIdRef.current.trim();
    if (!sessionId) {
      throw new Error("rpc session is not ready");
    }
    return sessionId;
  };

  const callRpc = async <T,>(method: string, params?: unknown): Promise<T> => {
    const client = requireRpcClient();
    return client.call<T>(method, params);
  };

  const toUiMessage = (message: RuntimeSessionState["messages"][number]): UiMessage => ({
    id: message.id,
    kind: message.kind,
    text: message.text,
    images: message.images,
  });

  const applyRuntimeSnapshot = (snapshot: RuntimeSnapshot) => {
    setEnabledProviders(snapshot.auth.enabledProviders);
    setOpenAiAccessToken(snapshot.auth.hasOpenAiToken ? "__rpc_openai__" : "");
    setOpenAiAccountId(null);
    setAntigravityOauthTokenInfo(
      snapshot.auth.hasAntigravityToken
        ? {
            accessToken: "__rpc_antigravity__",
            refreshToken: "",
            expiryDateSeconds: Number.MAX_SAFE_INTEGER,
            tokenType: "bearer",
          }
        : null,
    );
    setAntigravityOauthProfile(snapshot.auth.antigravityProfile);
    setOpenRouterApiKey(snapshot.auth.hasOpenRouterKey ? "__rpc_openrouter__" : "");
    setOnboardingCompleted(snapshot.onboarding.completed);
    setSelectedModel(snapshot.model.selectedModel);
    setSelectedThinking(snapshot.model.selectedThinking);
    setSelectedOpenRouterProvider(snapshot.model.selectedOpenRouterProvider);
  };

  const applyModelList = (payload: RpcModelListResponse) => {
    const openai = payload.models.filter((option) => option.provider === "openai");
    const openrouter = payload.models.filter((option) => option.provider === "openrouter");
    setModelOptionsByProvider({
      openai: openai.length > 0 ? openai : getDefaultModelOptionsForProvider("openai"),
      openrouter: openrouter.length > 0 ? openrouter : getDefaultModelOptionsForProvider("openrouter"),
    });
    setAntigravityOpenAiModelOptions(
      openai.filter((option) => (option.displayProvider ?? "").trim().toLowerCase() === "antigravity"),
    );
    setSelectedModel(payload.selected_model);
    setSelectedThinking(payload.selected_thinking);
    setSelectedOpenRouterProvider(payload.selected_openrouter_provider);
  };

  const applyRuntimeSessionState = (state: RuntimeSessionState) => {
    setPending(state.pending);
    setStatusLabel(state.statusLabel || (state.pending ? "working..." : "ready"));
    setMessages(state.messages.map((message) => toUiMessage(message)));
    pendingToolCallMessageIdsRef.current = [];
    setHistory(state.history);
    setConversationProvider(state.conversationProvider);
  };

  const refreshRuntimeState = async () => {
    const snapshot = await callRpc<RuntimeSnapshot>("state.get");
    applyRuntimeSnapshot(snapshot);
    const modelList = await callRpc<RpcModelListResponse>("model.list", {});
    applyModelList(modelList);
  };

  const refreshRuntimeSessionState = async () => {
    const sessionId = getRpcSessionId();
    const state = await callRpc<RuntimeSessionState>("session.get", {
      session_id: sessionId,
    });
    applyRuntimeSessionState(state);
  };

  const hasOpenAiToken = openAiAccessToken.trim().length > 0;
  const hasOpenRouterKey = openRouterApiKey.trim().length > 0;
  const antigravityAccessToken = (antigravityOauthTokenInfo?.accessToken ?? "").trim();
  const hasAntigravityToken = antigravityAccessToken.length > 0;
  const selectableModelProviders = useMemo(
    () => getSelectableModelProviders(enabledProviders, hasAntigravityToken),
    [enabledProviders, hasAntigravityToken],
  );
  const activeModelOptions = useMemo(
    () => getModelOptionsForProviders(selectableModelProviders, modelOptionsByProvider),
    [selectableModelProviders, modelOptionsByProvider],
  );
  const selectedModelProvider = useMemo(
    () => findProviderForModel(selectedModel, activeModelOptions) ?? null,
    [selectedModel, activeModelOptions],
  );
  const authSummary = useMemo(() => {
    const connected = [...enabledProviders] as string[];
    if (hasAntigravityToken) {
      connected.push("antigravity");
    }
    return connected.length > 0 ? connected.join(", ") : "not selected";
  }, [enabledProviders, hasAntigravityToken]);
  const authOptions = useMemo(
    () =>
      buildAuthOptions({
        hasOpenAi: hasOpenAiToken,
        hasOpenRouter: hasOpenRouterKey,
        hasAntigravity: hasAntigravityToken,
        includeAntigravity: true,
      }),
    [hasAntigravityToken, hasOpenAiToken, hasOpenRouterKey],
  );
  const providerAuthOptions = useMemo(
    () =>
      buildAuthOptions({
        hasOpenAi: hasOpenAiToken,
        hasOpenRouter: hasOpenRouterKey,
        hasAntigravity: hasAntigravityToken,
        includeAntigravity: false,
      }),
    [hasAntigravityToken, hasOpenAiToken, hasOpenRouterKey],
  );

  const commandSuggestions = useMemo(() => {
    if (!input.startsWith("/")) {
      return [] as CommandOption[];
    }
    const query = input.trim().toLowerCase();
    if (!query) {
      return COMMAND_OPTIONS;
    }
    return COMMAND_OPTIONS.filter((command) => command.name.startsWith(query));
  }, [input]);

  const activeSkillMention = useMemo(() => findActiveSkillMentionToken(input), [input]);
  const skillSuggestions = useMemo(() => {
    if (!activeSkillMention) {
      return [] as SkillDefinition[];
    }

    const query = activeSkillMention.query;
    if (!query) {
      return availableSkills;
    }
    return availableSkills.filter((skill) => skill.nameLower.startsWith(query));
  }, [activeSkillMention, availableSkills]);

  const suppressSkillSuggestions = Boolean(
    autocompletedSkillPrefix && input.startsWith(autocompletedSkillPrefix),
  );
  const showSkillSuggestions =
    Boolean(activeSkillMention) && skillSuggestions.length > 0 && !suppressSkillSuggestions;
  const showCommandSuggestionPanel = commandSuggestions.length > 0 && !selector;
  const showSkillSuggestionPanel = showSkillSuggestions && !selector;

  useEffect(() => {
    setCommandIndex((current) => {
      if (commandSuggestions.length === 0) {
        return 0;
      }
      return Math.min(current, commandSuggestions.length - 1);
    });
  }, [commandSuggestions.length]);

  useEffect(() => {
    setSkillIndex((current) => {
      if (skillSuggestions.length === 0) {
        return 0;
      }
      return Math.min(current, skillSuggestions.length - 1);
    });
  }, [skillSuggestions.length]);

  useEffect(() => {
    if (!activeSkillMention) {
      return;
    }
    const catalog = loadSkillsCatalog();
    setAvailableSkills(catalog.skills);
  }, [activeSkillMention]);

  useEffect(() => {
    if (autocompletedSkillPrefix && !input.startsWith(autocompletedSkillPrefix)) {
      setAutocompletedSkillPrefix(null);
    }
  }, [input, autocompletedSkillPrefix]);

  useEffect(() => {
    return () => {
      if (suppressCtrlVEchoRef.current.timeout) {
        clearTimeout(suppressCtrlVEchoRef.current.timeout);
        suppressCtrlVEchoRef.current.timeout = null;
      }
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    const appendRuntimeSystemMessage = (text: string) => {
      setMessages((current) => [
        ...current,
        {
          id: nextMessageId(),
          kind: "system",
          text,
        },
      ]);
    };

    const appendStreamingDelta = (delta: string) => {
      if (!delta) {
        return;
      }

      const existingId = streamingAssistantIdRef.current;
      if (existingId === null) {
        const messageId = nextMessageId();
        streamingAssistantIdRef.current = messageId;
        streamingAssistantTextRef.current = delta;
        setMessages((current) => [
          ...current,
          {
            id: messageId,
            kind: "assistant",
            text: delta,
          },
        ]);
        return;
      }

      streamingAssistantTextRef.current += delta;
      const nextText = streamingAssistantTextRef.current;
      setMessages((current) =>
        current.map((message) =>
          message.id === existingId
            ? {
                ...message,
                text: nextText,
              }
            : message,
        ),
      );
    };

    const finalizeStreamingAssistant = (text: string) => {
      const existingId = streamingAssistantIdRef.current;
      if (existingId === null) {
        setMessages((current) => [
          ...current,
          {
            id: nextMessageId(),
            kind: "assistant",
            text,
          },
        ]);
        return;
      }

      setMessages((current) =>
        current.map((message) =>
          message.id === existingId
            ? {
                ...message,
                text,
              }
            : message,
        ),
      );
      streamingAssistantIdRef.current = null;
      streamingAssistantTextRef.current = "";
    };

    const handleRuntimeEvent = (event: RuntimeEvent) => {
      if (cancelled) {
        return;
      }

      const payload = event.payload ?? {};
      const payloadSessionId = typeof payload.session_id === "string" ? payload.session_id : "";
      if (payloadSessionId && payloadSessionId !== rpcSessionIdRef.current) {
        return;
      }

      if (event.type === "session.status") {
        const nextPending = payload.pending === true;
        const label = typeof payload.status_label === "string" ? payload.status_label : nextPending ? "working..." : "ready";
        setPending(nextPending);
        setStatusLabel(label);
        if (!nextPending) {
          streamingAssistantIdRef.current = null;
          streamingAssistantTextRef.current = "";
        }
        return;
      }

      if (event.type === "session.stream.chunk") {
        // Runtime emits final assistant rows via session.message.appended.
        // Skip chunk-level rendering to avoid duplicate deltas.
        return;
      }

      if (event.type === "session.message.appended") {
        const message = payload.message as RuntimeSessionState["messages"][number] | undefined;
        if (!message) {
          return;
        }
        if (message.kind === "assistant") {
          finalizeStreamingAssistant(message.text);
        } else {
          setMessages((current) => [...current, toUiMessage(message)]);
        }
        const historyRole =
          message.kind === "user"
            ? "user"
            : message.kind === "assistant"
              ? "assistant"
              : null;
        if (historyRole) {
          setHistory((current) => [
            ...current,
            {
              role: historyRole,
              text: message.text,
              images: message.images,
            },
          ]);
        }
        return;
      }

      if (event.type === "session.tool.call.started") {
        const row = formatToolStartRow(payload.data);
        if (row) {
          const messageId = nextMessageId();
          setMessages((current) => [
            ...current,
            {
              id: messageId,
              kind: "system",
              text: row,
            },
          ]);
          pendingToolCallMessageIdsRef.current.push(messageId);
        }
        return;
      }

      if (event.type === "session.tool.call.completed") {
        const row = formatToolCompletedRow(payload.data);
        if (row) {
          const pendingId = pendingToolCallMessageIdsRef.current.shift();
          if (typeof pendingId === "number") {
            setMessages((current) =>
              current.map((message) =>
                message.id === pendingId
                  ? {
                      ...message,
                      text: row,
                    }
                  : message,
              ),
            );
          } else {
            appendRuntimeSystemMessage(row);
          }
        }
        return;
      }

      if (event.type === "session.tool.results") {
        // `session.tool.call.completed` already renders per-call completion rows.
        // Skip aggregate rows here to avoid duplicate tool output in transcript.
        return;
      }

      if (event.type === "auth.flow.url") {
        const provider = typeof payload.provider === "string" ? payload.provider : "auth";
        const url = typeof payload.url === "string" ? payload.url : "";
        if (url) {
          appendRuntimeSystemMessage(`${provider} auth url: ${url}`);
        }
        return;
      }

      if (event.type === "auth.flow.device_code") {
        const verificationUrl = typeof payload.verification_url === "string" ? payload.verification_url : "";
        const userCode = typeof payload.user_code === "string" ? payload.user_code : "";
        const expiresInSeconds =
          typeof payload.expires_in_seconds === "number" && Number.isFinite(payload.expires_in_seconds)
            ? payload.expires_in_seconds
            : 0;
        const expiresMinutes = Math.max(1, Math.ceil(expiresInSeconds / 60));
        if (verificationUrl) {
          appendRuntimeSystemMessage(`open ${verificationUrl}`);
        }
        if (userCode) {
          appendRuntimeSystemMessage(`enter code: ${userCode} (expires in ~${expiresMinutes} min)`);
        }
        return;
      }

      if (event.type === "auth.flow.completed") {
        const provider = typeof payload.provider === "string" ? payload.provider : "auth";
        appendRuntimeSystemMessage(`${provider} auth complete.`);
        void refreshRuntimeState();
        return;
      }

      if (event.type === "auth.flow.failed") {
        const provider = typeof payload.provider === "string" ? payload.provider : "auth";
        const message = typeof payload.message === "string" ? payload.message : "unknown error";
        appendRuntimeSystemMessage(`${provider} auth failed: ${message}`);
        return;
      }

      if (event.type === "state.changed") {
        const snapshot = payload.snapshot as RuntimeSnapshot | undefined;
        if (snapshot) {
          applyRuntimeSnapshot(snapshot);
        } else {
          void refreshRuntimeState();
        }
      }
    };

    void (async () => {
      try {
        setStartupStatusLabel("starting rpc runtime...");
        const rpcClient = await createInProcessRpcClient();
        if (cancelled) {
          await rpcClient.call("system.shutdown", {
            reason: "tui_cancelled",
          });
          return;
        }

        rpcClientRef.current = rpcClient;
        await rpcClient.call("rpc.handshake", {
          client_name: "loaf_tui",
          client_version: "dev",
          protocol_version: "1.0.0",
        });

        const created = await rpcClient.call<{
          session_id: string;
          state: RuntimeSessionState;
        }>("session.create", {
          title: "tui",
        });

        rpcSessionIdRef.current = created.session_id;
        applyRuntimeSessionState(created.state);

        const unsubscribe = rpcClient.onEvent(handleRuntimeEvent);
        rpcEventUnsubscribeRef.current = unsubscribe;

        await refreshRuntimeState();
        await refreshRuntimeSessionState();
        if (!cancelled) {
          setStartupStatusLabel("ready");
          setSecretsHydrated(true);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (!cancelled) {
          setMessages((current) => [
            ...current,
            {
              id: nextMessageId(),
              kind: "system",
              text: `rpc startup failed: ${message}`,
            },
          ]);
          setStartupStatusLabel("startup failed");
          setSecretsHydrated(true);
        }
      }
    })();

    return () => {
      cancelled = true;
      const unsubscribe = rpcEventUnsubscribeRef.current;
      rpcEventUnsubscribeRef.current = null;
      unsubscribe?.();
      const rpcClient = rpcClientRef.current;
      rpcClientRef.current = null;
      if (rpcClient) {
        void rpcClient.call("system.shutdown", {
          reason: "tui_exit",
        }).catch(() => undefined);
      }
    };
  }, []);

  useEffect(() => {
    if (TUI_RPC_DOGFOOD) {
      return;
    }
    configureBuiltinTools({
      exaApiKey,
    });
  }, [exaApiKey]);

  useEffect(() => {
    if (TUI_RPC_DOGFOOD) {
      return;
    }
    if (!secretsHydrated) {
      return;
    }
    void persistRuntimeSecrets({
      openRouterApiKey,
      exaApiKey,
    });
  }, [openRouterApiKey, exaApiKey, secretsHydrated]);

  useEffect(() => {
    if (TUI_RPC_DOGFOOD) {
      return;
    }
    savePersistedState({
      authProviders: enabledProviders,
      selectedModel,
      selectedThinking,
      selectedOpenRouterProvider,
      onboardingCompleted,
      inputHistory,
    });
  }, [
    enabledProviders,
    selectedModel,
    selectedThinking,
    selectedOpenRouterProvider,
    onboardingCompleted,
    inputHistory,
  ]);

  useEffect(() => {
    if (TUI_RPC_DOGFOOD) {
      return;
    }
    let cancelled = false;

    if (!antigravityAccessToken) {
      setAntigravityOpenAiModelOptions([]);
      skipNextAntigravitySyncTokenRef.current = null;
      return;
    }

    if (skipNextAntigravitySyncTokenRef.current === antigravityAccessToken) {
      skipNextAntigravitySyncTokenRef.current = null;
      return;
    }

    void (async () => {
      try {
        const result = await discoverAntigravityModelOptions({
          accessToken: antigravityAccessToken,
        });
        if (cancelled) {
          return;
        }

        setAntigravityOpenAiModelOptions(toAntigravityOpenAiModelOptions(result.models));
      } catch (error) {
        if (!cancelled) {
          setAntigravityOpenAiModelOptions([]);
          const message = error instanceof Error ? error.message : String(error);
          appendAntigravityModelSyncFailureMessages(appendSystemMessage, message);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [antigravityAccessToken]);

  useEffect(() => {
    if (TUI_RPC_DOGFOOD) {
      return;
    }
    setModelOptionsByProvider((current) => {
      const previousIds = previousAntigravityModelIdsRef.current;
      const baseOpenAi = current.openai.filter((option) => !previousIds.has(option.id));
      previousAntigravityModelIdsRef.current = new Set(antigravityOpenAiModelOptions.map((option) => option.id));

      return {
        ...current,
        openai: mergeOpenAiAndAntigravityModels(
          baseOpenAi.length > 0 ? baseOpenAi : getDefaultModelOptionsForProvider("openai"),
          antigravityOpenAiModelOptions,
        ),
      };
    });
  }, [antigravityOpenAiModelOptions]);

  useEffect(() => {
    if (TUI_RPC_DOGFOOD) {
      return;
    }
    let cancelled = false;
    void (async () => {
      for (const provider of enabledProviders) {
        if (provider === "openai") {
          if (!openAiAccessToken.trim()) {
            continue;
          }

          const result = await discoverOpenAiModelOptions({
            accessToken: openAiAccessToken,
            chatgptAccountId: openAiAccountId,
          });
          if (cancelled) {
            return;
          }

          setModelOptionsByProvider((current) => ({
            ...current,
            openai: mergeOpenAiAndAntigravityModels(result.models, antigravityOpenAiModelOptions),
          }));
          continue;
        }

        if (!openRouterApiKey.trim()) {
          continue;
        }

        const result = await discoverOpenRouterModelOptions({
          apiKey: openRouterApiKey,
        });
        if (cancelled) {
          return;
        }

        setModelOptionsByProvider((current) => ({
          ...current,
          openrouter: result.models,
        }));
      }

    })();

    return () => {
      cancelled = true;
    };
  }, [enabledProviders, openAiAccessToken, openAiAccountId, openRouterApiKey]);

  useEffect(() => {
    if (!secretsHydrated) {
      return;
    }
    setSelectedModel((currentModel) =>
      resolveModelForEnabledProviders(selectableModelProviders, currentModel, modelOptionsByProvider),
    );
  }, [selectableModelProviders, modelOptionsByProvider, secretsHydrated]);

  useEffect(() => {
    const provider = findProviderForModel(selectedModel, activeModelOptions);
    if (!provider) {
      return;
    }

    setSelectedThinking((currentThinking) =>
      normalizeThinkingForModel(selectedModel, provider, currentThinking, modelOptionsByProvider),
    );
  }, [selectedModel, activeModelOptions, modelOptionsByProvider]);

  const appendSystemMessage = (text: string) => {
    setMessages((current) => [
      ...current,
      {
        id: nextMessageId(),
        kind: "system",
        text,
      },
    ]);
  };

  const refreshSkillsCatalog = () => {
    const catalog = loadSkillsCatalog();
    setAvailableSkills(catalog.skills);
    return catalog;
  };

  const appendDebugEvent = (event: DebugEvent) => {
    if (!superDebug) {
      return;
    }

    const payload = safeJsonStringify(event.data);
    appendSystemMessage(
      `[debug] ${event.stage}\n${payload}`,
    );
  };

  const queueSteeringMessage = (rawText: string): boolean => {
    const text = rawText.trim();
    if (!text) {
      return false;
    }
    void (async () => {
      try {
        const response = await callRpc<{ session_id: string; accepted: boolean }>("session.steer", {
          session_id: getRpcSessionId(),
          text,
        });
        if (!response.accepted) {
          appendSystemMessage("steer rejected: no active inference.");
          return;
        }
        setInput("");
        setStatusLabel("steer queued...");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`steer failed: ${message}`);
      }
    })();
    return true;
  };

  const queuePendingPrompt = (rawText: string): boolean => {
    const text = rawText.trim();
    if (!text) {
      return false;
    }
    void (async () => {
      try {
        await callRpc("session.send", {
          session_id: getRpcSessionId(),
          text,
          enqueue: true,
        });
        const queue = await callRpc<{ session_id: string; queue: Array<{ id: string }> }>("session.queue.list", {
          session_id: getRpcSessionId(),
        });
        setQueuedPromptsVersion((current) => current + 1);
        setInput("");
        appendSystemMessage(`queued message (${queue.queue.length}): ${clipInline(text, 80)}`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`queue failed: ${message}`);
      }
    })();
    return true;
  };

  const queueImageAttachment = (image: ChatImageAttachment): boolean => {
    if (pendingImages.some((existing) => existing.path === image.path)) {
      appendSystemMessage(`image already attached: ${path.basename(image.path)}`);
      return false;
    }
    if (pendingImages.length >= MAX_PENDING_IMAGES) {
      appendSystemMessage(`image limit reached (${MAX_PENDING_IMAGES}). send or /clear first.`);
      return false;
    }

    const placeholder = `[Image ${pendingImages.length + 1}]`;
    setPendingImages((current) => [...current, image]);
    setInput((current) => {
      const base = suppressCtrlVEchoRef.current.active
        ? consumeCtrlVEchoArtifact(current, suppressCtrlVEchoRef.current.previousInput)
        : current;
      return appendImagePlaceholderToComposerInput(base, placeholder);
    });
    // Keep cursor at the end (right after the inserted image placeholder).
    setTextInputResetKey((current) => current + 1);
    appendSystemMessage(
      `image attached: ${path.basename(image.path)} (${image.mimeType}, ${formatByteSize(image.byteSize)})`,
    );
    return true;
  };

  const interruptActiveInference = (): boolean => {
    if (!pending) {
      return false;
    }
    void (async () => {
      try {
        const response = await callRpc<{ interrupted: boolean }>("session.interrupt", {
          session_id: getRpcSessionId(),
        });
        if (response.interrupted) {
          setStatusLabel("interrupting...");
          return;
        }
        appendSystemMessage("no active inference to interrupt.");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`interrupt failed: ${message}`);
      }
    })();
    return true;
  };

  const openOnboardingSelector = () => {
    setInput("");
    setSelector({
      kind: "onboarding",
      title: "onboarding 1/2 - auth providers",
      index: 0,
      options: buildOnboardingAuthOptions(enabledProviders),
    });
  };

  useEffect(() => {
    if (!secretsHydrated) {
      return;
    }
    if (selector) {
      return;
    }
    if (!onboardingCompleted) {
      setSelector({
        kind: "onboarding",
        title: "onboarding 1/2 - auth providers",
        index: 0,
        options: buildOnboardingAuthOptions(enabledProviders),
      });
      return;
    }
    if (enabledProviders.length === 0) {
      setSelector({
        kind: "auth",
        title: "select auth provider",
        index: 0,
        options: providerAuthOptions,
      });
    }
  }, [enabledProviders, exaApiKey, onboardingCompleted, providerAuthOptions, secretsHydrated, selector]);

  const openAuthSelector = (returnToOnboarding = false, providerOnly = false) => {
    const options = returnToOnboarding || providerOnly ? providerAuthOptions : authOptions;
    setInput("");
    const firstMissing = options.find(
      (option) =>
        !isAuthSelectionConfigured(option.id, {
          hasOpenAi: hasOpenAiToken,
          hasOpenRouter: hasOpenRouterKey,
          hasAntigravity: hasAntigravityToken,
        }),
    );
    const defaultIndex = firstMissing
      ? options.findIndex((option) => option.id === firstMissing.id)
      : 0;
    setSelector({
      kind: "auth",
      title: "add auth provider",
      index: defaultIndex,
      options,
      returnToOnboarding,
    });
  };

  const openExaApiKeySelector = (returnToOnboarding = false) => {
    setInput("");
    setSelector({
      kind: "exa_api_key",
      title: "onboarding 2/2 - exa api key (or type 'skip')",
      returnToOnboarding,
    });
  };

  const applyAuthSelection = async (
    selection: AuthSelection,
    openRouterKeyOverride?: string,
    returnToOnboarding = false,
  ): Promise<boolean> => {
    if (selection === "antigravity") {
      setPending(true);
      setStatusLabel("starting antigravity oauth...");
      appendSystemMessage(
        hasAntigravityToken ? "restarting antigravity oauth login..." : "starting antigravity oauth login...",
      );
      try {
        await callRpc("auth.connect.antigravity", {});
        await refreshRuntimeState();
        appendSystemMessage("antigravity oauth login complete.");
        return true;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`antigravity oauth login failed: ${message}`);
        return false;
      } finally {
        setPending(false);
        setStatusLabel("ready");
      }
    }

    const provider = selection;
    if (provider === "openrouter") {
      const key = (openRouterKeyOverride ?? openRouterApiKey).trim();
      if (!key) {
        setInput("");
        setSelector({
          kind: "openrouter_api_key",
          title: "enter openrouter api key",
          returnToOnboarding,
        });
        appendSystemMessage("enter your openrouter api key and press enter.");
        return false;
      }

      try {
        const result = await callRpc<{
          configured: boolean;
          model_count: number;
          source: string;
        }>("auth.set.openrouter_key", {
          api_key: key,
        });
        await refreshRuntimeState();
        appendSystemMessage(`openrouter models synced: ${result.model_count} (${result.source})`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`openrouter model sync failed: ${message}`);
        return false;
      }
      appendSystemMessage("openrouter auth configured.");
      return true;
    } else {
      setPending(true);
      setStatusLabel("starting oauth login...");
      appendSystemMessage("starting chatgpt account login...");
      try {
        await callRpc("auth.connect.openai", {
          mode: "auto",
          originator: "tui",
        });
        await refreshRuntimeState();
        appendSystemMessage("chatgpt oauth login complete.");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`chatgpt oauth login failed: ${message}`);
        return false;
      } finally {
        setPending(false);
        setStatusLabel("ready");
      }
      return true;
    }
  };

  const applyExaApiKeySelection = async (rawValue: string): Promise<"saved" | "skipped" | "invalid"> => {
    const value = rawValue.trim();
    if (!value) {
      return "invalid";
    }

    try {
      await callRpc("auth.set.exa_key", {
        api_key: value,
      });
      await refreshRuntimeState();
      if (value.toLowerCase() === "skip") {
        appendSystemMessage("exa api key skipped. search_web will be unavailable.");
        return "skipped";
      }
      appendSystemMessage("exa api key saved. search_web is now available.");
      return "saved";
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      appendSystemMessage(`failed to save exa api key: ${message}`);
      return "invalid";
    }
  };

  const openModelSelector = () => {
    void refreshRuntimeState().catch(() => undefined);
    if (!onboardingCompleted) {
      openOnboardingSelector();
      return;
    }
    if (selectableModelProviders.length === 0) {
      openAuthSelector(false, true);
      return;
    }

    setInput("");
    const modelOptions = getModelOptionsForProviders(selectableModelProviders, modelOptionsByProvider);
    const selectedIndex = Math.max(
      0,
      modelOptions.findIndex((option) => option.id === selectedModel),
    );
    setSelector({
      kind: "model",
      title: "select model",
      index: selectedIndex,
      options: modelOptions,
    });
  };

  const getModelSelectorOptions = (currentSelector: SelectorState | null): ModelOption[] => {
    if (!currentSelector || currentSelector.kind !== "model") {
      return [];
    }
    const query = input.trim().toLowerCase();
    if (!query) {
      return currentSelector.options;
    }
    return currentSelector.options.filter((option) => {
      const label = option.label.toLowerCase();
      const id = option.id.toLowerCase();
      const description = option.description.toLowerCase();
      return label.includes(query) || id.includes(query) || description.includes(query);
    });
  };

  const getSelectorAllOptions = (
    currentSelector: SelectorState,
  ): Array<
    | AuthOption
    | ModelOption
    | ThinkingOption
    | OpenRouterProviderOption
    | OnboardingOption
    | HistoryOption
    | ProviderSwitchConfirmOption
  > => {
    if (
      currentSelector.kind === "openrouter_api_key" ||
      currentSelector.kind === "exa_api_key"
    ) {
      return [];
    }
    if (currentSelector.kind === "model") {
      return getModelSelectorOptions(currentSelector);
    }
    return currentSelector.options;
  };

  const getSelectorOptionWindow = (
    options: Array<
      | AuthOption
      | ModelOption
      | ThinkingOption
      | OpenRouterProviderOption
      | OnboardingOption
      | HistoryOption
      | ProviderSwitchConfirmOption
    >,
    index: number,
  ) => {
    if (options.length <= SELECTOR_WINDOW_SIZE) {
      return {
        startIndex: 0,
        activeIndex: Math.max(0, Math.min(index, Math.max(0, options.length - 1))),
        options,
      };
    }

    const activeIndex = Math.max(0, Math.min(index, options.length - 1));
    const halfWindow = Math.floor(SELECTOR_WINDOW_SIZE / 2);
    const maxStart = Math.max(0, options.length - SELECTOR_WINDOW_SIZE);
    const startIndex = Math.max(0, Math.min(activeIndex - halfWindow, maxStart));
    return {
      startIndex,
      activeIndex,
      options: options.slice(startIndex, startIndex + SELECTOR_WINDOW_SIZE),
    };
  };

  const applyModelSelection = async (params: {
    modelId: string;
    modelLabel: string;
    modelProvider: AuthProvider;
    thinkingLevel: ThinkingLevel;
    openRouterProvider?: string;
    bypassProviderSwitchWarning?: boolean;
  }): Promise<void> => {
    const providerChanged =
      selectedModelProvider !== null && selectedModelProvider !== params.modelProvider;
    const hasContextToLose = history.length > 0;

    if (providerChanged && hasContextToLose && !params.bypassProviderSwitchWarning) {
      const fromProvider = selectedModelProvider ?? "unknown";
      const switchOptions: ProviderSwitchConfirmOption[] = [
        {
          id: "switch_confirm",
          label: "switch and clear context",
          description: "continue with the new provider and clear current conversation context",
        },
        {
          id: "switch_cancel",
          label: "cancel",
          description: "keep current provider and current context",
        },
      ];
      setInput("");
      setSelector({
        kind: "provider_switch_confirm",
        title: `warning: switching ${fromProvider} -> ${params.modelProvider} will clear conversation context`,
        index: 0,
        options: switchOptions,
        modelId: params.modelId,
        modelLabel: params.modelLabel,
        modelProvider: params.modelProvider,
        thinkingLevel: params.thinkingLevel,
        openRouterProvider: params.openRouterProvider,
      });
      return;
    }

    setPending(true);
    setStatusLabel("updating model...");
    try {
      await callRpc("model.select", {
        model_id: params.modelId,
        provider: params.modelProvider,
        thinking_level: params.thinkingLevel,
        openrouter_provider: params.openRouterProvider,
      });
      await refreshRuntimeState();
      if (providerChanged) {
        await callRpc("history.clear_session", {
          session_id: getRpcSessionId(),
        });
        await refreshRuntimeSessionState();
      }

      const providerSuffix =
        params.modelProvider === "openrouter"
          ? ` | provider ${params.openRouterProvider ?? OPENROUTER_PROVIDER_ANY_ID}`
          : "";
      const selectedOption = findModelOption(params.modelProvider, params.modelId, modelOptionsByProvider);
      const isAntigravityModel = (selectedOption?.displayProvider ?? "").trim().toLowerCase() === "antigravity";
      appendSystemMessage(
        `model updated: ${params.modelLabel}${isAntigravityModel ? "" : ` (${params.thinkingLevel.toLowerCase()})`}${providerSuffix}${providerChanged ? " - context reset for provider switch" : ""}`,
      );
      setInput("");
      setSelector(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      appendSystemMessage(`model update failed: ${message}`);
    } finally {
      setPending(false);
      setStatusLabel("ready");
    }
  };

  const openHistorySelector = async () => {
    try {
      const history = await callRpc<RpcHistoryListResponse>("history.list", {
        limit: 100,
        cursor: 0,
      });
      const sessions = history.sessions;
      if (sessions.length === 0) {
        appendSystemMessage("no saved chat history yet.");
        return;
      }

      const options: HistoryOption[] = sessions.map((session) => ({
        id: session.id,
        label: session.title,
        description:
          `${session.provider} | ${modelIdToLabel(session.model).toLowerCase()} | ` +
          `${session.messageCount} msgs | ${formatSessionTimestamp(session.updatedAt)} | ${session.id.slice(0, 8)}`,
        session,
      }));

      setInput("");
      setSelector({
        kind: "history",
        title: "resume chat history",
        index: 0,
        options,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      appendSystemMessage(`history list failed: ${message}`);
    }
  };

  const runHistoryCommand = async (args: string[]) => {
    if (args.length === 0 || args[0] === "list" || args[0] === "all") {
      await openHistorySelector();
      return;
    }

    const first = args[0]!.toLowerCase();
    try {
      await callRpc("command.execute", {
        session_id: getRpcSessionId(),
        raw_command: first === "last" ? "/history last" : `/history ${args[0]}`,
      });
      await refreshRuntimeState();
      await refreshRuntimeSessionState();
      setSelector(null);
      setInput("");
      appendSystemMessage(`resumed chat ${args[0] ?? "last"}.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      appendSystemMessage(message);
      void openHistorySelector();
    }
  };

  const runStatusCommand = async () => {
    if (statusCommandInFlightRef.current) {
      return;
    }

    const openAiToken = openAiAccessToken.trim();
    const hasAnyLimitsProvider = Boolean(openAiToken || antigravityAccessToken);
    if (!hasAnyLimitsProvider) {
      appendSystemMessage("limits requires at least one oauth provider. run /auth first.");
      return;
    }

    statusCommandInFlightRef.current = true;
    setPending(true);
    setStatusLabel("fetching oauth usage...");
    try {
      const limits = await callRpc<RpcLimitsResponse>("limits.get");
      const openAiResult = limits.openai;
      const antigravityResult = limits.antigravity;

      const outputBlocks: string[] = [];
      if (openAiResult) {
        if (openAiResult.ok) {
          outputBlocks.push(formatOpenAiUsageStatus(openAiResult.snapshot));
        } else {
          outputBlocks.push(`codex usage limits failed: ${clipInline(openAiResult.message, 180)}`);
        }
      }

      if (antigravityResult) {
        if (antigravityResult.ok) {
          outputBlocks.push(formatAntigravityUsageStatus(antigravityResult.snapshot));
        } else {
          outputBlocks.push(formatAntigravityUsageFailure(antigravityResult.message));
        }
      }

      appendSystemMessage(outputBlocks.join("\n\n"));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      appendSystemMessage(`limits failed: ${clipInline(message, 180)}`);
    } finally {
      statusCommandInFlightRef.current = false;
      setPending(false);
      setStatusLabel("ready");
    }
  };

  const applyCommand = async (rawCommand: string): Promise<void> => {
    const trimmed = rawCommand.trim();
    if (!trimmed) {
      return;
    }
    const tokens = trimmed.split(/\s+/).filter(Boolean);
    if (tokens.length === 0) {
      return;
    }
    const command = tokens[0]!.toLowerCase();
    const args = tokens.slice(1);

    if (command === "/auth") {
      openAuthSelector(false, false);
      return;
    }

    if (command === "/onboarding") {
      openOnboardingSelector();
      return;
    }

    if (command === "/forgeteverything") {
      try {
        await callRpc("command.execute", {
          session_id: getRpcSessionId(),
          raw_command: "/forgeteverything",
        });
        setPendingImages([]);
        setInputHistory([]);
        setInputHistoryIndex(null);
        setInputHistoryDraft("");
        await refreshRuntimeState();
        await refreshRuntimeSessionState();
        setSelector({
          kind: "onboarding",
          title: "onboarding 1/2 - auth providers",
          index: 0,
          options: buildOnboardingAuthOptions([]),
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`forgeteverything failed: ${message}`);
      }
      return;
    }

    if (command === "/model") {
      openModelSelector();
      return;
    }

    if (command === "/limits") {
      void runStatusCommand();
      return;
    }

    if (command === "/history") {
      await runHistoryCommand(args);
      return;
    }

    if (command === "/clear") {
      try {
        await callRpc("command.execute", {
          session_id: getRpcSessionId(),
          raw_command: "/clear",
        });
        setPendingImages([]);
        await refreshRuntimeSessionState();
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`clear failed: ${message}`);
      }
      return;
    }

    if (command === "/skills") {
      try {
        const result = await callRpc<{
          directories: string[];
          skills: Array<{ name: string; description_preview: string }>;
        }>("skills.list", {});
        if (result.skills.length === 0) {
          appendSystemMessage(`no skills found in: ${result.directories.join(", ")}`);
          return;
        }
        const lines = result.skills
          .map((skill) => `${skill.name} - ${skill.description_preview}`)
          .join("\n");
        appendSystemMessage(`available skills (${result.skills.length}):\n${lines}`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`skills failed: ${message}`);
      }
      return;
    }

    if (command === "/tools") {
      try {
        const result = await callRpc<{
          tools: Array<{ name: string; description: string }>;
        }>("tools.list", {});
        const lines = result.tools
          .map((tool) => `${tool.name} - ${tool.description}`)
          .join("\n");
        appendSystemMessage(`registered tools:\n${lines}`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`tools failed: ${message}`);
      }
      return;
    }

    if (command === "/help") {
      try {
        const result = await callRpc<{
          handled: boolean;
          output?: {
            commands?: Array<{ name: string; description: string }>;
          };
        }>("command.execute", {
          session_id: getRpcSessionId(),
          raw_command: "/help",
        });
        const commands = result.output?.commands ?? [];
        const commandLines = commands.length > 0
          ? commands.map((option) => `${option.name} - ${option.description}`).join("\n")
          : COMMAND_OPTIONS.map((option) => `${option.name} - ${option.description}`).join("\n");
        appendSystemMessage(`available commands:\n${commandLines}`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`help failed: ${message}`);
      }
      return;
    }

    if (command === "/quit" || command === "/exit") {
      exit();
      return;
    }

    if (command === SUPER_DEBUG_COMMAND) {
      const next = !superDebug;
      try {
        await callRpc("debug.set", {
          super_debug: next,
        });
        setSuperDebug(next);
        appendSystemMessage(`superdebug: ${next ? "on" : "off"}`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`superdebug failed: ${message}`);
      }
      return;
    }

    try {
      const result = await callRpc<{
        handled: boolean;
        output?: unknown;
      }>("command.execute", {
        session_id: getRpcSessionId(),
        raw_command: trimmed,
      });
      if (!result.handled) {
        appendSystemMessage(`unknown command: ${command}`);
        return;
      }
      if (result.output && typeof result.output === "object" && "error" in result.output) {
        const errorValue = (result.output as { error?: unknown }).error;
        if (typeof errorValue === "string") {
          appendSystemMessage(errorValue);
          return;
        }
      }
      if (result.output !== undefined) {
        appendSystemMessage(safeJsonStringify(result.output));
      }
      await refreshRuntimeSessionState();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      appendSystemMessage(`command failed: ${message}`);
    }
  };

  const runSlashCommand = (rawInput: string) => {
    const trimmed = rawInput.trim();
    if (!trimmed.startsWith("/")) {
      return false;
    }

    const parts = trimmed.split(/\s+/).filter(Boolean);
    let commandToRun = (parts[0] ?? "").toLowerCase();
    if (
      commandToRun !== SUPER_DEBUG_COMMAND &&
      commandSuggestions.length > 0 &&
      !COMMAND_OPTIONS.some((option) => option.name === commandToRun)
    ) {
      commandToRun = commandSuggestions[Math.min(commandIndex, commandSuggestions.length - 1)]!.name;
    }

    const commandPayload =
      commandToRun === (parts[0] ?? "").toLowerCase()
        ? trimmed
        : [commandToRun, ...parts.slice(1)].join(" ");
    const now = Date.now();
    const previous = recentSlashCommandRef.current;
    if (previous && previous.payload === commandPayload && now - previous.at < 500) {
      return true;
    }
    recentSlashCommandRef.current = {
      payload: commandPayload,
      at: now,
    };

    setInput("");
    setInputHistoryIndex(null);
    setInputHistoryDraft("");
    void applyCommand(commandPayload);
    return true;
  };

  const navigateInputHistory = (direction: "up" | "down"): boolean => {
    if (inputHistory.length === 0) {
      return false;
    }

    if (direction === "up") {
      if (inputHistoryIndex === null) {
        setInputHistoryDraft(input);
        const nextIndex = inputHistory.length - 1;
        setInputHistoryIndex(nextIndex);
        setInput(inputHistory[nextIndex] ?? "");
        return true;
      }

      if (inputHistoryIndex > 0) {
        const nextIndex = inputHistoryIndex - 1;
        setInputHistoryIndex(nextIndex);
        setInput(inputHistory[nextIndex] ?? "");
      }
      return true;
    }

    if (inputHistoryIndex === null) {
      return false;
    }

    if (inputHistoryIndex < inputHistory.length - 1) {
      const nextIndex = inputHistoryIndex + 1;
      setInputHistoryIndex(nextIndex);
      setInput(inputHistory[nextIndex] ?? "");
      return true;
    }

    setInputHistoryIndex(null);
    setInput(inputHistoryDraft);
    return true;
  };

  const applySkillAutocomplete = (rawInput: string, suggestionName: string) => {
    const replacement = replaceActiveSkillTokenWithSuggestion(rawInput, suggestionName);
    if (!replacement) {
      return;
    }
    setInput(replacement.text);
    setAutocompletedSkillPrefix(replacement.prefix);
    // Remount controlled TextInput so cursor jumps to the end after autocomplete.
    setTextInputResetKey((current) => current + 1);
  };

  useInput((character, key) => {
    if (key.ctrl && character === "c") {
      exit();
      return;
    }
    if (!secretsHydrated) {
      return;
    }

    if (!selector && isImageAttachKeybind(character, key)) {
      if (suppressCtrlVEchoRef.current.timeout) {
        clearTimeout(suppressCtrlVEchoRef.current.timeout);
      }
      suppressCtrlVEchoRef.current = {
        active: true,
        previousInput: input,
        timeout: setTimeout(() => {
          suppressCtrlVEchoRef.current.active = false;
          suppressCtrlVEchoRef.current.timeout = null;
        }, 120),
      };
      const attachment = readClipboardImageAttachment();
      if (attachment.ok) {
        queueImageAttachment(attachment.image);
        return;
      }
      suppressCtrlVEchoRef.current.active = false;
      if (suppressCtrlVEchoRef.current.timeout) {
        clearTimeout(suppressCtrlVEchoRef.current.timeout);
        suppressCtrlVEchoRef.current.timeout = null;
      }
      appendSystemMessage(`paste image failed: ${attachment.error}`);
      return;
    }

    if (!selector && pending && key.return) {
      // Some terminals don't reliably set key.shift for Enter variants.
      // Treat LF (`\n`) Enter as steer as a compatibility fallback.
      const isSteerEnter = key.shift || character === "\n";
      const trimmed = input.trim();

      if (!trimmed) {
        appendSystemMessage(
          isSteerEnter
            ? "enter a steer message first, then press shift+enter."
            : "cannot queue an empty prompt while running.",
        );
        suppressNextSubmitRef.current = true;
        return;
      }

      if (isSteerEnter) {
        const steerText = parseSteerCommand(trimmed) ?? trimmed;
        queueSteeringMessage(steerText);
        suppressNextSubmitRef.current = true;
        return;
      }

      const pendingCommand = trimmed.split(/\s+/)[0]?.toLowerCase();
      if (pendingCommand === "/quit" || pendingCommand === "/exit") {
        suppressNextSubmitRef.current = true;
        exit();
        return;
      }
      if (trimmed.startsWith("/")) {
        appendSystemMessage(
          "slash commands cannot be queued while running. use esc to interrupt first.",
        );
        suppressNextSubmitRef.current = true;
        return;
      }

      queuePendingPrompt(trimmed);
      suppressNextSubmitRef.current = true;
      return;
    }

    if (key.escape && pending) {
      if (interruptActiveInference()) {
        appendSystemMessage("interrupt requested. waiting for current step to stop...");
      }
      return;
    }

    if (selector) {
      if (key.escape) {
        if (selector.kind === "onboarding") {
          if (!onboardingCompleted) {
            return;
          }
          setInput("");
          setSelector(null);
          return;
        }
        if (
          (selector.kind === "openrouter_api_key" || selector.kind === "exa_api_key") &&
          selector.returnToOnboarding
        ) {
          openOnboardingSelector();
          return;
        }
        if (selector.kind === "auth" && selector.returnToOnboarding) {
          openOnboardingSelector();
          return;
        }
        if (
          selector.kind === "auth" &&
          enabledProviders.length === 0 &&
          selector.options.every((option) => option.id !== "antigravity")
        ) {
          return;
        }
        setInput("");
        setSelector(null);
        return;
      }

      if (key.upArrow || key.downArrow) {
        if (
          selector.kind === "openrouter_api_key" ||
          selector.kind === "exa_api_key"
        ) {
          return;
        }
        setSelector((current) => {
          if (!current) {
            return current;
          }
          if (
            current.kind === "openrouter_api_key" ||
            current.kind === "exa_api_key"
          ) {
            return current;
          }
          const direction = key.downArrow ? 1 : -1;
          const optionCount =
            current.kind === "model"
              ? Math.max(1, getModelSelectorOptions(current).length)
              : current.options.length;
          const nextIndex = (current.index + direction + optionCount) % optionCount;
          return {
            ...current,
            index: nextIndex,
          };
        });
        return;
      }

      if (key.return) {
        if (pending) {
          return;
        }

        if (selector.kind === "onboarding") {
          const onboardingChoice = selector.options[selector.index];
          if (!onboardingChoice) {
            return;
          }
          if (onboardingChoice.id === "auth_openai") {
            openAuthSelector(true, true);
            return;
          }
          if (onboardingChoice.id === "auth_openrouter") {
            openAuthSelector(true, true);
            return;
          }
          openExaApiKeySelector(true);
          appendSystemMessage("enter your exa api key, or type 'skip'.");
          return;
        }

        if (selector.kind === "exa_api_key") {
          const exaInput = input.trim();
          if (!exaInput) {
            appendSystemMessage("exa api key cannot be empty. type 'skip' to skip.");
            return;
          }
          void (async () => {
            const result = await applyExaApiKeySelection(exaInput);
            if (result === "invalid") {
              appendSystemMessage("invalid exa api key input.");
              return;
            }
            setInput("");
            if (selector.returnToOnboarding) {
              try {
                await callRpc("onboarding.complete", {});
                await refreshRuntimeState();
              } catch {
                // keep local completion to avoid blocking the user if onboarding status refresh fails
              }
              setOnboardingCompleted(true);
              setSelector(null);
              appendSystemMessage("onboarding complete. you can use /auth and /model any time.");
            } else {
              setSelector(null);
            }
          })();
          return;
        }

        if (selector.kind === "openrouter_api_key") {
          const keyInput = input.trim();
          if (!keyInput) {
            appendSystemMessage("openrouter api key cannot be empty.");
            return;
          }
          void (async () => {
            const ok = await applyAuthSelection(
              "openrouter",
              keyInput,
              selector.returnToOnboarding === true,
            );
            if (ok) {
              appendSystemMessage("openrouter api key saved.");
              setInput("");
              if (selector.returnToOnboarding) {
                openOnboardingSelector();
              } else {
                setSelector(null);
              }
            }
          })();
          return;
        }

        if (selector.kind === "auth") {
          const authChoice = selector.options[selector.index];
          if (!authChoice) {
            return;
          }
          void (async () => {
            const ok = await applyAuthSelection(
              authChoice.id,
              undefined,
              selector.returnToOnboarding === true,
            );
            if (ok) {
              if (selector.returnToOnboarding) {
                openOnboardingSelector();
              } else {
                setSelector(null);
              }
            }
          })();
          return;
        }

        if (selector.kind === "history") {
          const historyChoice = selector.options[selector.index];
          if (!historyChoice) {
            return;
          }
          void runHistoryCommand([historyChoice.session.id]);
          return;
        }

        if (selector.kind === "provider_switch_confirm") {
          const switchChoice = selector.options[selector.index];
          if (!switchChoice) {
            return;
          }
          if (switchChoice.id === "switch_cancel") {
            appendSystemMessage("provider switch canceled.");
            setSelector(null);
            setInput("");
            return;
          }
          void applyModelSelection({
            modelId: selector.modelId,
            modelLabel: selector.modelLabel,
            modelProvider: selector.modelProvider,
            thinkingLevel: selector.thinkingLevel,
            openRouterProvider: selector.openRouterProvider,
            bypassProviderSwitchWarning: true,
          });
          return;
        }

        if (selector.kind === "model") {
          const visibleOptions = getModelSelectorOptions(selector);
          const modelChoice = visibleOptions[selector.index];
          if (!modelChoice) {
            appendSystemMessage("no model matches your search.");
            return;
          }
          const isAntigravityModel =
            (modelChoice.displayProvider ?? "").trim().toLowerCase() === "antigravity";
          const thinkingOptions = getThinkingOptionsForModel(
            modelChoice.id,
            modelChoice.provider,
            modelOptionsByProvider,
          );
          const defaultThinking = normalizeThinkingForModel(
            modelChoice.id,
            modelChoice.provider,
            selectedThinking,
            modelOptionsByProvider,
          );
          const defaultIndex = Math.max(
            0,
            thinkingOptions.findIndex((option) => option.id === defaultThinking),
          );

          if (isAntigravityModel) {
            const antigravityThinking =
              modelChoice.defaultThinkingLevel ??
              (thinkingOptions[defaultIndex]?.id as ThinkingLevel | undefined) ??
              "MEDIUM";
            void applyModelSelection({
              modelId: modelChoice.id,
              modelLabel: modelChoice.label,
              modelProvider: modelChoice.provider,
              thinkingLevel: antigravityThinking,
            });
            return;
          }

          setSelector({
            kind: "thinking",
            title: `select thinking level for ${modelChoice.label}`,
            index: defaultIndex,
            modelId: modelChoice.id,
            modelLabel: modelChoice.label,
            modelProvider: modelChoice.provider,
            options: thinkingOptions,
          });
          setInput("");
          return;
        }

        if (selector.kind === "thinking") {
          const thinkingChoice = selector.options[selector.index];
          if (!thinkingChoice) {
            return;
          }
          if (selector.modelProvider === "openrouter") {
            void (async () => {
              let discoveredProviders: string[] = [];
              setPending(true);
              setStatusLabel("loading providers...");
              try {
                const result = await callRpc<{ model_id: string; providers: string[] }>("model.openrouter.providers", {
                  model_id: selector.modelId,
                });
                discoveredProviders = result.providers;
                await refreshRuntimeState();
              } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                appendSystemMessage(`openrouter provider fetch failed: ${message}`);
              } finally {
                setPending(false);
                setStatusLabel("ready");
              }

              const routeOptions = getOpenRouterProviderOptions(
                selector.modelId,
                modelOptionsByProvider,
                discoveredProviders,
              );
              const selectedRoute = normalizeOpenRouterProviderSelection(selectedOpenRouterProvider);
              const defaultIndex = Math.max(
                0,
                routeOptions.findIndex((option) => option.id === selectedRoute),
              );
              setSelector({
                kind: "openrouter_provider",
                title: `select provider for ${selector.modelLabel}`,
                index: defaultIndex,
                modelId: selector.modelId,
                modelLabel: selector.modelLabel,
                modelProvider: selector.modelProvider,
                thinkingLevel: thinkingChoice.id as ThinkingLevel,
                options: routeOptions,
              });
            })();
            return;
          }

          void applyModelSelection({
            modelId: selector.modelId,
            modelLabel: selector.modelLabel,
            modelProvider: selector.modelProvider,
            thinkingLevel: thinkingChoice.id as ThinkingLevel,
          });
          return;
        }

        if (selector.kind === "openrouter_provider") {
          const routeChoice = selector.options[selector.index];
          if (!routeChoice) {
            return;
          }
          void applyModelSelection({
            modelId: selector.modelId,
            modelLabel: selector.modelLabel,
            modelProvider: selector.modelProvider,
            thinkingLevel: selector.thinkingLevel,
            openRouterProvider: routeChoice.id,
          });
          return;
        }
      }

      return;
    }

    if (input.startsWith("/") && commandSuggestions.length > 0) {
      if (key.return) {
        // Prevent TextInput onSubmit from dispatching the same slash command again.
        suppressNextSubmitRef.current = true;
        runSlashCommand(input);
        return;
      }

      if (key.upArrow || key.downArrow) {
        const direction = key.downArrow ? 1 : -1;
        setCommandIndex((current) => {
          const optionCount = commandSuggestions.length;
          return (current + direction + optionCount) % optionCount;
        });
        return;
      }

      if (key.tab) {
        const suggestion = commandSuggestions[commandIndex];
        if (suggestion) {
          setInput(suggestion.name);
        }
      }
    }

    if (showSkillSuggestions) {
      if (key.upArrow || key.downArrow) {
        const direction = key.downArrow ? 1 : -1;
        setSkillIndex((current) => {
          const optionCount = skillSuggestions.length;
          return (current + direction + optionCount) % optionCount;
        });
        return;
      }

      if (key.return) {
        const suggestion = skillSuggestions[skillIndex];
        if (suggestion && shouldAutocompleteSkillInputOnEnter(input, suggestion.nameLower)) {
          applySkillAutocomplete(input, suggestion.name);
          return;
        }
      }

      if (key.tab) {
        const suggestion = skillSuggestions[skillIndex];
        if (suggestion) {
          applySkillAutocomplete(input, suggestion.name);
        }
        return;
      }
    }

    if (!input.startsWith("/") && (key.upArrow || key.downArrow)) {
      const usedHistory = navigateInputHistory(key.upArrow ? "up" : "down");
      if (usedHistory) {
        return;
      }
    }
  });

  const visibleMessageLimit = useMemo(() => {
    const terminalRows = Number.isFinite(process.stdout.rows) ? process.stdout.rows : 24;
    let reservedRows = BASE_LAYOUT_ROWS + (pending ? 1 : 0);

    if (showCommandSuggestionPanel) {
      reservedRows += commandSuggestions.length + 2;
    }

    if (showSkillSuggestionPanel) {
      reservedRows += skillSuggestions.length + 2;
    }

    if (selector) {
      if (selector.kind === "openrouter_api_key" || selector.kind === "exa_api_key") {
        reservedRows += 3;
      } else {
        const allOptions = getSelectorAllOptions(selector);
        const windowed = getSelectorOptionWindow(allOptions, selector.index);
        const hasWindowHint = allOptions.length > windowed.options.length;
        reservedRows += windowed.options.length + (hasWindowHint ? 1 : 0) + 3;
      }
    }

    const availableRows = Math.max(0, terminalRows - reservedRows);
    const maxMessagesByRows = Math.floor(availableRows / MIN_VISIBLE_MESSAGE_ROWS);
    return Math.max(1, Math.min(MAX_VISIBLE_MESSAGES, maxMessagesByRows));
  }, [
    pending,
    showCommandSuggestionPanel,
    showSkillSuggestionPanel,
    commandSuggestions.length,
    skillSuggestions.length,
    selector,
    input,
  ]);

  const visibleMessages = useMemo(
    () => messages.slice(-visibleMessageLimit),
    [messages, visibleMessageLimit],
  );

  const sendPrompt = async (prompt: string) => {
    const cleanPrompt = prompt.trim();
    if ((!cleanPrompt && pendingImages.length === 0) || pending) {
      return;
    }
    if (!onboardingCompleted) {
      appendSystemMessage("finish onboarding first.");
      openOnboardingSelector();
      return;
    }
    if (!selectedModelProvider) {
      appendSystemMessage("select a model from an enabled provider first.");
      openAuthSelector(false, true);
      return;
    }

    const promptImages = pendingImages.map((image) => ({
      path: image.path,
      mime_type: image.mimeType,
      data_url: image.dataUrl,
    }));

    setInput("");
    if (cleanPrompt) {
      setInputHistory((current) => {
        const next = [...current, cleanPrompt];
        return next.slice(-MAX_INPUT_HISTORY);
      });
    }
    setInputHistoryIndex(null);
    setInputHistoryDraft("");
    setPendingImages([]);

    try {
      await callRpc("session.send", {
        session_id: getRpcSessionId(),
        text: cleanPrompt,
        images: promptImages,
        enqueue: false,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      appendSystemMessage(`send failed: ${message}`);
    }
  };

  if (!secretsHydrated) {
    return (
      <Box flexDirection="column" paddingX={1}>
        <Text color="cyanBright">loaf | beta</Text>
        <Text color="yellow">
          {GLYPH_SYSTEM}
          starting...
        </Text>
        <Text color="gray">{startupStatusLabel}</Text>
        <Newline />
        <Text color="gray">{exitShortcutLabel} exit</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" paddingX={1}>
      <Text color="cyanBright">loaf | beta</Text>
      <Text color="gray">
        auth: {authSummary} |{" "}
        model:{" "}
        {selectedModelProvider
          ? formatModelSummary(
            selectedModelProvider,
            selectedModel,
            selectedThinking,
            modelOptionsByProvider,
            selectedOpenRouterProvider,
          )
          : "not selected"}
      </Text>
      <Newline />
      <Box flexDirection="column">
        {visibleMessages.map((message) => (
          <MemoizedMessageRow key={message.id} message={message} />
        ))}
      </Box>
      {selector && (
        <Box flexDirection="column" marginTop={1}>
          <Text color="cyanBright">{selector.title}</Text>
          {selector.kind === "openrouter_api_key" || selector.kind === "exa_api_key" ? (
            <Text color="gray">
              {selector.kind === "exa_api_key"
                ? "paste key, press enter to save. type 'skip' to skip. esc cancels."
                : "paste key, press enter to save. esc cancels."}
            </Text>
          ) : (
            <>
              {(() => {
                const allOptions = getSelectorAllOptions(selector);
                const windowed = getSelectorOptionWindow(allOptions, selector.index);
                return (
                  <>
                    {windowed.options.map((option, windowIndex) => {
                      const absoluteIndex = windowed.startIndex + windowIndex;
                      const selected = absoluteIndex === windowed.activeIndex;
                      return (
                        <Text key={`${selector.kind}-${absoluteIndex}-${option.id}`} color={selected ? "magentaBright" : "gray"}>
                          {selected ? ">" : " "} {option.label} - {selector.kind === "model" ? getModelProviderDisplayLabel(option as ModelOption) : option.description}
                        </Text>
                      );
                    })}
                    {allOptions.length > windowed.options.length && (
                      <Text color="gray">
                        showing {windowed.startIndex + 1}-{windowed.startIndex + windowed.options.length} of {allOptions.length}
                      </Text>
                    )}
                  </>
                );
              })()}
              <Text color="gray">
                {selector.kind === "onboarding"
                  ? "use up/down + enter to configure."
                  : selector.kind === "model"
                    ? "type to search | up/down select | enter confirm | esc cancels"
                    : "use up/down + enter. esc cancels."}
              </Text>
            </>
          )}
        </Box>
      )}
      <Newline />
      {pending && <Text color="yellow">{GLYPH_SYSTEM}{statusLabel}</Text>}
      {showCommandSuggestionPanel && (
        <Box flexDirection="column" marginTop={1}>
          {commandSuggestions.map((suggestion, index) => (
            <Text key={suggestion.name} color={index === commandIndex ? "magentaBright" : "gray"}>
              {index === commandIndex ? ">" : " "} {suggestion.name} - {suggestion.description}
            </Text>
          ))}
          <Text color="gray">tab autocomplete | up/down navigate suggestions</Text>
        </Box>
      )}
      {showSkillSuggestionPanel && (
        <Box flexDirection="column" marginTop={1}>
          {skillSuggestions.map((skill, index) => (
            <Text key={skill.name} color={index === skillIndex ? "magentaBright" : "gray"}>
              {index === skillIndex ? ">" : " "} ${skill.name} - {skill.descriptionPreview}
            </Text>
          ))}
          <Text color="gray">enter/tab autocomplete | up/down navigate skills</Text>
        </Box>
      )}
      <Box marginTop={1}>
        <Text color="magentaBright">{selector ? "select> " : GLYPH_USER}</Text>
        <TextInput
          key={`prompt-input-${textInputResetKey}`}
          value={input}
          onChange={(value) => {
            let nextValue = value;
            if (suppressCtrlVEchoRef.current.active) {
              const previousInput = suppressCtrlVEchoRef.current.previousInput;
              const consumed = consumeCtrlVEchoArtifact(nextValue, previousInput);
              if (consumed !== nextValue) {
                suppressCtrlVEchoRef.current.active = false;
                if (suppressCtrlVEchoRef.current.timeout) {
                  clearTimeout(suppressCtrlVEchoRef.current.timeout);
                  suppressCtrlVEchoRef.current.timeout = null;
                }
                // Swallow the Ctrl+V echo event and keep the current composer state.
                return;
              }
              suppressCtrlVEchoRef.current.active = false;
              if (suppressCtrlVEchoRef.current.timeout) {
                clearTimeout(suppressCtrlVEchoRef.current.timeout);
                suppressCtrlVEchoRef.current.timeout = null;
              }
            }
            if (
              selector?.kind === "openrouter_api_key" ||
              selector?.kind === "exa_api_key" ||
              selector?.kind === "model" ||
              !selector
            ) {
              const reconciled = reconcilePendingImagesWithComposerText(nextValue, pendingImages);
              nextValue = reconciled.text;
              if (!areImageAttachmentListsEqual(reconciled.images, pendingImages)) {
                setPendingImages(reconciled.images);
              }
              if (inputHistoryIndex !== null) {
                setInputHistoryIndex(null);
                setInputHistoryDraft("");
              }
              if (selector?.kind === "model") {
                setSelector((current) => {
                  if (!current || current.kind !== "model") {
                    return current;
                  }
                  return {
                    ...current,
                    index: 0,
                  };
                });
              }
              setInput(nextValue);
            }
          }}
          onSubmit={(submitted) => {
            if (suppressNextSubmitRef.current) {
              suppressNextSubmitRef.current = false;
              return;
            }

            if (
              selector &&
              selector.kind !== "openrouter_api_key" &&
              selector.kind !== "exa_api_key"
            ) {
              return;
            }

            if (showSkillSuggestions) {
              const suggestion = skillSuggestions[skillIndex];
              if (suggestion && shouldAutocompleteSkillInputOnEnter(submitted, suggestion.nameLower)) {
                applySkillAutocomplete(submitted, suggestion.name);
                return;
              }
            }

            const trimmed = submitted.trim();
            if (!trimmed && pendingImages.length === 0) {
              return;
            }

            if (
              selector?.kind === "openrouter_api_key" ||
              selector?.kind === "exa_api_key"
            ) {
              return;
            }

            if (pending) {
              if (!trimmed) {
                appendSystemMessage("cannot queue an empty prompt while running.");
                return;
              }

              const pendingCommand = trimmed.split(/\s+/)[0]?.toLowerCase();
              if (pendingCommand === "/quit" || pendingCommand === "/exit") {
                exit();
                return;
              }
              if (trimmed.startsWith("/")) {
                appendSystemMessage(
                  "slash commands cannot be queued while running. use esc to interrupt first.",
                );
                return;
              }
              queuePendingPrompt(trimmed);
              return;
            }

            if (trimmed.startsWith("/")) {
              runSlashCommand(trimmed);
              return;
            }

            sendPrompt(submitted);
          }}
          placeholder={
            selector?.kind === "openrouter_api_key"
              ? "enter openrouter api key..."
              : selector?.kind === "exa_api_key"
                ? "enter exa api key or type 'skip'..."
                : selector?.kind === "model"
                  ? "search model..."
                  : selector
                    ? "use selector above..."
                    : "type a prompt and press enter..."
          }
          showCursor
        />
      </Box>
      <Text color="gray">
        {pending ? "esc interrupt | enter queue | shift+enter steer | " : ""}
        {getAttachImageShortcutLabel()} attach image |{" "}
        {exitShortcutLabel} exit | /help for commands
      </Text>
    </Box>
  );
}

function parseSteerCommand(rawInput: string): string | null {
  const trimmed = rawInput.trim();
  const match = trimmed.match(/^\/steer(?:\s+([\s\S]+))?$/i);
  if (!match) {
    return null;
  }
  const payload = (match[1] ?? "").trim();
  return payload || null;
}

function shouldAutocompleteSkillInputOnEnter(rawInput: string, selectedSkillNameLower: string): boolean {
  const mention = findActiveSkillMentionToken(rawInput);
  if (!mention) {
    return false;
  }

  const enteredSkillName = mention.query;
  const trailingContent = rawInput.slice(mention.end).trim();
  const hasTrailingPromptText = trailingContent.length > 0;
  const isExactSkillMatch = enteredSkillName === selectedSkillNameLower;

  if (isExactSkillMatch && hasTrailingPromptText) {
    return false;
  }

  return true;
}

type ActiveSkillMentionToken = {
  start: number;
  end: number;
  query: string;
};

function findActiveSkillMentionToken(rawInput: string): ActiveSkillMentionToken | null {
  const mentionPattern = /(^|\s)\$([a-zA-Z0-9._:-]*)/g;
  let match: RegExpExecArray | null;
  while ((match = mentionPattern.exec(rawInput)) !== null) {
    const leftBoundary = match[1] ?? "";
    const mentionValue = match[2] ?? "";
    const start = match.index + leftBoundary.length;
    const end = start + 1 + mentionValue.length;
    const nextChar = rawInput[end];
    if (nextChar && !/\s/.test(nextChar)) {
      continue;
    }

    return {
      start,
      end,
      query: mentionValue.toLowerCase(),
    };
  }

  return null;
}

function replaceActiveSkillTokenWithSuggestion(
  rawInput: string,
  suggestionName: string,
): { text: string; prefix: string } | null {
  const mention = findActiveSkillMentionToken(rawInput);
  if (!mention) {
    return null;
  }

  const before = rawInput.slice(0, mention.start);
  const after = rawInput.slice(mention.end);
  const mentionText = `$${suggestionName}`;
  const spacer = after.length === 0 ? " " : "";
  const nextText = `${before}${mentionText}${spacer}${after}`;

  return {
    text: nextText,
    prefix: `${before}${mentionText} `,
  };
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function formatToolRows(data: unknown): string[] {
  const executed = (data as { executed?: unknown })?.executed;
  if (!Array.isArray(executed)) {
    return [];
  }

  return executed
    .map((item) => parseToolExecutionRecord(item))
    .filter((record): record is ToolExecutionRecord => Boolean(record))
    .map((record) => formatToolCompletedExecutionRow(record));
}

function formatToolStartRows(data: unknown): string[] {
  const functionCalls = (data as { functionCalls?: unknown })?.functionCalls;
  if (!Array.isArray(functionCalls)) {
    return [];
  }

  const rows: string[] = [];
  for (const rawCall of functionCalls) {
    const row = formatToolStartRow(rawCall);
    if (row) {
      rows.push(row);
    }
  }
  return rows;
}

function formatToolStartRow(rawCall: unknown): string | null {
  const parsed = parseToolCallPreview(rawCall);
  if (!parsed) {
    return null;
  }
  return formatToolLifecycleHeader("calling", parsed.name, parsed.input);
}

function formatToolCompletedRow(data: unknown): string | null {
  const executed = parseToolExecutionRecord((data as { executed?: unknown })?.executed);
  if (!executed) {
    return null;
  }
  return formatToolCompletedExecutionRow(executed);
}

function shouldCollapseSuccessDetail(name: string): boolean {
  return (
    name !== "create_persistent_tool" &&
    name !== "bash" &&
    name !== "install_js_packages" &&
    name !== "run_js" &&
    name !== "run_js_module" &&
    name !== "start_background_js" &&
    name !== "read_background_js" &&
    name !== "write_background_js" &&
    name !== "stop_background_js" &&
    name !== "list_background_js" &&
    name !== "search_web"
  );
}

type ToolExecutionRecord = {
  name: string;
  ok: boolean;
  input: Record<string, unknown>;
  result: unknown;
  error?: string;
};

const TOOL_RENDER_DEFAULT_WIDTH = 100;
const TOOL_RENDER_MIN_WIDTH = 24;
const TOOL_OUTPUT_MAX_LINES = 5;

function parseToolExecutionRecord(raw: unknown): ToolExecutionRecord | null {
  if (!isRecord(raw)) {
    return null;
  }
  const record = isRecord(raw.executed) ? raw.executed : raw;
  const name = readTrimmedString(record.name) || "tool";
  const input = isRecord(record.input) ? record.input : {};
  const error = readTrimmedString(record.error) || undefined;
  return {
    name,
    ok: record.ok === true,
    input,
    result: record.result,
    error,
  };
}

function formatToolCompletedExecutionRow(record: ToolExecutionRecord): string {
  const header = formatToolLifecycleHeader("called", record.name, record.input);
  const bodyLines = formatToolExecutionBodyLines(record);
  if (bodyLines.length === 0) {
    return header;
  }
  return [header, ...prefixToolBodyLines(bodyLines)].join("\n");
}

function formatToolLifecycleHeader(
  phase: "calling" | "called",
  name: string,
  input: Record<string, unknown>,
): string {
  const label = phase === "calling" ? "Calling" : "Called";
  const invocation = formatToolInvocation(name, input);
  const wrapWidth = getToolWrapWidth();
  const inline = `${label} ${invocation}`;
  if (inline.length <= wrapWidth) {
    return inline;
  }

  const invocationLines = wrapToolLines(invocation, Math.max(TOOL_RENDER_MIN_WIDTH, wrapWidth - 4));
  return [
    `${label}`,
    ...invocationLines.map((line, index) => `${index === 0 ? "  └ " : "    "}${line}`),
  ].join("\n");
}

function formatToolInvocation(name: string, input: Record<string, unknown>): string {
  const args =
    Object.keys(input).length === 0
      ? ""
      : safeJsonStringify(input)
          .replace(/\s+/g, " ")
          .trim();
  return `${name}(${args})`;
}

function formatToolExecutionBodyLines(record: ToolExecutionRecord): string[] {
  const outputLines = formatToolOutputLines(record.name, record.input, record.result);
  if (!record.ok) {
    const lines = [`Error: ${record.error ?? "tool execution failed"}`];
    if (outputLines.length > 0) {
      lines.push(...outputLines);
    }
    return truncateToolBodyLines(lines);
  }

  if (outputLines.length === 0) {
    return ["(no output)"];
  }

  return truncateToolBodyLines(outputLines);
}

function formatToolOutputLines(name: string, input: Record<string, unknown>, result: unknown): string[] {
  const record = isRecord(result) ? result : {};
  const combinedProcessOutput = getCombinedProcessOutput(record).trim();
  if (combinedProcessOutput) {
    return toMultilineLines(combinedProcessOutput);
  }

  if (name === "search_web") {
    const results = Array.isArray(record.results) ? record.results : [];
    if (results.length > 0) {
      const rows = results.slice(0, 6).map((entry, index) => {
        const row = isRecord(entry) ? entry : {};
        const title = readTrimmedString(row.title) || readTrimmedString(row.url) || `result ${index + 1}`;
        const url = readTrimmedString(row.url);
        return url ? `${title} - ${url}` : title;
      });
      if (results.length > rows.length) {
        rows.push(`... +${results.length - rows.length} results`);
      }
      return rows;
    }
  }

  if (name === "list_background_js") {
    const sessions = Array.isArray(record.sessions) ? record.sessions : [];
    if (sessions.length > 0) {
      return sessions.slice(0, 8).map((entry, index) => {
        const session = isRecord(entry) ? entry : {};
        const sessionName = readTrimmedString(session.session_name) || `session ${index + 1}`;
        const status = readTrimmedString(session.status) || "unknown";
        return `${sessionName} (${status})`;
      });
    }
  }

  const message = readTrimmedString(record.message);
  if (message) {
    return toMultilineLines(message);
  }

  const detail = formatToolDetail(name, input, result).trim();
  if (detail && detail.toLowerCase() !== "ok") {
    return toMultilineLines(detail);
  }

  if (typeof result === "string") {
    const raw = result.trim();
    if (raw) {
      return toMultilineLines(raw);
    }
  }

  if (isRecord(result) && Object.keys(result).length > 0) {
    return toMultilineLines(formatInlineJson(result));
  }

  return [];
}

function truncateToolBodyLines(lines: string[]): string[] {
  const wrapWidth = Math.max(TOOL_RENDER_MIN_WIDTH, getToolWrapWidth() - 4);
  const wrapped = lines.flatMap((line) => wrapToolLines(line, wrapWidth));
  if (wrapped.length <= TOOL_OUTPUT_MAX_LINES * 2) {
    return wrapped;
  }

  const head = wrapped.slice(0, TOOL_OUTPUT_MAX_LINES);
  const tail = wrapped.slice(-TOOL_OUTPUT_MAX_LINES);
  const omitted = wrapped.length - (head.length + tail.length);
  return [...head, `… +${omitted} lines`, ...tail];
}

function prefixToolBodyLines(lines: string[]): string[] {
  return lines.map((line, index) => `${index === 0 ? "  └ " : "    "}${line}`);
}

function getToolWrapWidth(): number {
  const columns =
    typeof process.stdout.columns === "number" && Number.isFinite(process.stdout.columns)
      ? process.stdout.columns
      : TOOL_RENDER_DEFAULT_WIDTH;
  return Math.max(TOOL_RENDER_MIN_WIDTH, columns - 8);
}

function toMultilineLines(text: string): string[] {
  return text
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trimEnd())
    .filter((line) => line.trim().length > 0);
}

function wrapToolLines(text: string, width: number): string[] {
  const normalized = text.replace(/\r\n/g, "\n");
  const lines = normalized.split("\n");
  const wrapped: string[] = [];

  for (const rawLine of lines) {
    const queue = rawLine.trimEnd();
    if (!queue) {
      wrapped.push("");
      continue;
    }

    let remaining = queue;
    while (remaining.length > width) {
      let breakIndex = remaining.lastIndexOf(" ", width);
      if (breakIndex <= 0) {
        breakIndex = width;
      }
      const segment = remaining.slice(0, breakIndex).trimEnd();
      if (segment) {
        wrapped.push(segment);
      }
      remaining = remaining.slice(breakIndex).trimStart();
    }

    if (remaining) {
      wrapped.push(remaining);
    }
  }

  return wrapped;
}

function formatToolSummary(name: string, input: unknown, result: unknown): string {
  const payload = getToolPayload(input, result);

  if (name === "install_js_packages") {
    const packages = getJavaScriptPackages(payload, result);
    if (packages.length > 0) {
      return `installed ${packages.length} package(s)`;
    }
    return "installed javascript package(s)";
  }

  if (name === "run_js") {
    return "executed javascript script";
  }

  if (name === "bash") {
    const command = readTrimmedString(payload.command);
    if (command) {
      return `ran shell command "${clipInline(command, 72)}"`;
    }
    return "ran shell command";
  }

  if (name === "run_js_module") {
    const record = isRecord(result) ? result : {};
    const moduleName = readTrimmedString(record.module) || readTrimmedString(payload.module);
    if (moduleName) {
      return `ran javascript module ${moduleName}`;
    }
    return "ran javascript module";
  }

  if (name === "search_web") {
    const query = readTrimmedString(payload.query);
    if (query) {
      return `searched web for "${clipInline(query, 72)}"`;
    }
    return "searched web";
  }

  if (name === "start_background_js") {
    return "started background javascript session";
  }

  if (name === "read_background_js") {
    return "read background javascript output";
  }

  if (name === "write_background_js") {
    return "sent input to background javascript session";
  }

  if (name === "stop_background_js") {
    return "stopped background javascript session";
  }

  if (name === "list_background_js") {
    return "listed background javascript sessions";
  }

  if (name === "create_persistent_tool") {
    return "created persistent tool";
  }

  return `${name} executed`;
}

function formatToolDetail(name: string, input: unknown, result: unknown): string {
  const record = isRecord(result) ? result : {};
  const status = typeof record.status === "string" ? record.status.toLowerCase() : "";
  const note = typeof record.note === "string" ? simplifyToolNote(record.note) : "";
  const payload = getToolPayload(input, result);

  const formatWithStatus = (value: string): string => {
    const trimmed = value.trim();
    if (status && status !== "ok") {
      return trimmed ? `${status} - ${trimmed}` : status;
    }
    if (trimmed) {
      return trimmed;
    }
    return note || status || "ok";
  };

  if (name === "install_js_packages") {
    return formatWithStatus(formatInstallJsPackagesDetail(payload, record));
  }

  if (name === "run_js") {
    return formatWithStatus(formatRunJsDetail(payload, record));
  }

  if (name === "bash") {
    return formatWithStatus(formatBashDetail(payload, record));
  }

  if (name === "run_js_module") {
    return formatWithStatus(formatRunJsModuleDetail(payload, record));
  }

  if (name === "search_web") {
    return formatWithStatus(formatSearchWebDetail(payload, record));
  }

  if (name === "start_background_js") {
    const sessionId = readTrimmedString(record.session_id);
    const sessionName = readTrimmedString(record.session_name);
    if (sessionId && sessionName) {
      return formatWithStatus(`${sessionName} (${sessionId})`);
    }
    if (sessionId) {
      return formatWithStatus(sessionId);
    }
  }

  if (name === "read_background_js") {
    const stdout = readTrimmedString(record.stdout);
    const stderr = readTrimmedString(record.stderr);
    const combined = stdout || stderr;
    if (combined) {
      const firstLine = combined
        .replace(/\r\n/g, "\n")
        .split("\n")
        .map((line) => line.trim())
        .find(Boolean);
      if (firstLine) {
        return formatWithStatus(clipInline(firstLine, 180));
      }
    }
    const running = record.running === true ? "running" : "idle";
    return formatWithStatus(running);
  }

  if (name === "write_background_js") {
    const bytes = typeof record.bytes_written === "number" ? record.bytes_written : null;
    if (typeof bytes === "number") {
      return formatWithStatus(`${bytes} byte(s) written`);
    }
  }

  if (name === "stop_background_js") {
    const signal = readTrimmedString(record.signal);
    if (signal) {
      return formatWithStatus(`stop requested (${signal})`);
    }
  }

  if (name === "list_background_js") {
    const count = typeof record.count === "number" ? record.count : null;
    if (typeof count === "number") {
      return formatWithStatus(`${count} session(s)`);
    }
  }

  if (name === "create_persistent_tool") {
    const message = readTrimmedString(record.message);
    if (message) {
      return formatWithStatus(clipInline(message, 180));
    }
  }

  if (status && note) {
    if (note.startsWith(status)) {
      return note;
    }
    return `${status} - ${note}`;
  }
  if (note) {
    return note;
  }
  if (status) {
    return status;
  }
  if (typeof result === "string") {
    return result;
  }
  return formatInlineJson(result);
}

function clipInline(value: string, maxLength: number): string {
  const compact = value.replace(/\s+/g, " ").trim();
  if (compact.length <= maxLength) {
    return compact;
  }
  return `${compact.slice(0, Math.max(0, maxLength - 3))}...`;
}

function collapseRedundantSuccessDetail(summary: string, detail: string): string {
  const trimmed = detail.trim();
  if (!trimmed) {
    return "ok";
  }

  if (trimmed.toLowerCase() === "ok") {
    return "ok";
  }

  const parsed = parseStatusDetail(trimmed);
  const status = parsed.status.toLowerCase();
  const body = parsed.body;
  if (status && status !== "ok") {
    return trimmed;
  }

  const normalizedSummary = normalizeLogText(summary);
  const normalizedBody = normalizeLogText(body);
  if (!normalizedBody) {
    return "ok";
  }

  if (isGenericSuccessBody(normalizedBody)) {
    return "ok";
  }

  if (
    normalizedSummary &&
    (normalizedBody === normalizedSummary ||
      normalizedBody.includes(normalizedSummary) ||
      normalizedSummary.includes(normalizedBody))
  ) {
    return "ok";
  }

  return `ok - ${body}`;
}

function parseStatusDetail(value: string): { status: string; body: string } {
  const match = value.match(/^([a-zA-Z_]+)\s*-\s*(.+)$/);
  if (match) {
    return {
      status: match[1] ?? "",
      body: (match[2] ?? "").trim(),
    };
  }
  return {
    status: "",
    body: value.trim(),
  };
}

function normalizeLogText(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

function isGenericSuccessBody(value: string): boolean {
  return (
    value === "tool executed" ||
    value === "ok"
  );
}

function simplifyToolNote(note: string): string {
  const normalized = note.trim().toLowerCase();
  if (!normalized) {
    return "";
  }
  if (normalized.includes("tool stub executed")) {
    return "stub executed";
  }
  return normalized;
}

function getToolPayload(input: unknown, result: unknown): Record<string, unknown> {
  if (isRecord(input)) {
    return input;
  }
  const received = isRecord(result) ? result.received : undefined;
  if (isRecord(received)) {
    return received;
  }
  return {};
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function getJavaScriptPackages(payload: Record<string, unknown>, result: unknown): string[] {
  const record = isRecord(result) ? result : {};
  return toStringArray(record.packages ?? payload.packages);
}

function formatInstallJsPackagesDetail(payload: Record<string, unknown>, record: Record<string, unknown>): string {
  const packages = getJavaScriptPackages(payload, record);
  const packageSummary =
    packages.length > 0
      ? packages.length <= 3
        ? packages.join(", ")
        : `${packages.slice(0, 3).join(", ")}, +${packages.length - 3} more`
      : "packages";

  const outcome = summarizeInstallJsOutput(record);
  if (outcome) {
    const firstLine = outcome
      .replace(/\r\n/g, "\n")
      .split("\n")
      .map((line) => line.trim())
      .find(Boolean);
    return firstLine ? `${packageSummary} | ${clipInline(firstLine, 140)}` : packageSummary;
  }

  return packageSummary;
}

function formatRunJsDetail(payload: Record<string, unknown>, record: Record<string, unknown>): string {
  const output = getCombinedProcessOutput(record);
  if (output) {
    const firstLine = output
      .replace(/\r\n/g, "\n")
      .split("\n")
      .map((line) => line.trim())
      .find(Boolean);
    if (firstLine) {
      return clipInline(firstLine, 180);
    }
  }
  return "no stdout/stderr";
}

function formatRunJsModuleDetail(payload: Record<string, unknown>, record: Record<string, unknown>): string {
  const moduleName = readTrimmedString(record.module) || readTrimmedString(payload.module);
  const modulePrefix = moduleName ? `${moduleName} | ` : "";

  const output = getCombinedProcessOutput(record);
  if (output) {
    const firstLine = output
      .replace(/\r\n/g, "\n")
      .split("\n")
      .map((line) => line.trim())
      .find(Boolean);
    if (firstLine) {
      return `${modulePrefix}${clipInline(firstLine, 180)}`;
    }
  }
  return `${modulePrefix}no stdout/stderr`.trim();
}

function formatBashDetail(payload: Record<string, unknown>, record: Record<string, unknown>): string {
  const command = readTrimmedString(payload.command);
  const output = getCombinedProcessOutput(record);
  const cwdAfter = readTrimmedString(record.cwd_after);

  const commandPrefix = command ? `${clipInline(command, 120)} | ` : "";
  const cwdSuffix = cwdAfter ? ` | cwd: ${clipInline(cwdAfter, 80)}` : "";

  if (output) {
    const firstLine = output
      .replace(/\r\n/g, "\n")
      .split("\n")
      .map((line) => line.trim())
      .find(Boolean);
    if (firstLine) {
      return `${commandPrefix}${clipInline(firstLine, 160)}${cwdSuffix}`.trim();
    }
  }
  return `${commandPrefix}no stdout/stderr${cwdSuffix}`.trim();
}

function formatSearchWebDetail(payload: Record<string, unknown>, record: Record<string, unknown>): string {
  const countRaw = typeof record.total_results === "number" ? record.total_results : undefined;
  return typeof countRaw === "number" ? `${countRaw} result(s)` : "results";
}

function getCombinedProcessOutput(record: Record<string, unknown>): string {
  const stdout = readTrimmedString(record.stdout);
  const stderr = readTrimmedString(record.stderr);
  if (stdout && stderr) {
    return `${stdout}\n\n[stderr]\n${stderr}`;
  }
  return stdout || stderr;
}

function summarizeInstallJsOutput(record: Record<string, unknown>): string {
  const output = getCombinedProcessOutput(record);
  if (!output) {
    return "";
  }
  const lines = output
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const important = lines.filter((line) =>
    /added|installed|up to date|already up-to-date|done in|error|err!/i.test(line),
  );
  if (important.length > 0) {
    return important.slice(0, 4).join("\n");
  }
  return lines.slice(0, 3).join("\n");
}


function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter(Boolean);
}

function readTrimmedString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function formatInlineJson(value: unknown): string {
  const compact = safeJsonStringify(value).replace(/\s+/g, " ").trim();
  if (compact.length > 240) {
    return `${compact.slice(0, 237)}...`;
  }
  return compact;
}

function MessageRow({ message }: { message: UiMessage }) {
  if (message.kind === "user") {
    return (
      <Box flexDirection="column" marginBottom={1}>
        <Text color="blueBright">{GLYPH_USER}{message.text || "[image]"}</Text>
        {message.images?.map((image, index) => (
          <Text key={`${message.id}-img-${index}`} color="gray">
            {"  "}image: {path.basename(image.path)} ({image.mimeType}, {formatByteSize(image.byteSize)})
          </Text>
        ))}
      </Box>
    );
  }
  if (message.kind === "assistant") {
    return (
      <Box flexDirection="column" marginBottom={1}>
        <MemoizedMarkdownText text={message.text} prefix={GLYPH_ASSISTANT} />
      </Box>
    );
  }
  return (
    <Box flexDirection="column" marginBottom={1}>
      <MemoizedMarkdownText text={message.text} prefix={GLYPH_SYSTEM} />
    </Box>
  );
}

const MemoizedMessageRow = React.memo(MessageRow);

function chatHistoryToUiMessages(history: ChatMessage[], nextMessageId: () => number): UiMessage[] {
  return history.map((message) => ({
    id: nextMessageId(),
    kind: message.role,
    text: message.text,
    images: message.images,
  }));
}

function formatSessionTimestamp(value: string): string {
  const time = Date.parse(value);
  if (!Number.isFinite(time)) {
    return value;
  }
  return new Date(time).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function getExitShortcutLabel(): string {
  if (process.platform === "darwin") {
    return "\u2303C";
  }
  if (process.platform === "win32") {
    return "ctrl+c";
  }
  return "control+c";
}

function getAttachImageShortcutLabel(): string {
  if (process.platform === "darwin") {
    return "control+v";
  }
  return "ctrl+v";
}

function isImageAttachKeybind(
  character: string,
  key: { ctrl?: boolean },
): boolean {
  const normalized = (character ?? "").toLowerCase();
  if (normalized !== "v") {
    return false;
  }
  // Match Codex behavior: paste images on Ctrl+V.
  return key.ctrl === true;
}

function readClipboardText(): string {
  const run = (command: string, args: string[]): string => {
    try {
      const result = spawnSync(command, args, {
        encoding: "utf8",
        maxBuffer: 1024 * 1024,
      });
      if (result.status !== 0) {
        return "";
      }
      return (result.stdout ?? "").trim();
    } catch {
      return "";
    }
  };

  if (process.platform === "darwin") {
    return run("pbpaste", []);
  }
  if (process.platform === "win32") {
    return run("powershell", ["-NoProfile", "-Command", "Get-Clipboard -Raw"]);
  }
  return run("sh", [
    "-lc",
    "(wl-paste -n 2>/dev/null || xclip -selection clipboard -o 2>/dev/null || xsel --clipboard --output 2>/dev/null)",
  ]);
}

function readClipboardImageAttachment():
  | { ok: true; image: ChatImageAttachment }
  | { ok: false; error: string } {
  const fromImageBytes = readClipboardImagePngBytes();
  if (fromImageBytes) {
    if (fromImageBytes.length > MAX_IMAGE_FILE_BYTES) {
      return {
        ok: false,
        error: `clipboard image too large (${formatByteSize(fromImageBytes.length)}). max is ${formatByteSize(MAX_IMAGE_FILE_BYTES)}`,
      };
    }
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    return {
      ok: true,
      image: {
        path: `clipboard-image-${stamp}.png`,
        mimeType: "image/png",
        dataUrl: `data:image/png;base64,${fromImageBytes.toString("base64")}`,
        byteSize: fromImageBytes.length,
      },
    };
  }

  const clipboardText = readClipboardText();
  if (clipboardText) {
    const fromPath = loadImageAttachment(clipboardText);
    if (fromPath.ok) {
      return fromPath;
    }
  }

  return { ok: false, error: "clipboard has no image (or image path)." };
}

function readClipboardImagePngBytes(): Buffer | null {
  if (process.platform !== "darwin") {
    return null;
  }

  const viaPngPaste = spawnBuffer("pngpaste", ["-"]);
  if (viaPngPaste && viaPngPaste.length > 0) {
    return viaPngPaste;
  }

  const viaAppleScriptPng = readClipboardImageViaAppleScript("PNGf");
  if (viaAppleScriptPng && viaAppleScriptPng.length > 0) {
    return viaAppleScriptPng;
  }

  return null;
}

function readClipboardImageViaAppleScript(classCode: "PNGf"): Buffer | null {
  const script = `the clipboard as «class ${classCode}»`;
  const raw = spawnText("osascript", ["-e", script], {
    timeoutMs: 900,
    maxBuffer: CLIPBOARD_OSASCRIPT_MAX_BUFFER,
  });
  if (!raw) {
    return null;
  }

  const match = raw.match(/«data\s+[A-Za-z0-9]{4}([A-Fa-f0-9\s]+)»/s);
  if (!match?.[1]) {
    return null;
  }
  const hex = match[1].replace(/\s+/g, "");
  if (!hex || hex.length % 2 !== 0) {
    return null;
  }

  try {
    return Buffer.from(hex, "hex");
  } catch {
    return null;
  }
}

function spawnBuffer(command: string, args: string[]): Buffer | null {
  try {
    const result = spawnSync(command, args, {
      maxBuffer: CLIPBOARD_IMAGE_CAPTURE_MAX_BUFFER,
      timeout: 600,
    });
    if (result.status !== 0) {
      return null;
    }
    const output = result.stdout;
    if (!output || (Buffer.isBuffer(output) && output.length === 0)) {
      return null;
    }
    return Buffer.isBuffer(output) ? output : Buffer.from(output);
  } catch {
    return null;
  }
}

function spawnText(
  command: string,
  args: string[],
  opts?: { timeoutMs?: number; maxBuffer?: number },
): string | null {
  try {
    const result = spawnSync(command, args, {
      encoding: "utf8",
      timeout: opts?.timeoutMs ?? 600,
      maxBuffer: opts?.maxBuffer ?? CLIPBOARD_IMAGE_CAPTURE_MAX_BUFFER,
    });
    if (result.status !== 0) {
      return null;
    }
    const text = (result.stdout ?? "").trim();
    return text || null;
  } catch {
    return null;
  }
}

function loadImageAttachment(rawPath: string): { ok: true; image: ChatImageAttachment } | { ok: false; error: string } {
  const resolvedPath = resolveImagePath(rawPath);
  if (!resolvedPath) {
    return { ok: false, error: "no file path provided" };
  }
  if (!fs.existsSync(resolvedPath)) {
    return { ok: false, error: `file not found: ${resolvedPath}` };
  }

  let stats: fs.Stats;
  try {
    stats = fs.statSync(resolvedPath);
  } catch {
    return { ok: false, error: `unable to stat file: ${resolvedPath}` };
  }
  if (!stats.isFile()) {
    return { ok: false, error: `not a file: ${resolvedPath}` };
  }
  if (stats.size > MAX_IMAGE_FILE_BYTES) {
    return {
      ok: false,
      error: `image too large (${formatByteSize(stats.size)}). max is ${formatByteSize(MAX_IMAGE_FILE_BYTES)}`,
    };
  }

  const extension = path.extname(resolvedPath).toLowerCase();
  const mimeType = IMAGE_MIME_BY_EXT[extension];
  if (!mimeType) {
    return {
      ok: false,
      error: `unsupported image type: ${extension || "(no extension)"}. supported: ${Object.keys(IMAGE_MIME_BY_EXT).join(", ")}`,
    };
  }

  let bytes: Buffer;
  try {
    bytes = fs.readFileSync(resolvedPath);
  } catch {
    return { ok: false, error: `unable to read image file: ${resolvedPath}` };
  }

  const dataUrl = `data:${mimeType};base64,${bytes.toString("base64")}`;
  return {
    ok: true,
    image: {
      path: resolvedPath,
      mimeType,
      dataUrl,
      byteSize: stats.size,
    },
  };
}

function resolveImagePath(rawPath: string): string {
  const trimmed = rawPath.trim();
  if (!trimmed) {
    return "";
  }
  let normalized = trimmed;
  if (
    (normalized.startsWith("\"") && normalized.endsWith("\"")) ||
    (normalized.startsWith("'") && normalized.endsWith("'"))
  ) {
    normalized = normalized.slice(1, -1).trim();
  }
  if (normalized.startsWith("file://")) {
    try {
      normalized = decodeURIComponent(normalized.slice("file://".length));
    } catch {
      normalized = normalized.slice("file://".length);
    }
  }
  if (normalized.startsWith("~/")) {
    normalized = path.join(os.homedir(), normalized.slice(2));
  }
  if (path.isAbsolute(normalized)) {
    return normalized;
  }
  return path.resolve(process.cwd(), normalized);
}

function formatByteSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "0 b";
  }
  if (bytes < 1024) {
    return `${Math.floor(bytes)} b`;
  }
  const kb = bytes / 1024;
  if (kb < 1024) {
    return `${kb.toFixed(kb >= 100 ? 0 : kb >= 10 ? 1 : 2)} kb`;
  }
  const mb = kb / 1024;
  return `${mb.toFixed(mb >= 100 ? 0 : mb >= 10 ? 1 : 2)} mb`;
}

function appendImagePlaceholderToComposerInput(input: string, placeholder: string): string {
  const current = input;
  const currentLower = current.toLowerCase();
  if (currentLower.includes(placeholder.toLowerCase())) {
    return current;
  }
  if (!current.trim()) {
    return placeholder;
  }
  const separator = current.endsWith(" ") ? "" : " ";
  return `${current}${separator}${placeholder}`;
}

function consumeCtrlVEchoArtifact(value: string, previousInput: string): string {
  const candidates = [
    `${previousInput}v`,
    `${previousInput}V`,
    `v${previousInput}`,
    `V${previousInput}`,
  ];
  for (const candidate of candidates) {
    if (value === candidate) {
      return previousInput;
    }
  }

  if (!previousInput.trim() && (value === "v" || value === "V")) {
    return "";
  }
  return value;
}

function reconcilePendingImagesWithComposerText(
  text: string,
  images: ChatImageAttachment[],
): { text: string; images: ChatImageAttachment[] } {
  if (images.length === 0) {
    return { text, images };
  }

  const placeholderRegex = /\[image\s+(\d+)\]/gi;
  const referenced = new Set<number>();
  let match: RegExpExecArray | null;
  while ((match = placeholderRegex.exec(text)) !== null) {
    const parsedIndex = Number.parseInt(match[1] ?? "", 10);
    if (!Number.isFinite(parsedIndex) || parsedIndex < 1 || parsedIndex > images.length) {
      continue;
    }
    referenced.add(parsedIndex);
  }

  const keptOldIndexes: number[] = [];
  for (let index = 1; index <= images.length; index += 1) {
    if (referenced.has(index)) {
      keptOldIndexes.push(index);
    }
  }

  const remap = new Map<number, number>();
  keptOldIndexes.forEach((oldIndex, offset) => {
    remap.set(oldIndex, offset + 1);
  });

  const nextImages = keptOldIndexes.map((oldIndex) => images[oldIndex - 1]!).filter(Boolean);
  const normalizedText = text.replace(placeholderRegex, (full, rawIndex: string) => {
    const parsedIndex = Number.parseInt(rawIndex, 10);
    const mappedIndex = remap.get(parsedIndex);
    if (!mappedIndex) {
      return full;
    }
    return `[Image ${mappedIndex}]`;
  });

  return {
    text: normalizedText,
    images: nextImages,
  };
}

function areImageAttachmentListsEqual(
  left: ChatImageAttachment[],
  right: ChatImageAttachment[],
): boolean {
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index += 1) {
    const leftImage = left[index];
    const rightImage = right[index];
    if (!leftImage || !rightImage) {
      return false;
    }
    if (leftImage.path !== rightImage.path || leftImage.byteSize !== rightImage.byteSize) {
      return false;
    }
  }
  return true;
}

function appendMissingImagePlaceholders(prompt: string, imageCount: number): string {
  const base = prompt.trim();
  if (imageCount <= 0) {
    return base;
  }

  const labels = Array.from({ length: imageCount }, (_, index) => `[Image ${index + 1}]`);
  if (!base) {
    return labels.join(" ");
  }

  const baseLower = base.toLowerCase();
  const missing = labels.filter((label) => !baseLower.includes(label.toLowerCase()));
  if (missing.length === 0) {
    return base;
  }

  return `${base} ${missing.join(" ")}`.trim();
}

function parseThoughtTitle(rawThought: string): string {
  const text = rawThought.trim();
  if (!text) {
    return "reasoning";
  }

  const titleMatch = text.match(/^\*\*(.+?)\*\*/s);
  if (titleMatch?.[1]) {
    return titleMatch[1].trim().toLowerCase();
  }
  const firstLine = text.split("\n")[0]?.trim();
  return (firstLine || "reasoning").toLowerCase();
}

function appendAntigravityModelSyncFailureMessages(
  appendSystemMessage: (text: string) => void,
  rawMessage: string,
): void {
  const message = rawMessage.trim() || "unknown error";
  const cloudCodeStatus = extractCloudCodeStatus(message);
  const enableApiUrl = extractGoogleApiEnableUrl(message);
  const projectId = extractGoogleProjectId(message, enableApiUrl);

  if (isAntigravityApiDisabledError(message)) {
    const projectSuffix = projectId ? ` for project ${projectId}` : "";
    appendSystemMessage(`antigravity model sync blocked: cloud code api is disabled${projectSuffix}.`);
    if (enableApiUrl) {
      appendSystemMessage(`enable antigravity cloud code api: ${enableApiUrl}`);
    }
    return;
  }

  if (cloudCodeStatus) {
    appendSystemMessage(`antigravity model sync failed (cloudcode ${cloudCodeStatus}).`);
    return;
  }

  const compact = message.replace(/\s+/g, " ").trim();
  appendSystemMessage(`antigravity model sync failed: ${clipInline(compact, 180)}`);
}

function isAntigravityApiDisabledError(message: string): boolean {
  const normalized = message.toLowerCase();
  return (
    normalized.includes("cloud code private api") &&
    (normalized.includes("has not been used in project") || normalized.includes("is disabled"))
  );
}

function extractGoogleApiEnableUrl(message: string): string | null {
  const compact = message.replace(/\s+/g, " ").trim();
  const match = compact.match(/https:\/\/console\.developers\.google\.com\/apis\/api\/[^\s"'`<>)]+/i);
  if (!match) {
    return null;
  }
  return match[0].replace(/[.,;:!?]+$/g, "");
}

function extractGoogleProjectId(message: string, enableApiUrl: string | null): string | null {
  if (enableApiUrl) {
    try {
      const parsed = new URL(enableApiUrl);
      const project = parsed.searchParams.get("project")?.trim();
      if (project) {
        return project;
      }
    } catch {
      // ignore malformed URLs
    }
  }

  const projectMatch = message.match(/projects\/([a-z0-9-]+)/i);
  return projectMatch?.[1]?.trim() || null;
}

function extractCloudCodeStatus(message: string): string | null {
  const match = message.match(/\[cloudcode\s+(\d{3})\]/i);
  return match?.[1] ?? null;
}

function toAntigravityOpenAiModelOptions(models: AntigravityDiscoveredModel[]): ModelOption[] {
  const options: ModelOption[] = [];
  for (const model of models) {
    const id = model.id.trim();
    if (!id) {
      continue;
    }
    const thinking = antigravityModelToThinkingLevels(model);
    options.push({
      id,
      provider: "openai",
      displayProvider: "antigravity",
      label: model.label.trim() || modelIdToLabel(id),
      description: model.description.trim() || "antigravity catalog model",
      supportedThinkingLevels: thinking.supportedThinkingLevels,
      defaultThinkingLevel: thinking.defaultThinkingLevel,
    });
  }
  return options;
}

function mergeOpenAiAndAntigravityModels(baseOpenAi: ModelOption[], antigravityOpenAi: ModelOption[]): ModelOption[] {
  if (antigravityOpenAi.length === 0) {
    return baseOpenAi;
  }

  const merged = new Map<string, ModelOption>();
  for (const option of baseOpenAi) {
    merged.set(option.id, option);
  }

  for (const option of antigravityOpenAi) {
    const existing = merged.get(option.id);
    if (!existing) {
      merged.set(option.id, option);
      continue;
    }

    merged.set(option.id, {
      ...existing,
      displayProvider: existing.displayProvider ?? option.displayProvider,
      supportedThinkingLevels:
        existing.supportedThinkingLevels && existing.supportedThinkingLevels.length > 0
          ? existing.supportedThinkingLevels
          : option.supportedThinkingLevels,
      defaultThinkingLevel: existing.defaultThinkingLevel ?? option.defaultThinkingLevel,
    });
  }

  return Array.from(merged.values());
}

function resolveInitialOnboardingCompleted(persistedState: LoafPersistedState | null): boolean {
  if (!persistedState) {
    return false;
  }
  if (persistedState.onboardingCompleted === true) {
    return true;
  }
  if (persistedState.onboardingCompleted === false) {
    return false;
  }
  return true;
}

function buildAuthOptions(params: {
  hasOpenAi: boolean;
  hasOpenRouter: boolean;
  hasAntigravity: boolean;
  includeAntigravity: boolean;
}): AuthOption[] {
  const options: AuthOption[] = [
    {
      id: "openai",
      label: params.hasOpenAi ? "openai oauth (connected)" : "openai oauth",
      description: "chatgpt account login (codex auth flow)",
    },
    {
      id: "openrouter",
      label: params.hasOpenRouter ? "openrouter api key (configured)" : "openrouter api key",
      description: "enter your openrouter key in /auth",
    },
  ];

  if (params.includeAntigravity) {
    options.splice(1, 0, {
      id: "antigravity",
      label: params.hasAntigravity ? "antigravity oauth (connected)" : "antigravity oauth",
      description: "antigravity oauth flow (google cloud scopes)",
    });
  }

  return options;
}

function isAuthSelectionConfigured(
  selection: AuthSelection,
  params: {
    hasOpenAi: boolean;
    hasOpenRouter: boolean;
    hasAntigravity: boolean;
  },
): boolean {
  if (selection === "openai") {
    return params.hasOpenAi;
  }
  if (selection === "openrouter") {
    return params.hasOpenRouter;
  }
  return params.hasAntigravity;
}

function buildOnboardingAuthOptions(enabledProviders: AuthProvider[]): OnboardingOption[] {
  const hasOpenAi = enabledProviders.includes("openai");
  const hasOpenRouter = enabledProviders.includes("openrouter");
  return [
    {
      id: "auth_openai",
      label: hasOpenAi ? "openai oauth (connected)" : "connect openai oauth",
      description: "chatgpt account login for gpt models",
    },
    {
      id: "auth_openrouter",
      label: hasOpenRouter ? "openrouter api key (configured)" : "connect openrouter api key",
      description: "openrouter models and provider routing",
    },
    {
      id: "auth_continue",
      label: "continue to exa setup",
      description: "next step: configure exa search key",
    },
  ];
}

function resolveInitialEnabledProviders(params: {
  persistedProviders: AuthProvider[] | undefined;
  legacyProvider: AuthProvider | undefined;
  hasOpenAiToken: boolean;
  hasOpenRouterKey: boolean;
}): AuthProvider[] {
  const fromPersisted = dedupeAuthProviders(
    params.persistedProviders ??
    (params.legacyProvider ? [params.legacyProvider] : []),
  );
  if (fromPersisted.length > 0) {
    return fromPersisted;
  }

  const inferred: AuthProvider[] = [];
  if (params.hasOpenAiToken) {
    inferred.push("openai");
  }
  if (params.hasOpenRouterKey) {
    inferred.push("openrouter");
  }
  if (loafConfig.preferredAuthProvider === "openai" && params.hasOpenAiToken) {
    inferred.unshift("openai");
  }
  if (loafConfig.preferredAuthProvider === "openrouter" && params.hasOpenRouterKey) {
    inferred.unshift("openrouter");
  }
  return dedupeAuthProviders(inferred);
}

function resolveInitialModel(
  providers: AuthProvider[],
  candidate: string | undefined,
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
): string {
  if (providers.length === 0) {
    return "";
  }

  return resolveModelForEnabledProviders(providers, candidate, modelOptionsByProvider);
}

function getSelectableModelProviders(enabledProviders: AuthProvider[], hasAntigravityToken: boolean): AuthProvider[] {
  if (!hasAntigravityToken) {
    return enabledProviders;
  }
  return dedupeAuthProviders([...enabledProviders, "openai"]);
}

function resolveModelForEnabledProviders(
  providers: AuthProvider[],
  candidate: string | undefined,
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
): string {
  const availableModels = getModelOptionsForProviders(providers, modelOptionsByProvider);
  if (
    candidate &&
    availableModels.some((option) => option.id === candidate)
  ) {
    return candidate;
  }

  if (candidate) {
    const inferredProvider = findProviderForModel(candidate, availableModels);
    if (inferredProvider && providers.includes(inferredProvider)) {
      // Preserve persisted/provider-valid model ids even before remote catalog hydration.
      return candidate;
    }
  }

  const preferredProvider =
    loafConfig.preferredAuthProvider && providers.includes(loafConfig.preferredAuthProvider)
      ? loafConfig.preferredAuthProvider
      : providers[0];

  if (preferredProvider) {
    const providerModels = getModelOptionsForProvider(preferredProvider, modelOptionsByProvider);
    const envModel = getEnvModelForProvider(preferredProvider);
    if (providerModels.some((option) => option.id === envModel)) {
      return envModel;
    }
    if (providerModels[0]?.id) {
      return providerModels[0].id;
    }
  }

  return availableModels[0]?.id ?? "";
}

function getEnvModelForProvider(provider: AuthProvider): string {
  if (provider === "openai") {
    return loafConfig.openaiModel.trim() || "gpt-4.1";
  }
  return loafConfig.openrouterModel.trim();
}

function getModelOptionsForProvider(
  provider: AuthProvider,
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
): ModelOption[] {
  const options = modelOptionsByProvider[provider];
  if (options.length > 0) {
    return options;
  }
  return getDefaultModelOptionsForProvider(provider);
}

function getModelOptionsForProviders(
  providers: AuthProvider[],
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
): ModelOption[] {
  const orderedProviders = AUTH_PROVIDER_ORDER.filter((provider) => providers.includes(provider));

  const combined: ModelOption[] = [];
  for (const provider of orderedProviders) {
    combined.push(...getModelOptionsForProvider(provider, modelOptionsByProvider));
  }
  return combined;
}

function findProviderForModel(
  modelId: string,
  availableModels: ModelOption[],
): AuthProvider | null {
  const direct = availableModels.find((option) => option.id === modelId)?.provider;
  if (direct) {
    return direct;
  }

  return inferProviderFromModelId(modelId);
}

function inferProviderFromModelId(modelId: string): AuthProvider | null {
  const normalized = modelIdToSlug(modelId).trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (
    normalized.startsWith("gpt-") ||
    normalized.startsWith("o1") ||
    normalized.startsWith("o3") ||
    normalized.startsWith("o4") ||
    normalized.includes("codex")
  ) {
    return "openai";
  }
  if (normalized.includes("gemini") || normalized.includes("claude") || normalized.includes("/")) {
    return "openrouter";
  }
  return null;
}

function dedupeAuthProviders(providers: AuthProvider[]): AuthProvider[] {
  const ordered: AuthProvider[] = [];
  for (const provider of providers) {
    if ((provider !== "openai" && provider !== "openrouter") || ordered.includes(provider)) {
      continue;
    }
    ordered.push(provider);
  }
  return ordered;
}

function resetConversationForProviderSwitch(params: {
  fromProvider: AuthProvider | null;
  toProvider: AuthProvider;
  setHistory: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  setMessages: React.Dispatch<React.SetStateAction<UiMessage[]>>;
  nextMessageId: () => number;
}): void {
  params.setHistory([]);
  const fromLabel = params.fromProvider ?? "unknown";
  params.setMessages([
    {
      id: params.nextMessageId(),
      kind: "system",
      text: `provider switched: ${fromLabel} -> ${params.toProvider}. conversation context was reset.`,
    },
  ]);
}

function getThinkingOptionsForModel(
  modelId: string,
  provider: AuthProvider,
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
): ThinkingOption[] {
  const modelOption = findModelOption(provider, modelId, modelOptionsByProvider);
  if (modelOption?.supportedThinkingLevels && modelOption.supportedThinkingLevels.length > 0) {
    return toThinkingOptions(modelOption.supportedThinkingLevels);
  }

  if (provider === "openai") {
    return THINKING_OPTIONS_OPENAI_DEFAULT;
  }
  return THINKING_OPTIONS_OPENROUTER_DEFAULT;
}

function normalizeThinkingForModel(
  modelId: string,
  provider: AuthProvider | null,
  thinking: ThinkingLevel,
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
): ThinkingLevel {
  if (!provider) {
    return thinking;
  }

  const thinkingOptions = getThinkingOptionsForModel(modelId, provider, modelOptionsByProvider);
  const supportedLevels = thinkingOptions.map((option) => option.id as ThinkingLevel);
  if (supportedLevels.includes(thinking)) {
    return thinking;
  }

  const modelOption = findModelOption(provider, modelId, modelOptionsByProvider);
  const defaultThinking = modelOption?.defaultThinkingLevel;
  if (defaultThinking && supportedLevels.includes(defaultThinking)) {
    return defaultThinking;
  }

  return supportedLevels[Math.floor(Math.max(0, (supportedLevels.length - 1) / 2))] ?? thinking;
}

function formatModelSummary(
  provider: AuthProvider,
  modelId: string,
  thinkingLevel: string,
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
  openRouterProvider: string,
): string {
  const providerLabel = getModelProviderDisplayLabelForModel(provider, modelId, modelOptionsByProvider);
  const options = getModelOptionsForProvider(provider, modelOptionsByProvider);
  const selectedOption = options.find((option) => option.id === modelId);
  const label = selectedOption?.label || modelIdToLabel(modelId);
  const isAntigravityModel = (selectedOption?.displayProvider ?? "").trim().toLowerCase() === "antigravity";
  if (isAntigravityModel) {
    return `${providerLabel} - ${label.toLowerCase()}`;
  }
  if (provider === "openrouter") {
    return `${providerLabel} - ${label.toLowerCase()} (${thinkingLevel.toLowerCase()}, provider: ${normalizeOpenRouterProviderSelection(openRouterProvider)})`;
  }
  return `${providerLabel} - ${label.toLowerCase()} (${thinkingLevel.toLowerCase()})`;
}

function findModelOption(
  provider: AuthProvider,
  modelId: string,
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
): ModelOption | undefined {
  return getModelOptionsForProvider(provider, modelOptionsByProvider).find((option) => option.id === modelId);
}

function getModelProviderDisplayLabel(option: ModelOption): string {
  const display = (option.displayProvider ?? "").trim().toLowerCase();
  if (display) {
    return display;
  }
  return option.provider;
}

function getModelProviderDisplayLabelForModel(
  provider: AuthProvider,
  modelId: string,
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
): string {
  const option = findModelOption(provider, modelId, modelOptionsByProvider);
  if (option) {
    return getModelProviderDisplayLabel(option);
  }
  return provider;
}

function normalizeOpenRouterProviderSelection(value: string | undefined | null): string {
  const normalized = (value ?? "").trim().toLowerCase();
  if (!normalized) {
    return OPENROUTER_PROVIDER_ANY_ID;
  }
  return normalized;
}

function getOpenRouterProviderOptions(
  modelId: string,
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
  overrideProviders?: string[],
): OpenRouterProviderOption[] {
  const modelOption = findModelOption("openrouter", modelId, modelOptionsByProvider);
  const providerTags = (overrideProviders ?? modelOption?.routingProviders ?? [])
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  const dedupedProviderTags = Array.from(new Set(providerTags)).sort((a, b) => a.localeCompare(b));

  const options: OpenRouterProviderOption[] = [
    {
      id: OPENROUTER_PROVIDER_ANY_ID,
      label: "any",
      description: "let openrouter auto-route and fallback",
    },
  ];

  for (const providerTag of dedupedProviderTags) {
    options.push({
      id: providerTag,
      label: providerTag,
      description: `force provider ${providerTag}`,
    });
  }

  return options;
}

function toThinkingOptions(levels: ThinkingLevel[]): ThinkingOption[] {
  const deduped: ThinkingLevel[] = [];
  for (const level of levels) {
    if (!deduped.includes(level)) {
      deduped.push(level);
    }
  }
  return deduped.map((level) => ({
    id: level,
    label: THINKING_OPTION_DETAILS[level].label,
    description: THINKING_OPTION_DETAILS[level].description,
  }));
}

function formatOpenAiUsageStatus(snapshot: OpenAiUsageSnapshot): string {
  const lines = ["codex usage limits:"];
  if (snapshot.planType) {
    lines.push(`plan: ${snapshot.planType}`);
  }
  lines.push(formatOpenAiUsageLine("5h", snapshot.primary));
  lines.push(formatOpenAiUsageLine("weekly", snapshot.secondary));
  return lines.join("\n");
}

function formatAntigravityUsageStatus(snapshot: AntigravityUsageSnapshot): string {
  const lines = ["antigravity usage limits:"];
  if (snapshot.models.length === 0) {
    lines.push("models: unavailable");
    return lines.join("\n");
  }

  const consolidated = consolidateAntigravityUsageModels(snapshot.models);
  const sorted = [...consolidated].sort((left, right) => {
    if (left.remainingPercent !== right.remainingPercent) {
      return left.remainingPercent - right.remainingPercent;
    }
    return left.name.localeCompare(right.name);
  });
  const displayRows = sorted.slice(0, 6);
  for (const quota of displayRows) {
    lines.push(formatAntigravityUsageLine(quota));
  }
  if (sorted.length > displayRows.length) {
    lines.push(`... +${sorted.length - displayRows.length} more model limits`);
  }
  return lines.join("\n");
}

function formatOpenAiUsageLine(label: string, window: OpenAiUsageSnapshot["primary"]): string {
  if (!window) {
    return `${label}: unavailable`;
  }

  const remaining = `${Math.round(window.remainingPercent)}% remaining`;
  const used = `${Math.round(window.usedPercent)}% used`;
  const reset =
    typeof window.resetAtEpochSeconds === "number"
      ? `, resets ${formatResetTimestamp(window.resetAtEpochSeconds)}`
      : "";
  return `${label}: ${remaining} (${used}${reset})`;
}

function formatAntigravityUsageLine(modelQuota: AntigravityUsageSnapshot["models"][number]): string {
  const resetAt = modelQuota.resetTime ? formatResetTimestampString(modelQuota.resetTime) : "unknown";
  return `${modelQuota.name}: ${modelQuota.remainingPercent}% remaining (resets ${resetAt})`;
}

function consolidateAntigravityUsageModels(
  models: AntigravityUsageSnapshot["models"],
): AntigravityUsageSnapshot["models"] {
  const groups = new Map<string, AntigravityUsageSnapshot["models"][number][]>();

  for (const model of models) {
    const normalizedName = model.name.trim().toLowerCase();
    const key = normalizedName.startsWith("claude")
      ? "claude"
      : normalizedName.startsWith("gemini")
        ? "gemini"
        : model.name;
    const existing = groups.get(key);
    if (existing) {
      existing.push(model);
    } else {
      groups.set(key, [model]);
    }
  }

  const consolidated: AntigravityUsageSnapshot["models"] = [];
  for (const [name, group] of groups) {
    const remainingPercent = group.reduce((min, row) => Math.min(min, row.remainingPercent), 100);
    consolidated.push({
      name,
      remainingPercent,
      resetTime: pickEarliestResetTime(group),
    });
  }

  return consolidated;
}

function pickEarliestResetTime(group: AntigravityUsageSnapshot["models"]): string | null {
  let earliest: { raw: string; epochMs: number } | null = null;
  let fallbackRaw: string | null = null;

  for (const row of group) {
    const raw = row.resetTime?.trim() ?? "";
    if (!raw) {
      continue;
    }
    if (!fallbackRaw) {
      fallbackRaw = raw;
    }
    const epochMs = Date.parse(raw);
    if (!Number.isFinite(epochMs)) {
      continue;
    }
    if (!earliest || epochMs < earliest.epochMs) {
      earliest = { raw, epochMs };
    }
  }

  return earliest?.raw ?? fallbackRaw;
}

function formatAntigravityUsageFailure(rawMessage: string): string {
  const message = rawMessage.trim() || "unknown error";
  const cloudCodeStatus = extractCloudCodeStatus(message);
  const enableApiUrl = extractGoogleApiEnableUrl(message);
  const projectId = extractGoogleProjectId(message, enableApiUrl);

  if (isAntigravityApiDisabledError(message)) {
    const projectSuffix = projectId ? ` for project ${projectId}` : "";
    if (enableApiUrl) {
      return `antigravity usage limits blocked: cloud code api is disabled${projectSuffix}.\nenable antigravity cloud code api: ${enableApiUrl}`;
    }
    return `antigravity usage limits blocked: cloud code api is disabled${projectSuffix}.`;
  }

  if (cloudCodeStatus) {
    return `antigravity usage limits failed (cloudcode ${cloudCodeStatus}).`;
  }

  return `antigravity usage limits failed: ${clipInline(message.replace(/\s+/g, " ").trim(), 180)}`;
}

function formatResetTimestamp(epochSeconds: number): string {
  const date = new Date(epochSeconds * 1000);
  if (Number.isNaN(date.getTime())) {
    return "unknown";
  }
  return date.toLocaleString();
}

function formatResetTimestampString(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "unknown";
  }
  return date.toLocaleString();
}

function MarkdownText({ text, prefix = "" }: { text: string; prefix?: string }) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  let usedPrefix = false;
  let inCodeBlock = false;
  let codeBlockLanguage = "";

  return (
    <Box flexDirection="column">
      {lines.map((line, index) => {
        const trimmed = line.trim();
        const linePrefix =
          trimmed.length > 0
            ? !usedPrefix
              ? prefix
              : " ".repeat(prefix.length)
            : "";
        if (trimmed.length > 0) {
          usedPrefix = true;
        }

        if (/^```/.test(trimmed)) {
          if (!inCodeBlock) {
            codeBlockLanguage = trimmed.replace(/^```+/, "").trim().toLowerCase();
            inCodeBlock = true;
          } else {
            inCodeBlock = false;
            codeBlockLanguage = "";
          }
          return null;
        }

        if (inCodeBlock) {
          const codeColor = codeBlockLanguage === "text" ? "gray" : "yellow";
          return (
            <Text key={`line-${index}`} color={codeColor}>
              {linePrefix}
              {line}
            </Text>
          );
        }

        if (!trimmed) {
          return <Text key={`line-${index}`}> </Text>;
        }

        if (/^---+$/.test(trimmed)) {
          return (
            <Text key={`line-${index}`} color="gray">
              {linePrefix}
              {"-".repeat(40)}
            </Text>
          );
        }

        const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
        if (headingMatch) {
          return (
            <Text key={`line-${index}`} color="cyanBright" bold>
              {linePrefix}
              {renderInlineMarkdown(headingMatch[2], `h-${index}`)}
            </Text>
          );
        }

        const bulletMatch = line.match(/^(\s*)[-*]\s+(.+)$/);
        if (bulletMatch) {
          return (
            <Text key={`line-${index}`} color="white">
              {linePrefix}
              {" ".repeat(bulletMatch[1]?.length ?? 0)}* {renderInlineMarkdown(bulletMatch[2], `b-${index}`)}
            </Text>
          );
        }

        const quoteMatch = line.match(/^\s*>\s+(.+)$/);
        if (quoteMatch) {
          return (
            <Text key={`line-${index}`} color="gray">
              {linePrefix}
              | {renderInlineMarkdown(quoteMatch[1], `q-${index}`)}
            </Text>
          );
        }

        return (
          <Text key={`line-${index}`} color="white">
            {linePrefix}
            {renderInlineMarkdown(line, `p-${index}`)}
          </Text>
        );
      })}
    </Box>
  );
}

const MemoizedMarkdownText = React.memo(MarkdownText);

function renderInlineMarkdown(input: string, keyPrefix: string): React.ReactNode[] {
  const text = normalizeInlineMarkdown(input);
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g);

  const nodes: React.ReactNode[] = [];
  let index = 0;
  for (const token of tokens) {
    if (!token) {
      continue;
    }

    if (token.startsWith("**") && token.endsWith("**")) {
      nodes.push(
        <Text key={`${keyPrefix}-${index++}`} bold>
          {token.slice(2, -2)}
        </Text>,
      );
      continue;
    }

    if (token.startsWith("`") && token.endsWith("`")) {
      nodes.push(
        <Text key={`${keyPrefix}-${index++}`} color="yellow">
          {token.slice(1, -1)}
        </Text>,
      );
      continue;
    }

    if (token.startsWith("*") && token.endsWith("*")) {
      nodes.push(
        <Text key={`${keyPrefix}-${index++}`} italic>
          {token.slice(1, -1)}
        </Text>,
      );
      continue;
    }

    nodes.push(<Text key={`${keyPrefix}-${index++}`}>{token}</Text>);
  }

  return nodes;
}

function normalizeInlineMarkdown(input: string): string {
  const withoutLinks = input.replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1 ($2)");
  return withoutLinks;
}

function buildRuntimeSystemInstruction(params: {
  baseInstruction: string;
  hasExaSearch: boolean;
  skillInstructionBlock?: string;
}): string {
  const base = params.baseInstruction.trim();
  const sections = [base];

  if (params.hasExaSearch && !base.toLowerCase().includes("search_web")) {
    sections.push(SEARCH_WEB_PROMPT_EXTENSION);
  }
  sections.push(OS_PROMPT_EXTENSION);

  const skillInstructionBlock = params.skillInstructionBlock?.trim() ?? "";
  if (skillInstructionBlock) {
    sections.push(skillInstructionBlock);
  }

  return sections.filter(Boolean).join("\n\n").trim();
}

function buildOsPromptExtension(): string {
  const platform = process.platform;
  const arch = process.arch;
  const osName =
    platform === "darwin"
      ? "macos"
      : platform === "win32"
        ? "windows"
        : platform === "linux"
          ? "linux"
          : platform;
  return `current host os: ${osName} (platform=${platform}, arch=${arch}). tailor commands and paths for this os.`;
}

export async function startTuiApp(): Promise<void> {
  try {
    const customTools = await loadCustomTools();
    if (customTools.loaded.length > 0) {
      const names = customTools.loaded.map((tool) => tool.name).join(", ");
    }
    for (const error of customTools.errors) {
      console.warn(`[loaf] custom tool warning: ${error}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`[loaf] custom tools initialization failed: ${message}`);
  }

  render(<App />);
}
