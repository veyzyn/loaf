import React, { useEffect, useMemo, useRef, useState } from "react";
import { Box, Newline, render, Text, useApp, useInput } from "ink";
import TextInput from "ink-text-input";
import { loafConfig, type AuthProvider, type ThinkingLevel } from "./config.js";
import {
  clearPersistedConfig,
  loadPersistedState,
  savePersistedState,
  type LoafPersistedState,
} from "./persistence.js";
import { getPersistedOpenAiChatgptAuth, runOpenAiOauthLogin } from "./openai-oauth.js";
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
import { runOpenAiInferenceStream } from "./openai.js";
import { listOpenRouterProvidersForModel, runOpenRouterInferenceStream } from "./openrouter.js";
import { configureBuiltinTools, defaultToolRegistry, loadCustomTools } from "./tools/index.js";
import type { ChatMessage, DebugEvent } from "./chat-types.js";
import {
  buildSkillPromptContext,
  loadSkillsCatalog,
  mapMessagesForModel,
  type SkillDefinition,
} from "./skills/index.js";

type UiMessage = {
  id: number;
  kind: "user" | "assistant" | "system";
  text: string;
};

type AuthOption = {
  id: AuthProvider;
  label: string;
  description: string;
};

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

const AUTH_OPTIONS: AuthOption[] = [
  {
    id: "openai",
    label: "openai oauth",
    description: "chatgpt account login (codex auth flow)",
  },
  {
    id: "openrouter",
    label: "openrouter api key",
    description: "enter your openrouter key in /auth",
  },
];

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
  { name: "/history", description: "resume a saved chat (/history, /history last, /history <id>)" },
  { name: "/skills", description: "list available skills from ~/.loaf/skills" },
  { name: "/tools", description: "list registered tools" },
  { name: "/clear", description: "clear conversation messages" },
  { name: "/quit", description: "exit loaf" },
  { name: "/help", description: "show available commands" },
  { name: "/quit", description: "exit loaf" },
  { name: "/exit", description: "exit loaf" },
];

const SUPER_DEBUG_COMMAND = "/superdebug-69";
const MAX_INPUT_HISTORY = 200;
const MAX_VISIBLE_MESSAGES = 14;
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
  const initialOpenAiAuth = getPersistedOpenAiChatgptAuth();
  const initialEnabledProviders = resolveInitialEnabledProviders({
    persistedProviders: persistedState?.authProviders,
    legacyProvider: persistedState?.authProvider,
    hasOpenAiToken: Boolean(initialOpenAiAuth?.accessToken),
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
  const [superDebug, setSuperDebug] = useState(false);
  const [onboardingCompleted, setOnboardingCompleted] = useState<boolean>(initialOnboardingCompleted);
  const [enabledProviders, setEnabledProviders] = useState<AuthProvider[]>(initialEnabledProviders);
  const [openAiAccessToken, setOpenAiAccessToken] = useState(initialOpenAiAuth?.accessToken ?? "");
  const [openAiAccountId, setOpenAiAccountId] = useState<string | null>(initialOpenAiAuth?.accountId ?? null);
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
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [activeSession, setActiveSession] = useState<ChatSessionSummary | null>(null);
  const [inputHistory, setInputHistory] = useState<string[]>(initialInputHistory);
  const [inputHistoryIndex, setInputHistoryIndex] = useState<number | null>(null);
  const [inputHistoryDraft, setInputHistoryDraft] = useState("");
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const nextIdRef = useRef(1);
  const activeInferenceAbortControllerRef = useRef<AbortController | null>(null);
  const steeringQueueRef = useRef<ChatMessage[]>([]);
  const queuedPromptsRef = useRef<string[]>([]);
  const [queuedPromptsVersion, setQueuedPromptsVersion] = useState(0);
  const suppressNextSubmitRef = useRef(false);

  const nextMessageId = () => {
    const id = nextIdRef.current;
    nextIdRef.current += 1;
    return id;
  };

  const activeModelOptions = useMemo(
    () => getModelOptionsForProviders(enabledProviders, modelOptionsByProvider),
    [enabledProviders, modelOptionsByProvider],
  );

  const selectedModelProvider = useMemo(
    () => findProviderForModel(selectedModel, activeModelOptions) ?? null,
    [selectedModel, activeModelOptions],
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

  const skillSuggestions = useMemo(() => {
    if (!input.startsWith("$")) {
      return [] as SkillDefinition[];
    }
    const firstToken = input.trim().split(/\s+/)[0] ?? "";
    const query = firstToken.slice(1).trim().toLowerCase();
    if (!query) {
      return availableSkills;
    }
    return availableSkills.filter((skill) => skill.nameLower.startsWith(query));
  }, [input, availableSkills]);

  const suppressSkillSuggestions = Boolean(
    autocompletedSkillPrefix && input.startsWith(autocompletedSkillPrefix),
  );
  const showSkillSuggestions =
    input.startsWith("$") && skillSuggestions.length > 0 && !suppressSkillSuggestions;

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
    if (!input.startsWith("$")) {
      return;
    }
    const catalog = loadSkillsCatalog();
    setAvailableSkills(catalog.skills);
  }, [input]);

  useEffect(() => {
    if (autocompletedSkillPrefix && !input.startsWith(autocompletedSkillPrefix)) {
      setAutocompletedSkillPrefix(null);
    }
  }, [input, autocompletedSkillPrefix]);

  useEffect(() => {
    configureBuiltinTools({
      exaApiKey,
    });
  }, [exaApiKey]);

  useEffect(() => {
    savePersistedState({
      authProviders: enabledProviders,
      selectedModel,
      selectedThinking,
      openRouterApiKey,
      exaApiKey,
      selectedOpenRouterProvider,
      onboardingCompleted,
      inputHistory,
    });
  }, [
    enabledProviders,
    selectedModel,
    selectedThinking,
    openRouterApiKey,
    exaApiKey,
    selectedOpenRouterProvider,
    onboardingCompleted,
    inputHistory,
  ]);

  useEffect(() => {
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
            openai: result.models,
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
    setSelectedModel((currentModel) =>
      resolveModelForEnabledProviders(enabledProviders, currentModel, modelOptionsByProvider),
    );
  }, [enabledProviders, modelOptionsByProvider]);

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

  const appendToolEvents = (event: DebugEvent) => {
    if (event.stage !== "tool_results") {
      return;
    }

    const rows = formatToolRows(event.data);
    for (const row of rows) {
      appendSystemMessage(row);
    }
  };

  const queueSteeringMessage = (rawText: string): boolean => {
    const text = rawText.trim();
    if (!text) {
      return false;
    }
    steeringQueueRef.current.push({
      role: "user",
      text,
    });
    setInput("");
    setStatusLabel("steer queued...");
    return true;
  };

  const queuePendingPrompt = (rawText: string): boolean => {
    const text = rawText.trim();
    if (!text) {
      return false;
    }
    queuedPromptsRef.current.push(text);
    setQueuedPromptsVersion((current) => current + 1);
    setInput("");
    appendSystemMessage(
      `queued message (${queuedPromptsRef.current.length}): ${clipInline(text, 80)}`,
    );
    return true;
  };

  const interruptActiveInference = (): boolean => {
    const controller = activeInferenceAbortControllerRef.current;
    if (!controller || controller.signal.aborted) {
      return false;
    }
    setStatusLabel("interrupting...");
    controller.abort();
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
        options: AUTH_OPTIONS,
      });
    }
  }, [enabledProviders, exaApiKey, onboardingCompleted, selector]);

  const openAuthSelector = (returnToOnboarding = false) => {
    setInput("");
    const firstMissing = AUTH_OPTIONS.find((option) => !enabledProviders.includes(option.id));
    const defaultIndex = firstMissing
      ? AUTH_OPTIONS.findIndex((option) => option.id === firstMissing.id)
      : 0;
    setSelector({
      kind: "auth",
      title: "add auth provider",
      index: defaultIndex,
      options: AUTH_OPTIONS,
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

  const applyAuthProviderSelection = async (
    provider: AuthProvider,
    openRouterKeyOverride?: string,
    returnToOnboarding = false,
  ): Promise<boolean> => {
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
      setOpenRouterApiKey(key);

      try {
        const result = await discoverOpenRouterModelOptions({
          apiKey: key,
        });
        setModelOptionsByProvider((current) => ({
          ...current,
          openrouter: result.models,
        }));
        appendSystemMessage(`openrouter models synced: ${result.models.length} (${result.source})`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`openrouter model sync failed: ${message}`);
      }

      if (enabledProviders.includes(provider)) {
        appendSystemMessage("openrouter auth already enabled (api key updated).");
        return true;
      }
    } else {
      if (enabledProviders.includes(provider)) {
        appendSystemMessage(`${provider} auth already enabled.`);
        return true;
      }

      let accessToken = openAiAccessToken.trim();
      let accountId = openAiAccountId?.trim() || null;
      if (!accessToken) {
        setPending(true);
        setStatusLabel("starting oauth login...");
        appendSystemMessage("starting chatgpt account login...");
        try {
          const loginResult = await runOpenAiOauthLogin({
            onDeviceCode: (info) => {
              const expiresMinutes = Math.max(1, Math.ceil(info.expiresInSeconds / 60));
              setStatusLabel("waiting for device code confirmation...");
              appendSystemMessage("headless login detected.");
              appendSystemMessage(`open ${info.verificationUrl}`);
              appendSystemMessage(`enter code: ${info.userCode} (expires in ~${expiresMinutes} min)`);
            },
          });
          accessToken = loginResult.chatgptAuth.accessToken;
          accountId = loginResult.chatgptAuth.accountId;
          setOpenAiAccessToken(accessToken);
          setOpenAiAccountId(accountId);
          appendSystemMessage(
            loginResult.loginMethod === "device_code"
              ? "chatgpt oauth login complete (device code flow)."
              : "chatgpt oauth login complete.",
          );
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          appendSystemMessage(`chatgpt oauth login failed: ${message}`);
          return false;
        } finally {
          setPending(false);
          setStatusLabel("ready");
        }
      }
    }

    const nextEnabledProviders = dedupeAuthProviders([...enabledProviders, provider]);
    const nextModel = resolveModelForEnabledProviders(
      nextEnabledProviders,
      selectedModel,
      modelOptionsByProvider,
    );
    const nextModelProvider =
      findProviderForModel(nextModel, getModelOptionsForProviders(nextEnabledProviders, modelOptionsByProvider)) ??
      null;
    const nextThinking = normalizeThinkingForModel(
      nextModel,
      nextModelProvider,
      selectedThinking,
      modelOptionsByProvider,
    );
    setEnabledProviders(nextEnabledProviders);
    setSelectedModel(nextModel);
    setSelectedThinking(nextThinking);
    if (provider === "openrouter" && !selectedOpenRouterProvider.trim()) {
      setSelectedOpenRouterProvider(OPENROUTER_PROVIDER_ANY_ID);
    }
    appendSystemMessage(
      `auth provider added: ${provider}. enabled: ${nextEnabledProviders.join(", ")}`,
    );
    return true;
  };

  const applyExaApiKeySelection = (rawValue: string): "saved" | "skipped" | "invalid" => {
    const value = rawValue.trim();
    if (!value) {
      return "invalid";
    }

    if (value.toLowerCase() === "skip") {
      setExaApiKey("");
      appendSystemMessage("exa api key skipped. search_web will be unavailable.");
      return "skipped";
    }

    setExaApiKey(value);
    appendSystemMessage("exa api key saved. search_web is now available.");
    return "saved";
  };

  const openModelSelector = () => {
    if (!onboardingCompleted) {
      openOnboardingSelector();
      return;
    }
    if (enabledProviders.length === 0) {
      openAuthSelector(false);
      return;
    }

    setInput("");
    const modelOptions = getModelOptionsForProviders(enabledProviders, modelOptionsByProvider);
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
    if (currentSelector.kind === "openrouter_api_key" || currentSelector.kind === "exa_api_key") {
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
    const WINDOW_SIZE = 10;
    if (options.length <= WINDOW_SIZE) {
      return {
        startIndex: 0,
        activeIndex: Math.max(0, Math.min(index, Math.max(0, options.length - 1))),
        options,
      };
    }

    const activeIndex = Math.max(0, Math.min(index, options.length - 1));
    const halfWindow = Math.floor(WINDOW_SIZE / 2);
    const maxStart = Math.max(0, options.length - WINDOW_SIZE);
    const startIndex = Math.max(0, Math.min(activeIndex - halfWindow, maxStart));
    return {
      startIndex,
      activeIndex,
      options: options.slice(startIndex, startIndex + WINDOW_SIZE),
    };
  };

  const applyModelSelection = (params: {
    modelId: string;
    modelLabel: string;
    modelProvider: AuthProvider;
    thinkingLevel: ThinkingLevel;
    openRouterProvider?: string;
    bypassProviderSwitchWarning?: boolean;
  }) => {
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

    setSelectedModel(params.modelId);
    setSelectedThinking(params.thinkingLevel);

    if (params.modelProvider === "openrouter") {
      setSelectedOpenRouterProvider((params.openRouterProvider ?? OPENROUTER_PROVIDER_ANY_ID).trim() || OPENROUTER_PROVIDER_ANY_ID);
    }

    if (providerChanged) {
      resetConversationForProviderSwitch({
        fromProvider: selectedModelProvider,
        toProvider: params.modelProvider,
        setHistory,
        setMessages,
        nextMessageId,
      });
      setConversationProvider(null);
      setActiveSession(null);
    }

    const providerSuffix =
      params.modelProvider === "openrouter"
        ? ` | provider ${params.openRouterProvider ?? OPENROUTER_PROVIDER_ANY_ID}`
        : "";
    appendSystemMessage(
      `model updated: ${params.modelLabel} (${params.thinkingLevel.toLowerCase()})${providerSuffix}${providerChanged ? " - context reset for provider switch" : ""}`,
    );
    setInput("");
    setSelector(null);
  };

  const openHistorySelector = () => {
    const sessions = listChatSessions({ limit: 100 });
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
  };

  const resumeSession = (session: ChatSessionRecord) => {
    if (!enabledProviders.includes(session.provider)) {
      setEnabledProviders((current) => dedupeAuthProviders([...current, session.provider]));
    }

    setSelectedModel(session.model);
    setSelectedThinking(
      normalizeThinkingForModel(
        session.model,
        session.provider,
        session.thinkingLevel,
        modelOptionsByProvider,
      ),
    );
    if (session.provider === "openrouter") {
      setSelectedOpenRouterProvider(
        normalizeOpenRouterProviderSelection(session.openRouterProvider) || OPENROUTER_PROVIDER_ANY_ID,
      );
    }

    setHistory(session.messages);
    setMessages(chatHistoryToUiMessages(session.messages, nextMessageId));
    setConversationProvider(session.provider);
    setActiveSession({
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
    });
    setSelector(null);
    setInput("");

    appendSystemMessage(
      `resumed chat ${session.id.slice(0, 8)} (${session.provider}, ${session.messageCount} msgs).`,
    );

    if (session.provider === "openai" && !openAiAccessToken.trim()) {
      appendSystemMessage("this chat uses openai. run /auth before sending the next prompt.");
    }
    if (session.provider === "openrouter" && !openRouterApiKey.trim()) {
      appendSystemMessage("this chat uses openrouter. run /auth before sending the next prompt.");
    }
  };

  const runHistoryCommand = (args: string[]) => {
    if (args.length === 0 || args[0] === "list" || args[0] === "all") {
      openHistorySelector();
      return;
    }

    const first = args[0]!.toLowerCase();
    if (first === "last") {
      const latest = loadLatestChatSession();
      if (!latest) {
        appendSystemMessage("no saved chat history yet.");
        return;
      }
      resumeSession(latest);
      return;
    }

    const byId = loadChatSessionById(args[0]!);
    if (!byId) {
      appendSystemMessage(`no saved chat matched id: ${args[0]}`);
      openHistorySelector();
      return;
    }
    resumeSession(byId);
  };

  const applyCommand = (rawCommand: string) => {
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
      openAuthSelector(false);
      return;
    }

    if (command === "/onboarding") {
      openOnboardingSelector();
      return;
    }

    if (command === "/forgeteverything") {
      clearPersistedConfig();
      setEnabledProviders([]);
      setOpenAiAccessToken("");
      setOpenAiAccountId(null);
      setOpenRouterApiKey("");
      setExaApiKey("");
      setSelectedOpenRouterProvider(OPENROUTER_PROVIDER_ANY_ID);
      setModelOptionsByProvider(initialModelOptionsByProvider);
      setSelectedModel("");
      setSelectedThinking(loafConfig.thinkingLevel);
      setPending(false);
      setStatusLabel("ready");
      setConversationProvider(null);
      setHistory([]);
      setActiveSession(null);
      setInputHistory([]);
      setInputHistoryIndex(null);
      setInputHistoryDraft("");
      setOnboardingCompleted(false);
      setInput("");
      setSelector({
        kind: "onboarding",
        title: "onboarding 1/2 - auth providers",
        index: 0,
        options: buildOnboardingAuthOptions([]),
      });
      setMessages([
        {
          id: nextMessageId(),
          kind: "system",
          text: "all local config was cleared. onboarding restarted.",
        },
      ]);
      return;
    }

    if (command === "/model") {
      openModelSelector();
      return;
    }

    if (command === "/history") {
      runHistoryCommand(args);
      return;
    }

    if (command === "/clear") {
      setMessages([]);
      setHistory([]);
      setConversationProvider(null);
      setActiveSession(null);
      return;
    }

    if (command === "/skills") {
      const catalog = refreshSkillsCatalog();
      if (catalog.skills.length === 0) {
        appendSystemMessage(`no skills found in ${catalog.directory}`);
        return;
      }
      const lines = catalog.skills
        .map((skill) => `${skill.name} - ${skill.descriptionPreview}`)
        .join("\n");
      appendSystemMessage(`available skills (${catalog.skills.length}):\n${lines}`);
      return;
    }

    if (command === "/tools") {
      const lines = defaultToolRegistry
        .list()
        .map((tool) => `${tool.name} - ${tool.description}`)
        .join("\n");
      appendSystemMessage(`registered tools:\n${lines}`);
      return;
    }

    if (command === "/help") {
      const commandLines = COMMAND_OPTIONS.map((option) => `${option.name} - ${option.description}`).join("\n");
      appendSystemMessage(`available commands:\n${commandLines}`);
      return;
    }

    if (command === "/quit" || command === "/exit") {
      exit();
      return;
    }

    if (command === SUPER_DEBUG_COMMAND) {
      const next = !superDebug;
      setSuperDebug(next);
      appendSystemMessage(`superdebug: ${next ? "on" : "off"}`);
      return;
    }

    appendSystemMessage(`unknown command: ${command}`);
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

    setInput("");
    setInputHistoryIndex(null);
    setInputHistoryDraft("");
    applyCommand(commandPayload);
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
    setInput(replaceLeadingSkillTokenWithSuggestion(rawInput, suggestionName));
    setAutocompletedSkillPrefix(`$${suggestionName} `);
    // Remount controlled TextInput so cursor jumps to the end after autocomplete.
    setTextInputResetKey((current) => current + 1);
  };

  useInput((character, key) => {
    if (key.ctrl && character === "c") {
      exit();
      return;
    }

    if (!selector && pending && key.return && key.shift) {
      const trimmed = input.trim();
      if (!trimmed) {
        appendSystemMessage("enter a steer message first, then press shift+enter.");
        suppressNextSubmitRef.current = true;
        return;
      }

      const steerText = parseSteerCommand(trimmed) ?? trimmed;
      queueSteeringMessage(steerText);
      suppressNextSubmitRef.current = true;
      return;
    }

    if (key.escape && pending && activeInferenceAbortControllerRef.current) {
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
        if (selector.kind === "auth" && enabledProviders.length === 0) {
          return;
        }
        setInput("");
        setSelector(null);
        return;
      }

      if (key.upArrow || key.downArrow) {
        if (selector.kind === "openrouter_api_key" || selector.kind === "exa_api_key") {
          return;
        }
        setSelector((current) => {
          if (!current) {
            return current;
          }
          if (current.kind === "openrouter_api_key" || current.kind === "exa_api_key") {
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
            openAuthSelector(true);
            return;
          }
          if (onboardingChoice.id === "auth_openrouter") {
            openAuthSelector(true);
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
          const result = applyExaApiKeySelection(exaInput);
          if (result === "invalid") {
            appendSystemMessage("invalid exa api key input.");
            return;
          }
          setInput("");
          if (selector.returnToOnboarding) {
            setOnboardingCompleted(true);
            setSelector(null);
            appendSystemMessage("onboarding complete. you can use /auth and /model any time.");
          } else {
            setSelector(null);
          }
          return;
        }

        if (selector.kind === "openrouter_api_key") {
          const keyInput = input.trim();
          if (!keyInput) {
            appendSystemMessage("openrouter api key cannot be empty.");
            return;
          }
          void (async () => {
            const ok = await applyAuthProviderSelection(
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
            const ok = await applyAuthProviderSelection(
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
          const session = loadChatSession(historyChoice.session.rolloutPath);
          if (!session) {
            appendSystemMessage(`failed to load chat: ${historyChoice.session.id}`);
            return;
          }
          resumeSession(session);
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
          applyModelSelection({
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
              if (openRouterApiKey.trim()) {
                setPending(true);
                setStatusLabel("loading providers...");
                try {
                  discoveredProviders = await listOpenRouterProvidersForModel(openRouterApiKey, selector.modelId);
                  if (discoveredProviders.length > 0) {
                    setModelOptionsByProvider((current) => ({
                      ...current,
                      openrouter: current.openrouter.map((option) =>
                        option.id === selector.modelId
                          ? { ...option, routingProviders: discoveredProviders }
                          : option,
                      ),
                    }));
                  }
                } catch (error) {
                  const message = error instanceof Error ? error.message : String(error);
                  appendSystemMessage(`openrouter provider fetch failed: ${message}`);
                } finally {
                  setPending(false);
                  setStatusLabel("ready");
                }
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

          applyModelSelection({
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
          applyModelSelection({
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

  const visibleMessages = useMemo(() => messages.slice(-MAX_VISIBLE_MESSAGES), [messages]);

  const sendPrompt = async (prompt: string) => {
    const cleanPrompt = prompt.trim();
    if (!cleanPrompt || pending) {
      return;
    }
    if (!onboardingCompleted) {
      appendSystemMessage("finish onboarding first.");
      openOnboardingSelector();
      return;
    }
    const provider = selectedModelProvider;
    if (!provider) {
      appendSystemMessage("select a model from an enabled provider first.");
      openAuthSelector(false);
      return;
    }
    if (!enabledProviders.includes(provider)) {
      appendSystemMessage(`provider ${provider} is not enabled. run /auth to add it.`);
      openAuthSelector(false);
      return;
    }
    if (provider === "openai" && !openAiAccessToken.trim()) {
      appendSystemMessage("openai model selected, but no chatgpt oauth token is available. run /auth.");
      openAuthSelector(false);
      return;
    }
    if (provider === "openrouter" && !openRouterApiKey.trim()) {
      appendSystemMessage("openrouter model selected, but no openrouter api key is available. run /auth.");
      openAuthSelector(false);
      return;
    }
    const skillCatalog = refreshSkillsCatalog();
    const skillPromptContext = buildSkillPromptContext(cleanPrompt, skillCatalog.skills);
    if (skillPromptContext.selection.combined.length > 0) {
      appendSystemMessage(
        `skills applied: ${skillPromptContext.selection.combined.map((skill) => skill.name).join(", ")}`,
      );
    }

    setInput("");
    setInputHistory((current) => {
      const next = [...current, cleanPrompt];
      return next.slice(-MAX_INPUT_HISTORY);
    });
    setInputHistoryIndex(null);
    setInputHistoryDraft("");
    setPending(true);
    setStatusLabel(selectedThinking === "OFF" ? "drafting response..." : "thinking...");
    steeringQueueRef.current = [];
    const inferenceAbortController = new AbortController();
    activeInferenceAbortControllerRef.current = inferenceAbortController;

    const providerSwitchRequiresReset =
      conversationProvider !== null && conversationProvider !== provider && history.length > 0;
    const historyBase = providerSwitchRequiresReset ? [] : history;
    if (providerSwitchRequiresReset) {
      resetConversationForProviderSwitch({
        fromProvider: conversationProvider,
        toProvider: provider,
        setHistory,
        setMessages,
        nextMessageId,
      });
      setActiveSession(null);
    }

    const normalizedOpenRouterProvider =
      provider === "openrouter"
        ? normalizeOpenRouterProviderSelection(selectedOpenRouterProvider)
        : undefined;
    const needsNewSession =
      historyBase.length === 0 || !activeSession || activeSession.provider !== provider;
    let sessionForTurn = activeSession;
    if (needsNewSession) {
      try {
        sessionForTurn = createChatSession({
          provider,
          model: selectedModel,
          thinkingLevel: selectedThinking,
          openRouterProvider: normalizedOpenRouterProvider,
          titleHint: cleanPrompt,
        });
        setActiveSession(sessionForTurn);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        appendSystemMessage(`history save disabled for this turn: ${message}`);
        sessionForTurn = null;
        setActiveSession(null);
      }
    }

    const nextHistory = [...historyBase, { role: "user" as const, text: cleanPrompt }];
    const modelHistory: ChatMessage[] = [
      ...mapMessagesForModel(historyBase, skillCatalog.skills),
      { role: "user", text: skillPromptContext.modelPrompt },
    ];
    const nextUserMessage: UiMessage = {
      id: nextMessageId(),
      kind: "user",
      text: cleanPrompt,
    };
    setMessages((current) => [...current, nextUserMessage]);

    let assistantDraftText = "";
    const appliedSteeringMessages: ChatMessage[] = [];

    const appendAssistantDraftDelta = (deltaText: string) => {
      if (!deltaText) {
        return;
      }
      assistantDraftText += deltaText;
    };

    const drainSteeringMessages = (): ChatMessage[] => {
      const queued = steeringQueueRef.current;
      if (queued.length === 0) {
        return [];
      }
      steeringQueueRef.current = [];
      appliedSteeringMessages.push(...queued);
      setMessages((current) => [
        ...current,
        ...queued.map((message) => ({
          id: nextMessageId(),
          kind: "user" as const,
          text: message.text,
        })),
      ]);
      setStatusLabel("steer applied; continuing...");
      return queued;
    };

    try {
      const handleChunk = (chunk: { thoughts: string[]; answerText: string }) => {
        for (const rawThought of chunk.thoughts) {
          const title = parseThoughtTitle(rawThought);
          setStatusLabel(`thinking: ${title}`);
        }

        if (chunk.answerText) {
          appendAssistantDraftDelta(chunk.answerText);
          setStatusLabel("drafting response...");
        }
      };

      const handleDebug = (event: DebugEvent) => {
        appendToolEvents(event);
        appendDebugEvent(event);
      };
      const runtimeSystemInstruction = buildRuntimeSystemInstruction({
        baseInstruction: loafConfig.systemInstruction,
        hasExaSearch: Boolean(exaApiKey.trim()),
        skillInstructionBlock: skillPromptContext.instructionBlock,
      });

      const result =
        provider === "openrouter"
          ? await runOpenRouterInferenceStream(
              {
                apiKey: openRouterApiKey,
                model: selectedModel,
                messages: modelHistory,
                thinkingLevel: selectedThinking,
                includeThoughts: selectedThinking !== "OFF",
                forcedProvider:
                  normalizeOpenRouterProviderSelection(selectedOpenRouterProvider) === OPENROUTER_PROVIDER_ANY_ID
                    ? null
                    : normalizeOpenRouterProviderSelection(selectedOpenRouterProvider),
                systemInstruction: runtimeSystemInstruction,
                signal: inferenceAbortController.signal,
                drainSteeringMessages,
              },
              handleChunk,
              handleDebug,
            )
          : await runOpenAiInferenceStream(
              {
                accessToken: openAiAccessToken,
                chatgptAccountId: openAiAccountId,
                model: selectedModel,
                messages: modelHistory,
                thinkingLevel: selectedThinking,
                includeThoughts: selectedThinking !== "OFF",
                systemInstruction: runtimeSystemInstruction,
                signal: inferenceAbortController.signal,
                drainSteeringMessages,
              },
              handleChunk,
              handleDebug,
            );

      assistantDraftText = result.answer;
      const assistantMessage: ChatMessage = {
        role: "assistant",
        text: result.answer,
      };
      const savedHistory = [...nextHistory, ...appliedSteeringMessages, assistantMessage];
      setHistory(savedHistory);
      setConversationProvider(provider);

      if (sessionForTurn) {
        try {
          const updatedSession = writeChatSession({
            session: sessionForTurn,
            messages: savedHistory,
            provider,
            model: selectedModel,
            thinkingLevel: selectedThinking,
            openRouterProvider: normalizedOpenRouterProvider,
            titleHint: cleanPrompt,
          });
          setActiveSession(updatedSession);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          appendSystemMessage(`chat history save failed: ${message}`);
        }
      }

      setMessages((current) => [
        ...current,
        {
          id: nextMessageId(),
          kind: "assistant",
          text: result.answer,
        },
      ]);

    } catch (error) {
      if (isAbortError(error)) {
        const interruptedAssistant = assistantDraftText.trim();
        const interruptedHistoryBase = [...nextHistory, ...appliedSteeringMessages];
        const interruptedHistory = interruptedAssistant
          ? [...interruptedHistoryBase, { role: "assistant" as const, text: interruptedAssistant }]
          : interruptedHistoryBase;
        setHistory(interruptedHistory);
        setConversationProvider(provider);

        if (sessionForTurn) {
          try {
            const updatedSession = writeChatSession({
              session: sessionForTurn,
              messages: interruptedHistory,
              provider,
              model: selectedModel,
              thinkingLevel: selectedThinking,
              openRouterProvider: normalizedOpenRouterProvider,
              titleHint: cleanPrompt,
            });
            setActiveSession(updatedSession);
          } catch (sessionError) {
            const message = sessionError instanceof Error ? sessionError.message : String(sessionError);
            appendSystemMessage(`chat history save failed: ${message}`);
          }
        }

        if (interruptedAssistant) {
          setMessages((current) => [
            ...current,
            {
              id: nextMessageId(),
              kind: "assistant",
              text: interruptedAssistant,
            },
          ]);
        }

        appendSystemMessage(
          interruptedAssistant
            ? "response interrupted by user. partial output kept."
            : "response interrupted by user.",
        );
        return;
      }

      const message = error instanceof Error ? error.message : String(error);
      setMessages((current) => [
        ...current,
        {
          id: nextMessageId(),
          kind: "system",
          text: `inference error: ${message}`,
        },
      ]);
    } finally {
      const unappliedSteerCount = steeringQueueRef.current.length;
      steeringQueueRef.current = [];
      activeInferenceAbortControllerRef.current = null;
      setPending(false);
      setStatusLabel("ready");
      if (unappliedSteerCount > 0) {
        appendSystemMessage(
          `${unappliedSteerCount} steer message(s) were queued too late and not applied.`,
        );
      }
    }
  };

  useEffect(() => {
    if (pending) {
      return;
    }
    const nextQueuedPrompt = queuedPromptsRef.current.shift();
    if (!nextQueuedPrompt) {
      return;
    }
    setQueuedPromptsVersion((current) => current + 1);
    void sendPrompt(nextQueuedPrompt);
  }, [pending, queuedPromptsVersion]);

  return (
    <Box flexDirection="column" paddingX={1}>
      <Text color="cyanBright">loaf | beta</Text>
      <Text color="gray">
        auth: {enabledProviders.length > 0 ? enabledProviders.join(", ") : "not selected"} |{" "}
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
          <MessageRow key={message.id} message={message} />
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
                        <Text key={`${selector.kind}-${option.id}`} color={selected ? "magentaBright" : "gray"}>
                          {selected ? ">" : " "} {option.label} - {selector.kind === "model" ? (option as ModelOption).provider : option.description}
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
      {commandSuggestions.length > 0 && !selector && (
        <Box flexDirection="column" marginTop={1}>
          {commandSuggestions.map((suggestion, index) => (
            <Text key={suggestion.name} color={index === commandIndex ? "magentaBright" : "gray"}>
              {index === commandIndex ? ">" : " "} {suggestion.name} - {suggestion.description}
            </Text>
          ))}
          <Text color="gray">tab autocomplete | up/down navigate suggestions</Text>
        </Box>
      )}
      {showSkillSuggestions && !selector && (
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
            if (
              selector?.kind === "openrouter_api_key" ||
              selector?.kind === "exa_api_key" ||
              selector?.kind === "model" ||
              !selector
            ) {
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
              setInput(value);
            }
          }}
          onSubmit={(submitted) => {
            if (suppressNextSubmitRef.current) {
              suppressNextSubmitRef.current = false;
              return;
            }

            if (selector && selector.kind !== "openrouter_api_key" && selector.kind !== "exa_api_key") {
              return;
            }

            if (showSkillSuggestions && submitted.startsWith("$")) {
              const suggestion = skillSuggestions[skillIndex];
              if (suggestion && shouldAutocompleteSkillInputOnEnter(submitted, suggestion.nameLower)) {
                applySkillAutocomplete(submitted, suggestion.name);
                return;
              }
            }

            const trimmed = submitted.trim();
            if (!trimmed) {
              return;
            }

            if (selector?.kind === "openrouter_api_key" || selector?.kind === "exa_api_key") {
              return;
            }

            if (pending) {
              if (!activeInferenceAbortControllerRef.current) {
                appendSystemMessage("busy. wait for current operation to finish.");
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

function replaceLeadingSkillTokenWithSuggestion(rawInput: string, suggestionName: string): string {
  const trimmed = rawInput.trimStart();
  const firstSpaceIndex = trimmed.indexOf(" ");
  if (firstSpaceIndex < 0) {
    return `$${suggestionName} `;
  }
  const remainder = trimmed.slice(firstSpaceIndex).trim();
  if (!remainder) {
    return `$${suggestionName} `;
  }
  return `$${suggestionName} ${remainder}`;
}

function shouldAutocompleteSkillInputOnEnter(rawInput: string, selectedSkillNameLower: string): boolean {
  const trimmed = rawInput.trimStart();
  if (!trimmed.startsWith("$")) {
    return false;
  }

  const firstSpaceIndex = trimmed.indexOf(" ");
  const firstToken = firstSpaceIndex < 0 ? trimmed : trimmed.slice(0, firstSpaceIndex);
  const enteredSkillName = firstToken.slice(1).trim().toLowerCase();
  const trailingContent = firstSpaceIndex < 0 ? "" : trimmed.slice(firstSpaceIndex + 1).trim();
  const hasTrailingPromptText = trailingContent.length > 0;
  const isExactSkillMatch = enteredSkillName === selectedSkillNameLower;

  if (isExactSkillMatch && hasTrailingPromptText) {
    return false;
  }

  return true;
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

  const rows: string[] = [];
  for (const item of executed) {
    const record = (item ?? {}) as {
      name?: unknown;
      ok?: unknown;
      input?: unknown;
      result?: unknown;
      error?: unknown;
    };

    const name = typeof record.name === "string" && record.name ? record.name : "tool";
    const ok = Boolean(record.ok);
    const baseSummary = formatToolSummary(name, record.input, record.result);
    const summary = ok ? baseSummary : `${baseSummary} (failed)`;
    const detailBody = formatToolDetail(name, record.input, record.result);
    const errorLine = `error: ${typeof record.error === "string" ? record.error : "tool execution failed"}`;
    const detailRaw = ok
      ? detailBody
      : detailBody
        ? `${errorLine}\n${detailBody}`
        : errorLine;
    const detail = ok && shouldCollapseSuccessDetail(name)
      ? collapseRedundantSuccessDetail(summary, detailRaw)
      : detailRaw;

    rows.push(formatToolRow(summary, detail));
  }

  return rows;
}

function shouldCollapseSuccessDetail(name: string): boolean {
  return (
    name !== "create_persistent_tool" &&
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

function formatToolRow(summary: string, detail: string): string {
  const lines = detail.replace(/\r\n/g, "\n").split("\n");
  const firstLine = lines.find((line) => line.trim())?.trim() || "ok";
  return `${summary}\n  -> ${clipInline(firstLine, 180)}`;
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
        <Text color="blueBright">{GLYPH_USER}{message.text}</Text>
      </Box>
    );
  }
  if (message.kind === "assistant") {
    return (
      <Box flexDirection="column" marginBottom={1}>
        <MarkdownText text={message.text} prefix={GLYPH_ASSISTANT} />
      </Box>
    );
  }
  return (
    <Box flexDirection="column" marginBottom={1}>
      <MarkdownText text={message.text} prefix={GLYPH_SYSTEM} />
    </Box>
  );
}

function chatHistoryToUiMessages(history: ChatMessage[], nextMessageId: () => number): UiMessage[] {
  return history.map((message) => ({
    id: nextMessageId(),
    kind: message.role,
    text: message.text,
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
  const orderedProviders = AUTH_OPTIONS
    .map((option) => option.id)
    .filter((provider) => providers.includes(provider));

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
  if (availableModels.length === 0) {
    return null;
  }

  const direct = availableModels.find((option) => option.id === modelId)?.provider;
  if (direct) {
    return direct;
  }

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
  const options = getModelOptionsForProvider(provider, modelOptionsByProvider);
  const label = options.find((option) => option.id === modelId)?.label || modelIdToLabel(modelId);
  if (provider === "openrouter") {
    return `${provider} - ${label.toLowerCase()} (${thinkingLevel.toLowerCase()}, provider: ${normalizeOpenRouterProviderSelection(openRouterProvider)})`;
  }
  return `${provider} - ${label.toLowerCase()} (${thinkingLevel.toLowerCase()})`;
}

function findModelOption(
  provider: AuthProvider,
  modelId: string,
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
): ModelOption | undefined {
  return getModelOptionsForProvider(provider, modelOptionsByProvider).find((option) => option.id === modelId);
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

void startApp();

async function startApp(): Promise<void> {
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
