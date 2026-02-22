import { randomUUID } from "node:crypto";
import path from "node:path";
import { loafConfig, type AuthProvider, type ThinkingLevel } from "../config.js";
import {
  clearPersistedConfig,
  loadPersistedRuntimeSecrets,
  loadPersistedState,
  persistRuntimeSecrets,
  savePersistedState,
  type LoafPersistedState,
} from "../persistence.js";
import { loadPersistedOpenAiChatgptAuth, runOpenAiOauthLogin } from "../openai-oauth.js";
import {
  fetchAntigravityProfileData,
  loadPersistedAntigravityOauthTokenInfo,
  runAntigravityOauthLogin,
  type AntigravityOauthProfile,
  type AntigravityOauthTokenInfo,
} from "../antigravity-oauth.js";
import {
  antigravityModelToThinkingLevels,
  discoverAntigravityModelOptions,
  fetchAntigravityUsageSnapshot,
  type AntigravityDiscoveredModel,
  type AntigravityUsageSnapshot,
} from "../antigravity-models.js";
import { runAntigravityInferenceStream } from "../antigravity.js";
import {
  createChatSession,
  listChatSessions,
  loadChatSession,
  loadChatSessionById,
  loadLatestChatSession,
  type ChatSessionRecord,
  type ChatSessionSummary,
  writeChatSession,
} from "../chat-history.js";
import {
  discoverOpenAiModelOptions,
  discoverOpenRouterModelOptions,
  getDefaultModelOptionsForProvider,
  modelIdToLabel,
  modelIdToSlug,
  type ModelOption,
} from "../models.js";
import { fetchOpenAiUsageSnapshot, runOpenAiInferenceStream, type OpenAiUsageSnapshot } from "../openai.js";
import { listOpenRouterProvidersForModel, runOpenRouterInferenceStream } from "../openrouter.js";
import { configureBuiltinTools, defaultToolRegistry, loadCustomTools } from "../tools/index.js";
import type { ChatImageAttachment, ChatMessage, DebugEvent, StreamChunk } from "../chat-types.js";
import { buildSkillPromptContext, loadSkillsCatalog, mapMessagesForModel, type SkillDefinition } from "../skills/index.js";
import {
  loadRuntimeImageAttachments,
  normalizeRuntimeImageInputs,
  type RuntimeImageInput,
} from "./images.js";

export type RuntimeUiMessage = {
  id: number;
  kind: "user" | "assistant" | "system";
  text: string;
  images?: ChatImageAttachment[];
};

export type RuntimeTurnQueueItem = {
  id: string;
  text: string;
  images: RuntimeImageInput[];
  enqueuedAt: string;
};

export type RuntimeSessionState = {
  id: string;
  createdAt: string;
  updatedAt: string;
  pending: boolean;
  statusLabel: string;
  messages: RuntimeUiMessage[];
  history: ChatMessage[];
  queue: RuntimeTurnQueueItem[];
  pendingSteerCount: number;
  conversationProvider: AuthProvider | null;
  activeSessionId: string | null;
};

export type RuntimeEvent = {
  type: string;
  payload: Record<string, unknown>;
};

export type RuntimeSnapshot = {
  auth: {
    enabledProviders: AuthProvider[];
    hasOpenAiToken: boolean;
    hasOpenRouterKey: boolean;
    hasAntigravityToken: boolean;
    antigravityProfile: AntigravityOauthProfile | null;
  };
  onboarding: {
    completed: boolean;
  };
  model: {
    selectedModel: string;
    selectedThinking: ThinkingLevel;
    selectedOpenRouterProvider: string;
    selectedProvider: AuthProvider | null;
  };
  sessions: {
    count: number;
    ids: string[];
  };
  skills: {
    count: number;
    directories: string[];
  };
};

export type RuntimeInitOptions = {
  rpcMode?: boolean;
};

type RuntimeSession = {
  id: string;
  createdAt: string;
  updatedAt: string;
  pending: boolean;
  statusLabel: string;
  messages: RuntimeUiMessage[];
  history: ChatMessage[];
  activeSession: ChatSessionSummary | null;
  conversationProvider: AuthProvider | null;
  queuedPrompts: RuntimeTurnQueueItem[];
  steeringQueue: ChatMessage[];
  activeAbortController: AbortController | null;
};

const OPENROUTER_PROVIDER_ANY_ID = "any";
const MAX_INPUT_HISTORY = 200;

const AUTH_PROVIDER_ORDER: AuthProvider[] = ["openai", "openrouter"];

const THINKING_OPTIONS_OPENAI_DEFAULT: ThinkingLevel[] = [
  "OFF",
  "MINIMAL",
  "LOW",
  "MEDIUM",
  "HIGH",
  "XHIGH",
];
const THINKING_OPTIONS_OPENROUTER_DEFAULT: ThinkingLevel[] = [
  "OFF",
  "MINIMAL",
  "LOW",
  "MEDIUM",
  "HIGH",
];

const SEARCH_WEB_PROMPT_EXTENSION = [
  "for facts that may be stale/uncertain (dates, releases, pricing, availability, docs), proactively use search_web.",
  "prefer at least one search_web pass before answering factual questions from memory.",
  "if search results are weak or conflicting, refine the query and search_web again before switching tools.",
  "for factual web lookups, call search_web first and use returned highlights before writing custom scrapers.",
].join("\n");

const OS_PROMPT_EXTENSION = buildOsPromptExtension();

export class LoafCoreRuntime {
  private readonly rpcMode: boolean;

  private persistedState: LoafPersistedState | null = null;
  private enabledProviders: AuthProvider[] = [];
  private openAiAccessToken = "";
  private openAiAccountId: string | null = null;
  private antigravityOauthTokenInfo: AntigravityOauthTokenInfo | null = null;
  private antigravityOauthProfile: AntigravityOauthProfile | null = null;
  private antigravityOpenAiModelOptions: ModelOption[] = [];
  private openRouterApiKey = "";
  private exaApiKey = "";
  private selectedOpenRouterProvider = OPENROUTER_PROVIDER_ANY_ID;
  private modelOptionsByProvider: Record<AuthProvider, ModelOption[]> = {
    openai: getDefaultModelOptionsForProvider("openai"),
    openrouter: getDefaultModelOptionsForProvider("openrouter"),
  };
  private selectedModel = "";
  private selectedThinking: ThinkingLevel = loafConfig.thinkingLevel;
  private onboardingCompleted = false;
  private inputHistory: string[] = [];
  private availableSkills: SkillDefinition[] = [];
  private skillsDirectories: string[] = [];
  private listeners = new Set<(event: RuntimeEvent) => void>();
  private sessions = new Map<string, RuntimeSession>();
  private nextMessageId = 1;
  private superDebug = false;
  private shuttingDown = false;

  private constructor(options: RuntimeInitOptions = {}) {
    this.rpcMode = options.rpcMode === true;
  }

  static async create(options: RuntimeInitOptions = {}): Promise<LoafCoreRuntime> {
    const runtime = new LoafCoreRuntime(options);
    await runtime.initialize();
    return runtime;
  }

  onEvent(listener: (event: RuntimeEvent) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private emit(type: string, payload: Record<string, unknown>): void {
    const event: RuntimeEvent = {
      type,
      payload,
    };
    for (const listener of this.listeners) {
      listener(event);
    }
  }

  async shutdown(reason?: string): Promise<{ accepted: true; reason?: string }> {
    this.shuttingDown = true;
    for (const session of this.sessions.values()) {
      session.activeAbortController?.abort();
      session.activeAbortController = null;
      session.pending = false;
      session.statusLabel = "ready";
      session.queuedPrompts = [];
      session.steeringQueue = [];
    }

    this.emit("state.changed", {
      reason: reason ?? "shutdown",
      snapshot: this.getState(),
    });

    return {
      accepted: true,
      reason,
    };
  }

  async initialize(): Promise<void> {
    this.persistedState = loadPersistedState();
    const initialEnabledProviders = resolveInitialEnabledProviders({
      persistedProviders: this.persistedState?.authProviders,
      legacyProvider: this.persistedState?.authProvider,
      hasOpenAiToken: false,
      hasOpenRouterKey: Boolean((this.persistedState?.openRouterApiKey ?? loafConfig.openrouterApiKey).trim()),
    });

    this.enabledProviders = initialEnabledProviders;
    this.openRouterApiKey = this.persistedState?.openRouterApiKey ?? loafConfig.openrouterApiKey;
    this.exaApiKey = this.persistedState?.exaApiKey ?? loafConfig.exaApiKey;
    this.selectedOpenRouterProvider =
      this.persistedState?.selectedOpenRouterProvider ?? OPENROUTER_PROVIDER_ANY_ID;
    this.onboardingCompleted = resolveInitialOnboardingCompleted(this.persistedState);
    this.inputHistory = this.persistedState?.inputHistory ?? [];

    const initialModel = resolveInitialModel(
      this.enabledProviders,
      this.persistedState?.selectedModel,
      this.modelOptionsByProvider,
    );
    const initialModelProvider =
      findProviderForModel(
        initialModel,
        getModelOptionsForProviders(this.enabledProviders, this.modelOptionsByProvider),
      ) ?? null;

    this.selectedModel = initialModel;
    this.selectedThinking = normalizeThinkingForModel(
      initialModel,
      initialModelProvider,
      this.persistedState?.selectedThinking ?? loafConfig.thinkingLevel,
      this.modelOptionsByProvider,
    );

    const skillCatalog = loadSkillsCatalog();
    this.availableSkills = skillCatalog.skills;
    this.skillsDirectories = skillCatalog.directories;

    const runtimeSecrets = await loadPersistedRuntimeSecrets(this.persistedState);
    if (runtimeSecrets.openRouterApiKey) {
      this.openRouterApiKey = runtimeSecrets.openRouterApiKey;
      this.enabledProviders = dedupeAuthProviders([...this.enabledProviders, "openrouter"]);
    }
    if (runtimeSecrets.exaApiKey) {
      this.exaApiKey = runtimeSecrets.exaApiKey;
    }

    const openAiAuth = await loadPersistedOpenAiChatgptAuth();
    if (openAiAuth?.accessToken.trim()) {
      this.openAiAccessToken = openAiAuth.accessToken;
      this.openAiAccountId = openAiAuth.accountId ?? null;
      this.enabledProviders = dedupeAuthProviders([...this.enabledProviders, "openai"]);
    }

    const antigravityTokenInfo = await loadPersistedAntigravityOauthTokenInfo();
    if (antigravityTokenInfo?.accessToken.trim()) {
      this.antigravityOauthTokenInfo = antigravityTokenInfo;
      try {
        this.antigravityOauthProfile = await fetchAntigravityProfileData(antigravityTokenInfo.accessToken);
      } catch {
        this.antigravityOauthProfile = null;
      }
      try {
        const result = await discoverAntigravityModelOptions({
          accessToken: antigravityTokenInfo.accessToken,
        });
        this.antigravityOpenAiModelOptions = toAntigravityOpenAiModelOptions(result.models);
      } catch {
        this.antigravityOpenAiModelOptions = [];
      }
    }

    configureBuiltinTools({ exaApiKey: this.exaApiKey });
    await loadCustomTools();
    await this.syncModelCatalogs();
    this.ensureModelAndThinkingSelections();
    await this.persistState();
  }

  private async persistState(): Promise<void> {
    savePersistedState({
      authProviders: this.enabledProviders,
      selectedModel: this.selectedModel,
      selectedThinking: this.selectedThinking,
      selectedOpenRouterProvider: this.selectedOpenRouterProvider,
      onboardingCompleted: this.onboardingCompleted,
      inputHistory: this.inputHistory,
    });

    await persistRuntimeSecrets({
      openRouterApiKey: this.openRouterApiKey,
      exaApiKey: this.exaApiKey,
    });
  }

  private ensureModelAndThinkingSelections(): void {
    const selectableProviders = getSelectableModelProviders(this.enabledProviders, this.hasAntigravityToken);
    this.selectedModel = resolveModelForEnabledProviders(
      selectableProviders,
      this.selectedModel,
      this.modelOptionsByProvider,
    );

    const provider =
      findProviderForModel(
        this.selectedModel,
        getModelOptionsForProviders(selectableProviders, this.modelOptionsByProvider),
      ) ?? null;

    this.selectedThinking = normalizeThinkingForModel(
      this.selectedModel,
      provider,
      this.selectedThinking,
      this.modelOptionsByProvider,
    );
  }

  private get hasOpenAiToken(): boolean {
    return this.openAiAccessToken.trim().length > 0;
  }

  private get hasOpenRouterKey(): boolean {
    return this.openRouterApiKey.trim().length > 0;
  }

  private get hasAntigravityToken(): boolean {
    return (this.antigravityOauthTokenInfo?.accessToken ?? "").trim().length > 0;
  }

  private appendSessionMessage(session: RuntimeSession, message: RuntimeUiMessage): void {
    session.messages.push(message);
    session.updatedAt = new Date().toISOString();
    this.emit("session.message.appended", {
      session_id: session.id,
      message,
    });
  }

  private appendSystemMessage(session: RuntimeSession, text: string): void {
    this.appendSessionMessage(session, {
      id: this.nextUiMessageId(),
      kind: "system",
      text,
    });
  }

  private nextUiMessageId(): number {
    const id = this.nextMessageId;
    this.nextMessageId += 1;
    return id;
  }

  getState(): RuntimeSnapshot {
    return {
      auth: {
        enabledProviders: [...this.enabledProviders],
        hasOpenAiToken: this.hasOpenAiToken,
        hasOpenRouterKey: this.hasOpenRouterKey,
        hasAntigravityToken: this.hasAntigravityToken,
        antigravityProfile: this.antigravityOauthProfile,
      },
      onboarding: {
        completed: this.onboardingCompleted,
      },
      model: {
        selectedModel: this.selectedModel,
        selectedThinking: this.selectedThinking,
        selectedOpenRouterProvider: this.selectedOpenRouterProvider,
        selectedProvider: this.selectedModelProvider,
      },
      sessions: {
        count: this.sessions.size,
        ids: [...this.sessions.keys()],
      },
      skills: {
        count: this.availableSkills.length,
        directories: [...this.skillsDirectories],
      },
    };
  }

  private get selectedModelProvider(): AuthProvider | null {
    const selectableProviders = getSelectableModelProviders(this.enabledProviders, this.hasAntigravityToken);
    const options = getModelOptionsForProviders(selectableProviders, this.modelOptionsByProvider);
    return findProviderForModel(this.selectedModel, options) ?? null;
  }

  createSession(params?: { title?: string }): { session_id: string; state: RuntimeSessionState } {
    const id = randomUUID();
    const now = new Date().toISOString();
    const session: RuntimeSession = {
      id,
      createdAt: now,
      updatedAt: now,
      pending: false,
      statusLabel: "ready",
      messages: [],
      history: [],
      activeSession: null,
      conversationProvider: null,
      queuedPrompts: [],
      steeringQueue: [],
      activeAbortController: null,
    };

    if (params?.title?.trim()) {
      this.appendSystemMessage(session, `session created: ${params.title.trim()}`);
    }

    this.sessions.set(id, session);
    this.emit("state.changed", {
      reason: "session_created",
      session_id: id,
      snapshot: this.getState(),
    });

    return {
      session_id: id,
      state: this.toSessionState(session),
    };
  }

  getSession(sessionId: string): RuntimeSessionState {
    const session = this.requireSession(sessionId);
    return this.toSessionState(session);
  }

  private toSessionState(session: RuntimeSession): RuntimeSessionState {
    return {
      id: session.id,
      createdAt: session.createdAt,
      updatedAt: session.updatedAt,
      pending: session.pending,
      statusLabel: session.statusLabel,
      messages: [...session.messages],
      history: [...session.history],
      queue: [...session.queuedPrompts],
      pendingSteerCount: session.steeringQueue.length,
      conversationProvider: session.conversationProvider,
      activeSessionId: session.activeSession?.id ?? null,
    };
  }

  queueList(sessionId: string): { session_id: string; queue: RuntimeTurnQueueItem[] } {
    const session = this.requireSession(sessionId);
    return {
      session_id: session.id,
      queue: [...session.queuedPrompts],
    };
  }

  queueClear(sessionId: string): { session_id: string; cleared_count: number } {
    const session = this.requireSession(sessionId);
    const cleared = session.queuedPrompts.length;
    session.queuedPrompts = [];
    session.updatedAt = new Date().toISOString();
    return {
      session_id: session.id,
      cleared_count: cleared,
    };
  }

  interruptSession(sessionId: string): { session_id: string; interrupted: boolean } {
    const session = this.requireSession(sessionId);
    const controller = session.activeAbortController;
    if (!controller || controller.signal.aborted) {
      return {
        session_id: session.id,
        interrupted: false,
      };
    }

    controller.abort();
    session.statusLabel = "interrupting...";
    this.emit("session.status", {
      session_id: session.id,
      pending: session.pending,
      status_label: session.statusLabel,
    });

    return {
      session_id: session.id,
      interrupted: true,
    };
  }

  steerSession(sessionId: string, text: string): { session_id: string; accepted: boolean } {
    const session = this.requireSession(sessionId);
    const trimmed = text.trim();
    if (!trimmed) {
      return {
        session_id: session.id,
        accepted: false,
      };
    }
    if (!session.pending || !session.activeAbortController) {
      return {
        session_id: session.id,
        accepted: false,
      };
    }

    session.steeringQueue.push({ role: "user", text: trimmed });
    return {
      session_id: session.id,
      accepted: true,
    };
  }

  async sendSessionPrompt(params: {
    session_id: string;
    text?: string;
    images?: unknown;
    enqueue?: boolean;
  }): Promise<{ session_id: string; turn_id: string; accepted: boolean; queued: boolean }> {
    const session = this.requireSession(params.session_id);
    const text = (params.text ?? "").trim();
    const imagesInput = normalizeRuntimeImageInputs(params.images);

    if (!text && imagesInput.length === 0) {
      throw new Error("session.send requires non-empty text or at least one image");
    }

    const turnId = randomUUID();
    if (session.pending) {
      if (params.enqueue !== true) {
        throw new Error("session is busy; use enqueue=true to queue the prompt");
      }
      const queueItem: RuntimeTurnQueueItem = {
        id: turnId,
        text,
        images: imagesInput,
        enqueuedAt: new Date().toISOString(),
      };
      session.queuedPrompts.push(queueItem);
      session.updatedAt = new Date().toISOString();
      this.emit("session.status", {
        session_id: session.id,
        pending: true,
        status_label: `queued (${session.queuedPrompts.length})`,
      });
      return {
        session_id: session.id,
        turn_id: turnId,
        accepted: true,
        queued: true,
      };
    }

    void this.runSessionTurn({
      session,
      turnId,
      text,
      imagesInput,
    });

    return {
      session_id: session.id,
      turn_id: turnId,
      accepted: true,
      queued: false,
    };
  }

  private async runSessionTurn(input: {
    session: RuntimeSession;
    turnId: string;
    text: string;
    imagesInput: RuntimeImageInput[];
  }): Promise<void> {
    const { session, turnId } = input;

    const loadedImagesResult = loadRuntimeImageAttachments(input.imagesInput);
    if (!loadedImagesResult.ok) {
      this.appendSystemMessage(session, `inference error: ${loadedImagesResult.error}`);
      this.emit("session.error", {
        session_id: session.id,
        turn_id: turnId,
        message: loadedImagesResult.error,
      });
      return;
    }

    const promptImages = loadedImagesResult.images;
    const cleanPrompt = input.text.trim();
    const promptWithImageRefs = appendMissingImagePlaceholders(cleanPrompt, promptImages.length);

    const provider = this.selectedModelProvider;
    if (!provider) {
      const message = "select a model from an enabled provider first";
      this.appendSystemMessage(session, message);
      this.emit("session.error", {
        session_id: session.id,
        turn_id: turnId,
        message,
      });
      return;
    }

    const selectedOption = findModelOption(provider, this.selectedModel, this.modelOptionsByProvider);
    const isAntigravityModel =
      (selectedOption?.displayProvider ?? "").trim().toLowerCase() === "antigravity";

    const hasProviderAccess =
      this.enabledProviders.includes(provider) || (isAntigravityModel && provider === "openai");
    if (!hasProviderAccess) {
      const message = `provider ${provider} is not enabled`;
      this.appendSystemMessage(session, message);
      this.emit("session.error", {
        session_id: session.id,
        turn_id: turnId,
        message,
      });
      return;
    }

    if (isAntigravityModel && !this.hasAntigravityToken) {
      const message = "antigravity model selected, but no antigravity oauth token is available";
      this.appendSystemMessage(session, message);
      this.emit("session.error", {
        session_id: session.id,
        turn_id: turnId,
        message,
      });
      return;
    }

    if (!isAntigravityModel && provider === "openai" && !this.hasOpenAiToken) {
      const message = "openai model selected, but no chatgpt oauth token is available";
      this.appendSystemMessage(session, message);
      this.emit("session.error", {
        session_id: session.id,
        turn_id: turnId,
        message,
      });
      return;
    }

    if (provider === "openrouter" && !this.hasOpenRouterKey) {
      const message = "openrouter model selected, but no openrouter api key is available";
      this.appendSystemMessage(session, message);
      this.emit("session.error", {
        session_id: session.id,
        turn_id: turnId,
        message,
      });
      return;
    }

    const skillCatalog = this.refreshSkillsCatalog();
    const skillPromptContext = buildSkillPromptContext(promptWithImageRefs, skillCatalog.skills);
    if (skillPromptContext.selection.combined.length > 0) {
      this.appendSystemMessage(
        session,
        `skills applied: ${skillPromptContext.selection.combined.map((skill) => skill.name).join(", ")}`,
      );
    }

    if (cleanPrompt) {
      this.inputHistory = [...this.inputHistory, cleanPrompt].slice(-MAX_INPUT_HISTORY);
    }
    await this.persistState();

    session.pending = true;
    session.statusLabel = this.selectedThinking === "OFF" ? "drafting response..." : "thinking...";
    session.steeringQueue = [];
    session.activeAbortController = new AbortController();
    this.emit("session.status", {
      session_id: session.id,
      pending: true,
      status_label: session.statusLabel,
      turn_id: turnId,
    });

    const providerSwitchRequiresReset =
      session.conversationProvider !== null && session.conversationProvider !== provider && session.history.length > 0;
    const historyBase = providerSwitchRequiresReset ? [] : session.history;
    if (providerSwitchRequiresReset) {
      resetConversationForProviderSwitch({
        fromProvider: session.conversationProvider,
        toProvider: provider,
        session,
        nextMessageId: () => this.nextUiMessageId(),
      });
      session.activeSession = null;
    }

    const normalizedOpenRouterProvider =
      !isAntigravityModel && provider === "openrouter"
        ? normalizeOpenRouterProviderSelection(this.selectedOpenRouterProvider)
        : undefined;

    const needsNewSession =
      historyBase.length === 0 || !session.activeSession || session.activeSession.provider !== provider;
    let sessionForTurn = session.activeSession;
    if (needsNewSession) {
      try {
        sessionForTurn = createChatSession({
          provider,
          model: this.selectedModel,
          thinkingLevel: this.selectedThinking,
          openRouterProvider: normalizedOpenRouterProvider,
          titleHint: cleanPrompt || (promptImages.length > 0 ? `image: ${path.basename(promptImages[0]!.path)}` : undefined),
        });
        session.activeSession = sessionForTurn;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        this.appendSystemMessage(session, `history save disabled for this turn: ${message}`);
        sessionForTurn = null;
        session.activeSession = null;
      }
    }

    const nextHistory: ChatMessage[] = [
      ...historyBase,
      {
        role: "user",
        text: promptWithImageRefs,
        images: promptImages.length > 0 ? promptImages : undefined,
      },
    ];

    const modelHistory: ChatMessage[] = [
      ...mapMessagesForModel(historyBase, skillCatalog.skills),
      {
        role: "user",
        text: skillPromptContext.modelPrompt,
        images: promptImages.length > 0 ? promptImages : undefined,
      },
    ];

    const userMessage: RuntimeUiMessage = {
      id: this.nextUiMessageId(),
      kind: "user",
      text: promptWithImageRefs,
      images: promptImages.length > 0 ? promptImages : undefined,
    };
    this.appendSessionMessage(session, userMessage);

    let assistantDraftText = "";
    const appliedSteeringMessages: ChatMessage[] = [];

    const appendAssistantDelta = (deltaText: string) => {
      if (!deltaText) {
        return;
      }
      assistantDraftText += deltaText;
      this.emit("session.stream.chunk", {
        session_id: session.id,
        turn_id: turnId,
        chunk: {
          thoughts: [],
          answerText: deltaText,
          segments: [{ kind: "answer", text: deltaText }],
        } satisfies StreamChunk,
      });
    };

    const handleChunk = (chunk: StreamChunk) => {
      const segments = Array.isArray(chunk.segments) ? chunk.segments : [];
      if (segments.length > 0) {
        for (const segment of segments) {
          if (segment.kind === "thought") {
            const title = parseThoughtTitle(segment.text);
            session.statusLabel = `thinking: ${title}`;
            this.emit("session.status", {
              session_id: session.id,
              pending: true,
              status_label: session.statusLabel,
            });
          }
          if (segment.kind === "answer" && segment.text) {
            appendAssistantDelta(segment.text);
            session.statusLabel = "drafting response...";
            this.emit("session.status", {
              session_id: session.id,
              pending: true,
              status_label: session.statusLabel,
            });
          }
        }
      } else {
        if (chunk.answerText) {
          appendAssistantDelta(chunk.answerText);
          session.statusLabel = "drafting response...";
          this.emit("session.status", {
            session_id: session.id,
            pending: true,
            status_label: session.statusLabel,
          });
        }
      }

      this.emit("session.stream.chunk", {
        session_id: session.id,
        turn_id: turnId,
        chunk,
      });
    };

    const handleDebug = (event: DebugEvent) => {
      if (event.stage === "tool_call_started") {
        this.emit("session.tool.call.started", {
          session_id: session.id,
          turn_id: turnId,
          data: event.data as Record<string, unknown>,
        });
      }
      if (event.stage === "tool_call_completed") {
        this.emit("session.tool.call.completed", {
          session_id: session.id,
          turn_id: turnId,
          data: event.data as Record<string, unknown>,
        });
      }
      if (event.stage === "tool_results") {
        this.emit("session.tool.results", {
          session_id: session.id,
          turn_id: turnId,
          data: event.data as Record<string, unknown>,
        });
      }
      if (this.superDebug) {
        this.emit("session.debug", {
          session_id: session.id,
          turn_id: turnId,
          stage: event.stage,
          data: event.data as Record<string, unknown>,
        });
      }
    };

    const drainSteeringMessages = (): ChatMessage[] => {
      const queued = session.steeringQueue;
      if (queued.length === 0) {
        return [];
      }
      session.steeringQueue = [];
      appliedSteeringMessages.push(...queued);
      for (const steerMessage of queued) {
        this.appendSessionMessage(session, {
          id: this.nextUiMessageId(),
          kind: "user",
          text: steerMessage.text,
        });
      }
      session.statusLabel = "steer applied; continuing...";
      this.emit("session.status", {
        session_id: session.id,
        pending: true,
        status_label: session.statusLabel,
      });
      return queued;
    };

    try {
      const runtimeSystemInstruction = buildRuntimeSystemInstruction({
        baseInstruction: loafConfig.systemInstruction,
        hasExaSearch: Boolean(this.exaApiKey.trim()),
        skillInstructionBlock: skillPromptContext.instructionBlock,
      });

      const result =
        isAntigravityModel
          ? await runAntigravityInferenceStream(
              {
                accessToken: this.antigravityOauthTokenInfo?.accessToken ?? "",
                model: this.selectedModel,
                messages: modelHistory,
                thinkingLevel: this.selectedThinking,
                includeThoughts: this.selectedThinking !== "OFF",
                systemInstruction: runtimeSystemInstruction,
                signal: session.activeAbortController.signal,
                drainSteeringMessages,
              },
              handleChunk,
              handleDebug,
            )
          : provider === "openrouter"
            ? await runOpenRouterInferenceStream(
                {
                  apiKey: this.openRouterApiKey,
                  model: this.selectedModel,
                  messages: modelHistory,
                  thinkingLevel: this.selectedThinking,
                  includeThoughts: this.selectedThinking !== "OFF",
                  forcedProvider:
                    normalizeOpenRouterProviderSelection(this.selectedOpenRouterProvider) === OPENROUTER_PROVIDER_ANY_ID
                      ? null
                      : normalizeOpenRouterProviderSelection(this.selectedOpenRouterProvider),
                  systemInstruction: runtimeSystemInstruction,
                  signal: session.activeAbortController.signal,
                  drainSteeringMessages,
                },
                handleChunk,
                handleDebug,
              )
            : await runOpenAiInferenceStream(
                {
                  accessToken: this.openAiAccessToken,
                  chatgptAccountId: this.openAiAccountId,
                  model: this.selectedModel,
                  messages: modelHistory,
                  thinkingLevel: this.selectedThinking,
                  includeThoughts: this.selectedThinking !== "OFF",
                  systemInstruction: runtimeSystemInstruction,
                  signal: session.activeAbortController.signal,
                  drainSteeringMessages,
                },
                handleChunk,
                handleDebug,
              );

      const finalAssistantText = assistantDraftText || result.answer;
      const assistantMessage: ChatMessage = {
        role: "assistant",
        text: finalAssistantText,
      };

      this.appendSessionMessage(session, {
        id: this.nextUiMessageId(),
        kind: "assistant",
        text: finalAssistantText,
      });

      const savedHistory = [...nextHistory, ...appliedSteeringMessages, assistantMessage];
      session.history = savedHistory;
      session.conversationProvider = provider;

      if (sessionForTurn) {
        try {
          session.activeSession = writeChatSession({
            session: sessionForTurn,
            messages: savedHistory,
            provider,
            model: this.selectedModel,
            thinkingLevel: this.selectedThinking,
            openRouterProvider: normalizedOpenRouterProvider,
            titleHint: cleanPrompt,
          });
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          this.appendSystemMessage(session, `chat history save failed: ${message}`);
        }
      }

      this.emit("session.completed", {
        session_id: session.id,
        turn_id: turnId,
        answer_length: finalAssistantText.length,
      });
    } catch (error) {
      const isAbort = error instanceof Error && error.name === "AbortError";
      if (isAbort) {
        const interruptedAssistant = assistantDraftText.trim();
        const interruptedHistoryBase = [...nextHistory, ...appliedSteeringMessages];
        const interruptedHistory = interruptedAssistant
          ? [...interruptedHistoryBase, { role: "assistant" as const, text: interruptedAssistant }]
          : interruptedHistoryBase;
        session.history = interruptedHistory;
        session.conversationProvider = provider;

        if (sessionForTurn) {
          try {
            session.activeSession = writeChatSession({
              session: sessionForTurn,
              messages: interruptedHistory,
              provider,
              model: this.selectedModel,
              thinkingLevel: this.selectedThinking,
              openRouterProvider: normalizedOpenRouterProvider,
              titleHint: cleanPrompt,
            });
          } catch {
            // ignore write failure on aborted turn
          }
        }

        if (interruptedAssistant) {
          this.appendSessionMessage(session, {
            id: this.nextUiMessageId(),
            kind: "assistant",
            text: interruptedAssistant,
          });
        }

        this.appendSystemMessage(
          session,
          interruptedAssistant ? "response interrupted by user. partial output kept." : "response interrupted by user.",
        );

        this.emit("session.interrupted", {
          session_id: session.id,
          turn_id: turnId,
          partial_output: Boolean(interruptedAssistant),
        });
      } else {
        const message = error instanceof Error ? error.message : String(error);
        this.appendSystemMessage(session, `inference error: ${message}`);
        this.emit("session.error", {
          session_id: session.id,
          turn_id: turnId,
          message,
        });
      }
    } finally {
      const unappliedSteerCount = session.steeringQueue.length;
      session.steeringQueue = [];
      session.activeAbortController = null;
      session.pending = false;
      session.statusLabel = "ready";
      session.updatedAt = new Date().toISOString();
      this.emit("session.status", {
        session_id: session.id,
        pending: false,
        status_label: "ready",
      });

      if (unappliedSteerCount > 0) {
        this.appendSystemMessage(session, `${unappliedSteerCount} steer message(s) were queued too late and not applied.`);
      }

      if (session.queuedPrompts.length > 0 && !this.shuttingDown) {
        const queued = session.queuedPrompts.shift();
        if (queued) {
          void this.runSessionTurn({
            session,
            turnId: queued.id,
            text: queued.text,
            imagesInput: queued.images,
          });
        }
      }
    }
  }

  async executeCommand(params: {
    session_id: string;
    raw_command: string;
  }): Promise<{
    handled: boolean;
    output?: unknown;
  }> {
    const session = this.requireSession(params.session_id);
    const trimmed = params.raw_command.trim();
    if (!trimmed.startsWith("/")) {
      return {
        handled: false,
      };
    }

    const tokens = trimmed.split(/\s+/).filter(Boolean);
    const command = tokens[0]!.toLowerCase();
    const args = tokens.slice(1);

    if (command === "/auth") {
      return {
        handled: true,
        output: {
          auth_status: this.authStatus(),
          hint: "use auth.connect.* and auth.set.* methods for explicit RPC auth flows",
        },
      };
    }

    if (command === "/onboarding") {
      return {
        handled: true,
        output: this.onboardingStatus(),
      };
    }

    if (command === "/forgeteverything") {
      clearPersistedConfig();
      this.enabledProviders = [];
      this.openAiAccessToken = "";
      this.openAiAccountId = null;
      this.antigravityOauthTokenInfo = null;
      this.antigravityOauthProfile = null;
      this.openRouterApiKey = "";
      this.exaApiKey = "";
      this.selectedOpenRouterProvider = OPENROUTER_PROVIDER_ANY_ID;
      this.modelOptionsByProvider = {
        openai: getDefaultModelOptionsForProvider("openai"),
        openrouter: getDefaultModelOptionsForProvider("openrouter"),
      };
      this.selectedModel = "";
      this.selectedThinking = loafConfig.thinkingLevel;
      this.inputHistory = [];
      this.onboardingCompleted = false;
      session.messages = [];
      session.history = [];
      session.activeSession = null;
      session.conversationProvider = null;
      session.queuedPrompts = [];
      await this.persistState();
      this.appendSystemMessage(session, "all local config was cleared. onboarding restarted.");
      this.emit("state.changed", {
        reason: "forgeteverything",
        snapshot: this.getState(),
      });
      return {
        handled: true,
        output: {
          cleared: true,
        },
      };
    }

    if (command === "/model") {
      return {
        handled: true,
        output: {
          selected_model: this.selectedModel,
          selected_provider: this.selectedModelProvider,
          selected_thinking: this.selectedThinking,
          selected_openrouter_provider: this.selectedOpenRouterProvider,
          available: this.modelList(),
        },
      };
    }

    if (command === "/limits") {
      return {
        handled: true,
        output: await this.getLimits(),
      };
    }

    if (command === "/history") {
      return {
        handled: true,
        output: this.runHistoryCommand(session, args),
      };
    }

    if (command === "/clear") {
      session.messages = [];
      session.history = [];
      session.conversationProvider = null;
      session.activeSession = null;
      return {
        handled: true,
        output: {
          cleared: true,
        },
      };
    }

    if (command === "/skills") {
      return {
        handled: true,
        output: this.skillsList(),
      };
    }

    if (command === "/tools") {
      return {
        handled: true,
        output: this.toolsList(),
      };
    }

    if (command === "/help") {
      return {
        handled: true,
        output: {
          commands: COMMAND_OPTIONS,
        },
      };
    }

    if (command === "/quit" || command === "/exit") {
      await this.shutdown("quit_command");
      return {
        handled: true,
        output: {
          shutting_down: true,
        },
      };
    }

    return {
      handled: true,
      output: {
        error: `unknown command: ${command}`,
      },
    };
  }

  authStatus() {
    return {
      enabled_providers: [...this.enabledProviders],
      has_openai_token: this.hasOpenAiToken,
      has_openrouter_key: this.hasOpenRouterKey,
      has_antigravity_token: this.hasAntigravityToken,
      antigravity_profile: this.antigravityOauthProfile,
    };
  }

  async connectOpenAi(params?: {
    mode?: "auto" | "browser" | "device_code";
    originator?: string;
  }) {
    this.emit("auth.flow.started", {
      provider: "openai",
    });

    try {
      const result = await runOpenAiOauthLogin({
        mode: params?.mode,
        originator: params?.originator,
        openBrowser: !this.rpcMode,
        onAuthUrl: (url) => {
          this.emit("auth.flow.url", {
            provider: "openai",
            url,
          });
        },
        onDeviceCode: (info) => {
          this.emit("auth.flow.device_code", {
            provider: "openai",
            verification_url: info.verificationUrl,
            user_code: info.userCode,
            interval_seconds: info.intervalSeconds,
            expires_in_seconds: info.expiresInSeconds,
          });
        },
      });

      this.openAiAccessToken = result.chatgptAuth.accessToken;
      this.openAiAccountId = result.chatgptAuth.accountId;
      this.enabledProviders = dedupeAuthProviders([...this.enabledProviders, "openai"]);
      await this.syncModelCatalogs();
      this.ensureModelAndThinkingSelections();
      await this.persistState();

      const output = {
        provider: "openai",
        login_method: result.loginMethod,
        account_id: result.chatgptAuth.accountId,
      };
      this.emit("auth.flow.completed", output);
      this.emit("state.changed", {
        reason: "auth_openai_connected",
        snapshot: this.getState(),
      });
      return output;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.emit("auth.flow.failed", {
        provider: "openai",
        message,
      });
      throw error;
    }
  }

  async connectAntigravity() {
    this.emit("auth.flow.started", {
      provider: "antigravity",
    });

    try {
      const result = await runAntigravityOauthLogin({
        openBrowser: !this.rpcMode,
        onAuthUrl: (url) => {
          this.emit("auth.flow.url", {
            provider: "antigravity",
            url,
          });
        },
      });

      this.antigravityOauthTokenInfo = result.tokenInfo;
      this.antigravityOauthProfile = result.profile;
      await this.syncAntigravityModelCatalog();
      this.ensureModelAndThinkingSelections();
      await this.persistState();

      const output = {
        provider: "antigravity",
        profile: result.profile,
      };
      this.emit("auth.flow.completed", output);
      this.emit("state.changed", {
        reason: "auth_antigravity_connected",
        snapshot: this.getState(),
      });
      return output;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.emit("auth.flow.failed", {
        provider: "antigravity",
        message,
      });
      throw error;
    }
  }

  async setOpenRouterKey(apiKey: string): Promise<{ configured: boolean; model_count: number; source: string }> {
    const key = apiKey.trim();
    if (!key) {
      throw new Error("openrouter api key cannot be empty");
    }

    this.openRouterApiKey = key;
    this.enabledProviders = dedupeAuthProviders([...this.enabledProviders, "openrouter"]);
    let modelCount = 0;
    let source = "fallback";

    try {
      const result = await discoverOpenRouterModelOptions({ apiKey: key });
      this.modelOptionsByProvider = {
        ...this.modelOptionsByProvider,
        openrouter: result.models,
      };
      modelCount = result.models.length;
      source = result.source;
    } catch {
      // keep prior options; runtime still configured
    }

    await this.persistState();
    this.emit("state.changed", {
      reason: "openrouter_key_updated",
      snapshot: this.getState(),
    });

    return {
      configured: true,
      model_count: modelCount,
      source,
    };
  }

  async setExaKey(apiKeyOrSkip: string): Promise<{ configured: boolean; skipped: boolean }> {
    const value = apiKeyOrSkip.trim();
    if (!value) {
      throw new Error("exa api key cannot be empty. use 'skip' to disable");
    }

    if (value.toLowerCase() === "skip") {
      this.exaApiKey = "";
      configureBuiltinTools({ exaApiKey: "" });
      await this.persistState();
      this.emit("state.changed", {
        reason: "exa_key_skipped",
        snapshot: this.getState(),
      });
      return {
        configured: false,
        skipped: true,
      };
    }

    this.exaApiKey = value;
    configureBuiltinTools({ exaApiKey: value });
    await this.persistState();
    this.emit("state.changed", {
      reason: "exa_key_updated",
      snapshot: this.getState(),
    });
    return {
      configured: true,
      skipped: false,
    };
  }

  onboardingStatus() {
    return {
      completed: this.onboardingCompleted,
    };
  }

  async onboardingComplete() {
    this.onboardingCompleted = true;
    await this.persistState();
    this.emit("state.changed", {
      reason: "onboarding_complete",
      snapshot: this.getState(),
    });
    return {
      completed: true,
    };
  }

  modelList(params?: { provider?: AuthProvider }) {
    const providers = params?.provider
      ? [params.provider]
      : getSelectableModelProviders(this.enabledProviders, this.hasAntigravityToken);

    return {
      providers,
      models: getModelOptionsForProviders(providers, this.modelOptionsByProvider),
      selected_model: this.selectedModel,
      selected_thinking: this.selectedThinking,
      selected_provider: this.selectedModelProvider,
      selected_openrouter_provider: this.selectedOpenRouterProvider,
    };
  }

  async modelSelect(params: {
    model_id: string;
    provider: AuthProvider;
    thinking_level: ThinkingLevel;
    openrouter_provider?: string;
  }) {
    const modelId = params.model_id.trim();
    if (!modelId) {
      throw new Error("model_id cannot be empty");
    }

    const option = findModelOption(params.provider, modelId, this.modelOptionsByProvider);
    if (!option) {
      throw new Error(`unknown model for provider ${params.provider}: ${modelId}`);
    }

    this.selectedModel = modelId;
    this.selectedThinking = normalizeThinkingForModel(
      modelId,
      params.provider,
      params.thinking_level,
      this.modelOptionsByProvider,
    );

    if (params.provider === "openrouter") {
      this.selectedOpenRouterProvider =
        normalizeOpenRouterProviderSelection(params.openrouter_provider) || OPENROUTER_PROVIDER_ANY_ID;
    }

    await this.persistState();
    this.emit("state.changed", {
      reason: "model_selected",
      snapshot: this.getState(),
    });

    return {
      selected_model: this.selectedModel,
      selected_provider: params.provider,
      selected_thinking: this.selectedThinking,
      selected_openrouter_provider: this.selectedOpenRouterProvider,
    };
  }

  async getLimits() {
    const openAiToken = this.openAiAccessToken.trim();
    const antigravityToken = this.antigravityOauthTokenInfo?.accessToken.trim() ?? "";

    if (!openAiToken && !antigravityToken) {
      throw new Error("limits requires at least one oauth provider");
    }

    const [openAiResult, antigravityResult] = await Promise.all([
      openAiToken
        ? fetchOpenAiUsageSnapshot(openAiToken, this.openAiAccountId)
            .then((snapshot) => ({ ok: true as const, snapshot }))
            .catch((error) => ({ ok: false as const, message: error instanceof Error ? error.message : String(error) }))
        : Promise.resolve(null),
      antigravityToken
        ? fetchAntigravityUsageSnapshot({ accessToken: antigravityToken })
            .then((snapshot) => ({ ok: true as const, snapshot }))
            .catch((error) => ({ ok: false as const, message: error instanceof Error ? error.message : String(error) }))
        : Promise.resolve(null),
    ]);

    return {
      openai: openAiResult,
      antigravity: antigravityResult,
    };
  }

  historyList(params?: { limit?: number; cursor?: number }) {
    const limit = params?.limit && Number.isFinite(params.limit) ? Math.max(1, Math.floor(params.limit)) : 100;
    const cursor = params?.cursor && Number.isFinite(params.cursor) ? Math.max(0, Math.floor(params.cursor)) : 0;
    const sessions = listChatSessions({ limit: Math.max(limit + cursor, limit) });
    const rows = sessions.slice(cursor, cursor + limit);

    return {
      total: sessions.length,
      cursor,
      limit,
      sessions: rows,
      next_cursor: cursor + rows.length < sessions.length ? cursor + rows.length : null,
    };
  }

  historyGet(params: { id?: string; last?: boolean; rollout_path?: string }): ChatSessionRecord {
    if (params.last === true) {
      const latest = loadLatestChatSession();
      if (!latest) {
        throw new Error("no saved chat history yet");
      }
      return latest;
    }

    if (params.rollout_path?.trim()) {
      const session = loadChatSession(params.rollout_path.trim());
      if (!session) {
        throw new Error(`failed to load chat: ${params.rollout_path.trim()}`);
      }
      return session;
    }

    if (params.id?.trim()) {
      const session = loadChatSessionById(params.id.trim());
      if (!session) {
        throw new Error(`no saved chat matched id: ${params.id.trim()}`);
      }
      return session;
    }

    throw new Error("history.get requires one of: id, last, rollout_path");
  }

  historyClearSession(sessionId: string): { session_id: string; cleared: true } {
    const session = this.requireSession(sessionId);
    session.history = [];
    session.messages = [];
    session.activeSession = null;
    session.conversationProvider = null;
    session.queuedPrompts = [];
    session.steeringQueue = [];
    session.pending = false;
    session.statusLabel = "ready";
    session.updatedAt = new Date().toISOString();
    return {
      session_id: session.id,
      cleared: true,
    };
  }

  skillsList() {
    const catalog = this.refreshSkillsCatalog();
    return {
      directories: catalog.directories,
      errors: catalog.errors,
      skills: catalog.skills.map((skill) => ({
        name: skill.name,
        description: skill.description,
        description_preview: skill.descriptionPreview,
        source_path: skill.sourcePath,
      })),
    };
  }

  toolsList() {
    return {
      tools: defaultToolRegistry.list().map((tool) => ({
        name: tool.name,
        description: tool.description,
        input_schema: tool.inputSchema,
      })),
    };
  }

  setDebug(superDebug: boolean) {
    this.superDebug = superDebug;
    return {
      super_debug: this.superDebug,
    };
  }

  async listOpenRouterProvidersForModel(modelId: string): Promise<{ model_id: string; providers: string[] }> {
    if (!this.openRouterApiKey.trim()) {
      throw new Error("openrouter api key is not configured");
    }

    const providers = await listOpenRouterProvidersForModel(this.openRouterApiKey, modelId);
    this.modelOptionsByProvider = {
      ...this.modelOptionsByProvider,
      openrouter: this.modelOptionsByProvider.openrouter.map((option) =>
        option.id === modelId
          ? {
              ...option,
              routingProviders: providers,
            }
          : option,
      ),
    };

    return {
      model_id: modelId,
      providers,
    };
  }

  private requireSession(sessionId: string): RuntimeSession {
    const normalized = sessionId.trim();
    if (!normalized) {
      throw new Error("session_id is required");
    }

    const session = this.sessions.get(normalized);
    if (!session) {
      throw new Error(`unknown session: ${normalized}`);
    }
    return session;
  }

  private refreshSkillsCatalog() {
    const catalog = loadSkillsCatalog();
    this.availableSkills = catalog.skills;
    this.skillsDirectories = catalog.directories;
    return catalog;
  }

  private async syncModelCatalogs(): Promise<void> {
    await this.syncAntigravityModelCatalog();

    const next: Record<AuthProvider, ModelOption[]> = {
      openai: [...this.modelOptionsByProvider.openai],
      openrouter: [...this.modelOptionsByProvider.openrouter],
    };

    if (this.hasOpenAiToken) {
      try {
        const result = await discoverOpenAiModelOptions({
          accessToken: this.openAiAccessToken,
          chatgptAccountId: this.openAiAccountId,
        });
        next.openai = mergeOpenAiAndAntigravityModels(result.models, this.antigravityOpenAiModelOptions);
      } catch {
        next.openai = mergeOpenAiAndAntigravityModels(next.openai, this.antigravityOpenAiModelOptions);
      }
    } else {
      next.openai = mergeOpenAiAndAntigravityModels(next.openai, this.antigravityOpenAiModelOptions);
    }

    if (this.hasOpenRouterKey) {
      try {
        const result = await discoverOpenRouterModelOptions({
          apiKey: this.openRouterApiKey,
        });
        next.openrouter = result.models;
      } catch {
        // keep previous options
      }
    }

    this.modelOptionsByProvider = next;
  }

  private async syncAntigravityModelCatalog(): Promise<void> {
    const accessToken = this.antigravityOauthTokenInfo?.accessToken.trim() ?? "";
    if (!accessToken) {
      this.antigravityOpenAiModelOptions = [];
      return;
    }

    try {
      const result = await discoverAntigravityModelOptions({
        accessToken,
      });
      this.antigravityOpenAiModelOptions = toAntigravityOpenAiModelOptions(result.models);
      this.modelOptionsByProvider = {
        ...this.modelOptionsByProvider,
        openai: mergeOpenAiAndAntigravityModels(this.modelOptionsByProvider.openai, this.antigravityOpenAiModelOptions),
      };
    } catch {
      this.antigravityOpenAiModelOptions = [];
    }
  }

  private runHistoryCommand(session: RuntimeSession, args: string[]) {
    if (args.length === 0 || args[0] === "list" || args[0] === "all") {
      return this.historyList({ limit: 100 });
    }

    const first = args[0]!.toLowerCase();
    if (first === "last") {
      const latest = loadLatestChatSession();
      if (!latest) {
        throw new Error("no saved chat history yet");
      }
      this.resumeSession(session, latest);
      return {
        resumed: latest.id,
      };
    }

    const byId = loadChatSessionById(args[0]!);
    if (!byId) {
      throw new Error(`no saved chat matched id: ${args[0]}`);
    }

    this.resumeSession(session, byId);
    return {
      resumed: byId.id,
    };
  }

  private resumeSession(session: RuntimeSession, chat: ChatSessionRecord): void {
    if (!this.enabledProviders.includes(chat.provider)) {
      this.enabledProviders = dedupeAuthProviders([...this.enabledProviders, chat.provider]);
    }

    this.selectedModel = chat.model;
    this.selectedThinking = normalizeThinkingForModel(
      chat.model,
      chat.provider,
      chat.thinkingLevel,
      this.modelOptionsByProvider,
    );
    if (chat.provider === "openrouter") {
      this.selectedOpenRouterProvider =
        normalizeOpenRouterProviderSelection(chat.openRouterProvider) || OPENROUTER_PROVIDER_ANY_ID;
    }

    session.history = chat.messages;
    session.messages = chatHistoryToUiMessages(chat.messages, () => this.nextUiMessageId());
    session.conversationProvider = chat.provider;
    session.activeSession = {
      id: chat.id,
      title: chat.title,
      provider: chat.provider,
      model: chat.model,
      thinkingLevel: chat.thinkingLevel,
      openRouterProvider: chat.openRouterProvider,
      createdAt: chat.createdAt,
      updatedAt: chat.updatedAt,
      messageCount: chat.messageCount,
      rolloutPath: chat.rolloutPath,
    };
    session.updatedAt = new Date().toISOString();
    this.emit("state.changed", {
      reason: "session_resumed",
      snapshot: this.getState(),
    });
  }
}

const COMMAND_OPTIONS: Array<{ name: string; description: string }> = [
  { name: "/auth", description: "add auth provider" },
  { name: "/onboarding", description: "open setup flow (auth + exa key)" },
  { name: "/forgeteverything", description: "wipe local config and restart onboarding" },
  { name: "/model", description: "choose model and thinking level" },
  { name: "/limits", description: "show oauth usage limits" },
  { name: "/history", description: "resume a saved chat (/history, /history last, /history <id>)" },
  { name: "/skills", description: "list available skills" },
  { name: "/tools", description: "list registered tools" },
  { name: "/clear", description: "clear conversation messages" },
  { name: "/quit", description: "exit loaf" },
  { name: "/help", description: "show available commands" },
  { name: "/exit", description: "exit loaf" },
];

function chatHistoryToUiMessages(history: ChatMessage[], nextMessageId: () => number): RuntimeUiMessage[] {
  return history.map((message) => ({
    id: nextMessageId(),
    kind: message.role,
    text: message.text,
    images: message.images,
  }));
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

function getThinkingOptionsForModel(
  modelId: string,
  provider: AuthProvider,
  modelOptionsByProvider: Record<AuthProvider, ModelOption[]>,
): ThinkingLevel[] {
  const modelOption = findModelOption(provider, modelId, modelOptionsByProvider);
  if (modelOption?.supportedThinkingLevels && modelOption.supportedThinkingLevels.length > 0) {
    return [...new Set(modelOption.supportedThinkingLevels)];
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
  const supportedLevels = thinkingOptions;
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

function resetConversationForProviderSwitch(params: {
  fromProvider: AuthProvider | null;
  toProvider: AuthProvider;
  session: RuntimeSession;
  nextMessageId: () => number;
}): void {
  params.session.history = [];
  const fromLabel = params.fromProvider ?? "unknown";
  params.session.messages = [
    {
      id: params.nextMessageId(),
      kind: "system",
      text: `provider switched: ${fromLabel} -> ${params.toProvider}. conversation context was reset.`,
    },
  ];
}
