import type { LoafCoreRuntime } from "../core/runtime.js";
import type { ThinkingLevel } from "../config.js";
import {
  JSON_RPC_ERROR,
  assertBoolean,
  assertObjectParams,
  assertOptionalBoolean,
  assertOptionalNumber,
  assertOptionalString,
  assertString,
  buildRpcMethodError,
  type JsonRpcRequest,
} from "./protocol.js";

type RpcMethodHandler = (params: unknown) => Promise<unknown>;

const PROTOCOL_VERSION = "1.0.0";

export class RpcRouter {
  private readonly runtime: LoafCoreRuntime;
  private readonly handlers = new Map<string, RpcMethodHandler>();

  constructor(runtime: LoafCoreRuntime) {
    this.runtime = runtime;
    this.registerHandlers();
  }

  listMethods(): string[] {
    return [...this.handlers.keys()].sort((a, b) => a.localeCompare(b));
  }

  async dispatch(request: JsonRpcRequest): Promise<unknown> {
    const handler = this.handlers.get(request.method);
    if (!handler) {
      throw buildRpcMethodError(
        JSON_RPC_ERROR.METHOD_NOT_FOUND,
        `method not found: ${request.method}`,
        {
          reason: "method_not_found",
          method: request.method,
        },
      );
    }

    return handler(request.params);
  }

  private registerHandlers(): void {
    this.handlers.set("rpc.handshake", async (params) => {
      const body = assertObjectParams(params ?? {}, "rpc.handshake");
      const protocolVersion = assertOptionalString(body.protocol_version, "protocol_version", "rpc.handshake");
      const strict = assertOptionalBoolean(body.strict, "strict", "rpc.handshake") ?? false;

      if (strict && protocolVersion && protocolVersion !== PROTOCOL_VERSION) {
        throw buildRpcMethodError(
          JSON_RPC_ERROR.INVALID_PARAMS,
          `unsupported protocol_version: ${protocolVersion}`,
          {
            reason: "unsupported_protocol_version",
            supported: PROTOCOL_VERSION,
          },
        );
      }

      return {
        protocol_version: PROTOCOL_VERSION,
        server_name: "loaf",
        capabilities: {
          events: true,
          command_execute: true,
          multi_session: true,
          image_inputs: ["path", "data_url"],
        },
        methods: this.listMethods(),
      };
    });

    this.handlers.set("system.ping", async () => ({
      ok: true,
      time: new Date().toISOString(),
    }));

    this.handlers.set("system.shutdown", async (params) => {
      const body = assertObjectParams(params ?? {}, "system.shutdown");
      const reason = assertOptionalString(body.reason, "reason", "system.shutdown");
      return this.runtime.shutdown(reason);
    });

    this.handlers.set("state.get", async () => this.runtime.getState());

    this.handlers.set("session.create", async (params) => {
      const body = assertObjectParams(params ?? {}, "session.create");
      const title = assertOptionalString(body.title, "title", "session.create");
      return this.runtime.createSession({ title });
    });

    this.handlers.set("session.get", async (params) => {
      const body = assertObjectParams(params, "session.get");
      const sessionId = assertString(body.session_id, "session_id", "session.get");
      return this.runtime.getSession(sessionId);
    });

    this.handlers.set("session.send", async (params) => {
      const body = assertObjectParams(params, "session.send");
      const sessionId = assertString(body.session_id, "session_id", "session.send");
      const text = assertOptionalString(body.text, "text", "session.send");
      const enqueue = assertOptionalBoolean(body.enqueue, "enqueue", "session.send") ?? false;
      return this.runtime.sendSessionPrompt({
        session_id: sessionId,
        text,
        images: body.images,
        enqueue,
      });
    });

    this.handlers.set("session.steer", async (params) => {
      const body = assertObjectParams(params, "session.steer");
      const sessionId = assertString(body.session_id, "session_id", "session.steer");
      const text = assertString(body.text, "text", "session.steer");
      return this.runtime.steerSession(sessionId, text);
    });

    this.handlers.set("session.interrupt", async (params) => {
      const body = assertObjectParams(params, "session.interrupt");
      const sessionId = assertString(body.session_id, "session_id", "session.interrupt");
      return this.runtime.interruptSession(sessionId);
    });

    this.handlers.set("session.queue.list", async (params) => {
      const body = assertObjectParams(params, "session.queue.list");
      const sessionId = assertString(body.session_id, "session_id", "session.queue.list");
      return this.runtime.queueList(sessionId);
    });

    this.handlers.set("session.queue.clear", async (params) => {
      const body = assertObjectParams(params, "session.queue.clear");
      const sessionId = assertString(body.session_id, "session_id", "session.queue.clear");
      return this.runtime.queueClear(sessionId);
    });

    this.handlers.set("command.execute", async (params) => {
      const body = assertObjectParams(params, "command.execute");
      const sessionId = assertString(body.session_id, "session_id", "command.execute");
      const rawCommand = assertString(body.raw_command, "raw_command", "command.execute");
      return this.runtime.executeCommand({
        session_id: sessionId,
        raw_command: rawCommand,
      });
    });

    this.handlers.set("auth.status", async () => this.runtime.authStatus());

    this.handlers.set("auth.connect.openai", async (params) => {
      const body = assertObjectParams(params ?? {}, "auth.connect.openai");
      const mode = assertOptionalString(body.mode, "mode", "auth.connect.openai") as
        | "auto"
        | "browser"
        | "device_code"
        | undefined;
      const originator = assertOptionalString(body.originator, "originator", "auth.connect.openai");
      return this.runtime.connectOpenAi({
        mode,
        originator,
      });
    });

    this.handlers.set("auth.connect.antigravity", async () => this.runtime.connectAntigravity());

    this.handlers.set("auth.set.openrouter_key", async (params) => {
      const body = assertObjectParams(params, "auth.set.openrouter_key");
      const apiKey = assertString(body.api_key, "api_key", "auth.set.openrouter_key");
      return this.runtime.setOpenRouterKey(apiKey);
    });

    this.handlers.set("auth.set.exa_key", async (params) => {
      const body = assertObjectParams(params, "auth.set.exa_key");
      const apiKey = assertString(body.api_key, "api_key", "auth.set.exa_key");
      return this.runtime.setExaKey(apiKey);
    });

    this.handlers.set("onboarding.status", async () => this.runtime.onboardingStatus());
    this.handlers.set("onboarding.complete", async () => this.runtime.onboardingComplete());

    this.handlers.set("model.list", async (params) => {
      const body = assertObjectParams(params ?? {}, "model.list");
      const provider = assertOptionalString(body.provider, "provider", "model.list") as
        | "openai"
        | "openrouter"
        | "antigravity"
        | undefined;
      if (provider && provider !== "openai" && provider !== "openrouter" && provider !== "antigravity") {
        throw buildRpcMethodError(
          JSON_RPC_ERROR.INVALID_PARAMS,
          "provider must be one of: openai, openrouter, antigravity",
          { reason: "invalid_params", field: "provider" },
        );
      }
      return this.runtime.modelList({ provider });
    });

    this.handlers.set("model.select", async (params) => {
      const body = assertObjectParams(params, "model.select");
      const modelId = assertString(body.model_id, "model_id", "model.select");
      const provider = assertString(body.provider, "provider", "model.select") as "openai" | "openrouter" | "antigravity";
      if (provider !== "openai" && provider !== "openrouter" && provider !== "antigravity") {
        throw buildRpcMethodError(
          JSON_RPC_ERROR.INVALID_PARAMS,
          "provider must be openai, openrouter, or antigravity",
          {
          reason: "invalid_params",
          field: "provider",
          },
        );
      }
      const thinkingLevel = assertString(body.thinking_level, "thinking_level", "model.select");
      const openrouterProvider = assertOptionalString(body.openrouter_provider, "openrouter_provider", "model.select");
      const sessionId = assertOptionalString(body.session_id, "session_id", "model.select");
      const compressImmediately =
        assertOptionalBoolean(body.compress_immediately, "compress_immediately", "model.select") ?? false;
      return this.runtime.modelSelect({
        model_id: modelId,
        provider,
        thinking_level: thinkingLevel as ThinkingLevel,
        openrouter_provider: openrouterProvider,
        session_id: sessionId,
        compress_immediately: compressImmediately,
      });
    });

    this.handlers.set("model.openrouter.providers", async (params) => {
      const body = assertObjectParams(params, "model.openrouter.providers");
      const modelId = assertString(body.model_id, "model_id", "model.openrouter.providers");
      return this.runtime.listOpenRouterProvidersForModel(modelId);
    });

    this.handlers.set("limits.get", async () => this.runtime.getLimits());

    this.handlers.set("history.list", async (params) => {
      const body = assertObjectParams(params ?? {}, "history.list");
      const limit = assertOptionalNumber(body.limit, "limit", "history.list");
      const cursor = assertOptionalNumber(body.cursor, "cursor", "history.list");
      return this.runtime.historyList({ limit, cursor });
    });

    this.handlers.set("history.get", async (params) => {
      const body = assertObjectParams(params ?? {}, "history.get");
      const id = assertOptionalString(body.id, "id", "history.get");
      const last = assertOptionalBoolean(body.last, "last", "history.get") ?? false;
      const rolloutPath = assertOptionalString(body.rollout_path, "rollout_path", "history.get");
      return this.runtime.historyGet({
        id,
        last,
        rollout_path: rolloutPath,
      });
    });

    this.handlers.set("history.clear_session", async (params) => {
      const body = assertObjectParams(params, "history.clear_session");
      const sessionId = assertString(body.session_id, "session_id", "history.clear_session");
      return this.runtime.historyClearSession(sessionId);
    });

    this.handlers.set("skills.list", async () => this.runtime.skillsList());
    this.handlers.set("tools.list", async () => this.runtime.toolsList());

    this.handlers.set("debug.set", async (params) => {
      const body = assertObjectParams(params, "debug.set");
      const superDebug = assertBoolean(body.super_debug, "super_debug", "debug.set");
      return this.runtime.setDebug(superDebug);
    });
  }
}
