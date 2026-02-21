import { LoafCoreRuntime, type RuntimeEvent } from "../core/runtime.js";
import { RpcRouter } from "./router.js";
import {
  JSON_RPC_ERROR,
  buildRpcMethodError,
  isRpcMethodError,
  type JsonRpcRequest,
} from "./protocol.js";

export class InProcessRpcClient {
  private readonly runtime: LoafCoreRuntime;
  private readonly router: RpcRouter;

  constructor(runtime: LoafCoreRuntime) {
    this.runtime = runtime;
    this.router = new RpcRouter(runtime);
  }

  onEvent(listener: (event: RuntimeEvent) => void): () => void {
    return this.runtime.onEvent(listener);
  }

  async call<T>(method: string, params?: unknown): Promise<T> {
    const request: JsonRpcRequest = {
      jsonrpc: "2.0",
      id: 1,
      method,
      params,
    };

    try {
      const result = await this.router.dispatch(request);
      return result as T;
    } catch (error) {
      if (isRpcMethodError(error)) {
        throw error;
      }
      throw buildRpcMethodError(
        JSON_RPC_ERROR.INTERNAL_ERROR,
        error instanceof Error ? error.message : String(error),
        {
          reason: "internal_error",
        },
      );
    }
  }
}

export async function createInProcessRpcClient(): Promise<InProcessRpcClient> {
  const runtime = await LoafCoreRuntime.create({ rpcMode: true });
  return new InProcessRpcClient(runtime);
}
