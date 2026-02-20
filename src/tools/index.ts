import { loafConfig } from "../config.js";
import { createExaBuiltinTools } from "./builtin/exa.js";
import { discoverCustomTools } from "./custom.js";
import { JAVASCRIPT_BUILTIN_TOOLS } from "./builtin/javascript.js";
import { createToolRegistry } from "./registry.js";
import { ToolRuntime } from "./runtime.js";

let configuredExaApiKey = loafConfig.exaApiKey;

export function configureBuiltinTools(config: { exaApiKey?: string }): void {
  configuredExaApiKey = (config.exaApiKey ?? "").trim();
}

const EXA_BUILTIN_TOOLS = createExaBuiltinTools({
  getApiKey: () => configuredExaApiKey || loafConfig.exaApiKey,
});

export const defaultToolRegistry = createToolRegistry()
  .registerMany(JAVASCRIPT_BUILTIN_TOOLS)
  .registerMany(EXA_BUILTIN_TOOLS);

export const defaultToolRuntime = new ToolRuntime(defaultToolRegistry);

let customToolsLoadPromise:
  | Promise<{
      searchedDirectories: string[];
      loaded: Array<{ name: string; sourcePath: string }>;
      errors: string[];
    }>
  | null = null;

export async function loadCustomTools(): Promise<{
  searchedDirectories: string[];
  loaded: Array<{ name: string; sourcePath: string }>;
  errors: string[];
}> {
  if (customToolsLoadPromise) {
    return customToolsLoadPromise;
  }

  customToolsLoadPromise = (async () => {
    const discovered = await discoverCustomTools();
    const loaded: Array<{ name: string; sourcePath: string }> = [];
    const errors = [...discovered.errors];

    for (const item of discovered.loaded) {
      try {
        defaultToolRegistry.register(item.tool);
        loaded.push({
          name: item.name,
          sourcePath: item.sourcePath,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        errors.push(`failed registering ${item.sourcePath}: ${message}`);
      }
    }

    return {
      searchedDirectories: discovered.searchedDirectories,
      loaded,
      errors,
    };
  })();

  return customToolsLoadPromise;
}
