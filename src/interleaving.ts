export type ToolCallPreview = {
  name: string;
  input: Record<string, unknown>;
};

export function consumeAssistantBoundary(input: {
  fullText: string;
  emittedChars: number;
}): {
  delta: string;
  emittedChars: number;
} {
  const fullText = input.fullText;
  const start = Math.max(0, Math.min(input.emittedChars, fullText.length));
  return {
    delta: fullText.slice(start),
    emittedChars: fullText.length,
  };
}

export function buildToolReplacement(input: {
  pendingIds: number[];
  toolRows: string[];
}): {
  replacements: Array<{ id: number; row: string }>;
  extraRows: string[];
  consumed: number;
} {
  const replaceCount = Math.min(input.pendingIds.length, input.toolRows.length);
  const replacements: Array<{ id: number; row: string }> = [];
  for (let index = 0; index < replaceCount; index += 1) {
    const id = input.pendingIds[index];
    const row = input.toolRows[index];
    if (typeof id === "number" && row) {
      replacements.push({ id, row });
    }
  }
  return {
    replacements,
    extraRows: input.toolRows.slice(replaceCount),
    consumed: replaceCount,
  };
}

export function parseToolCallPreview(rawCall: unknown): ToolCallPreview | null {
  if (!isRecord(rawCall)) {
    return null;
  }
  const call = isRecord(rawCall.call) ? rawCall.call : rawCall;

  const directName = readTrimmedString(call.name);
  if (directName) {
    return {
      name: directName,
      input: parseToolCallInput(call.input ?? call.arguments),
    };
  }

  const providerToolName = readTrimmedString(call.providerToolName);
  if (providerToolName) {
    return {
      name: providerToolName,
      input: parseToolCallInput(call.args),
    };
  }

  const functionRecord = isRecord(call.function) ? call.function : null;
  const functionName = readTrimmedString(functionRecord?.name);
  if (functionName) {
    return {
      name: functionName,
      input: parseToolCallInput(functionRecord?.arguments),
    };
  }

  return null;
}

function parseToolCallInput(rawInput: unknown): Record<string, unknown> {
  if (isRecord(rawInput)) {
    return rawInput;
  }
  if (typeof rawInput !== "string") {
    return {};
  }
  try {
    const parsed = JSON.parse(rawInput) as unknown;
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function readTrimmedString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
