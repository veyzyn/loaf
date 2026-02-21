import { describe, expect, it } from "vitest";
import {
  buildToolReplacement,
  consumeAssistantBoundary,
  parseToolCallPreview,
} from "./interleaving.js";

describe("consumeAssistantBoundary", () => {
  it("returns only unseen assistant text and advances cursor", () => {
    const first = consumeAssistantBoundary({
      fullText: "hello world",
      emittedChars: 0,
    });
    expect(first.delta).toBe("hello world");
    expect(first.emittedChars).toBe(11);

    const second = consumeAssistantBoundary({
      fullText: "hello world",
      emittedChars: first.emittedChars,
    });
    expect(second.delta).toBe("");
    expect(second.emittedChars).toBe(11);

    const third = consumeAssistantBoundary({
      fullText: "hello world again",
      emittedChars: second.emittedChars,
    });
    expect(third.delta).toBe(" again");
    expect(third.emittedChars).toBe(17);
  });

  it("clamps emitted cursor to valid bounds", () => {
    const result = consumeAssistantBoundary({
      fullText: "abc",
      emittedChars: 999,
    });
    expect(result.delta).toBe("");
    expect(result.emittedChars).toBe(3);
  });
});

describe("buildToolReplacement", () => {
  it("maps pending tool rows in order and keeps extra rows", () => {
    const plan = buildToolReplacement({
      pendingIds: [10, 11],
      toolRows: ["row-a", "row-b", "row-c"],
    });
    expect(plan.replacements).toEqual([
      { id: 10, row: "row-a" },
      { id: 11, row: "row-b" },
    ]);
    expect(plan.extraRows).toEqual(["row-c"]);
    expect(plan.consumed).toBe(2);
  });
});

describe("parseToolCallPreview", () => {
  it("parses wrapped incremental tool-call events", () => {
    const parsed = parseToolCallPreview({
      call: {
        name: "bash",
        input: { command: "pwd" },
      },
    });
    expect(parsed?.name).toBe("bash");
    expect(parsed?.input).toEqual({ command: "pwd" });
  });

  it("parses provider-shaped calls with args object", () => {
    const parsed = parseToolCallPreview({
      providerToolName: "run_js",
      args: { script: "1+1" },
    });
    expect(parsed).toEqual({
      name: "run_js",
      input: { script: "1+1" },
    });
  });

  it("parses OpenRouter function-call shape with json arguments", () => {
    const parsed = parseToolCallPreview({
      function: {
        name: "search_web",
        arguments: "{\"query\":\"cats\"}",
      },
    });
    expect(parsed).toEqual({
      name: "search_web",
      input: { query: "cats" },
    });
  });
});
