import { describe, expect, it } from "vitest";
import type { StreamChunk } from "./chat-types.js";
import { __openAiInternals } from "./openai.js";

describe("computeUnstreamedAnswerDelta", () => {
  const { computeUnstreamedAnswerDelta } = __openAiInternals;

  it("returns full answer when nothing streamed", () => {
    expect(computeUnstreamedAnswerDelta("hello", "")).toBe("hello");
  });

  it("returns only missing suffix when stream already emitted a prefix", () => {
    expect(computeUnstreamedAnswerDelta("hello world", "hello")).toBe(" world");
  });

  it("returns empty string when stream already emitted the full answer", () => {
    expect(computeUnstreamedAnswerDelta("hello world", "hello world")).toBe("");
  });

  it("returns empty string for mismatched stream content to avoid duplicate spam", () => {
    expect(computeUnstreamedAnswerDelta("hello world", "other")).toBe("");
  });
});

describe("extractAnswerDeltaFromChunk", () => {
  const { extractAnswerDeltaFromChunk } = __openAiInternals;

  it("joins answer segments when segments are present", () => {
    const chunk: StreamChunk = {
      thoughts: [],
      answerText: "ignored",
      segments: [
        { kind: "thought", text: "hmm" },
        { kind: "answer", text: "hello " },
        { kind: "answer", text: "world" },
      ],
    };

    expect(extractAnswerDeltaFromChunk(chunk)).toBe("hello world");
  });

  it("falls back to answerText when no answer segment exists", () => {
    const chunk: StreamChunk = {
      thoughts: [],
      answerText: "hello",
      segments: [{ kind: "thought", text: "hmm" }],
    };

    expect(extractAnswerDeltaFromChunk(chunk)).toBe("hello");
  });
});

describe("selectActionableFunctionCalls", () => {
  const { selectActionableFunctionCalls } = __openAiInternals;

  it("drops duplicate calls and in-progress calls", () => {
    const calls = selectActionableFunctionCalls([
      {
        type: "function_call",
        name: "bash",
        call_id: "call-1",
        arguments: "{\"command\":\"pwd\"}",
        status: "completed",
      },
      {
        type: "function_call",
        name: "bash",
        call_id: "call-1",
        arguments: "{\"command\":\"pwd\"}",
        status: "completed",
      },
      {
        type: "function_call",
        name: "bash",
        call_id: "call-2",
        arguments: "{\"command\":\"ls\"}",
        status: "in_progress",
      },
      {
        type: "message",
      },
    ]);

    expect(calls).toHaveLength(1);
    expect(calls[0]?.call_id).toBe("call-1");
  });
});

describe("extractResponseText", () => {
  const { extractResponseText } = __openAiInternals;

  it("reads direct output_text", () => {
    expect(
      extractResponseText({
        output_text: "direct",
      }),
    ).toBe("direct");
  });

  it("reads message content across supported text shapes", () => {
    expect(
      extractResponseText({
        output: [
          {
            type: "message",
            content: [
              { type: "output_text", text: "first" },
              { type: "text", text: { value: "second" } },
              { type: "text", value: "third" },
            ],
          },
        ],
      }),
    ).toBe("first\n\nsecond\n\nthird");
  });
});
