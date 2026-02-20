export type ChatMessage = {
  role: "user" | "assistant";
  text: string;
};

export type ModelResult = {
  thoughts: string[];
  answer: string;
};

export type StreamChunk = {
  thoughts: string[];
  answerText: string;
};

export type DebugEvent = {
  stage: string;
  data: unknown;
};
