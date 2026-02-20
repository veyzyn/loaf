import { describe, expect, it } from "vitest";
import type { ChatMessage } from "../chat-types.js";
import {
  buildSkillInstructionBlock,
  buildSkillPromptContext,
  hasSkillMentions,
  mapMessagesForModel,
  transformPromptMentionsForModel,
} from "./prompt.js";
import type { SkillDefinition } from "./types.js";

function makeSkill(name: string, description: string, sourcePath = `/tmp/${name}/SKILL.md`): SkillDefinition {
  return {
    name,
    nameLower: name.toLowerCase(),
    description,
    descriptionPreview: description,
    content: `${name} instructions`,
    sourcePath,
    directoryPath: sourcePath.replace(/\/SKILL\.md$/, ""),
  };
}

describe("transformPromptMentionsForModel", () => {
  it("does not rewrite prompt mentions", () => {
    const skills = [makeSkill("alpha", "alpha description")];
    const output = transformPromptMentionsForModel("please run $alpha then $unknown", skills);
    expect(output).toBe("please run $alpha then $unknown");
  });
});

describe("mapMessagesForModel", () => {
  it("passes messages through unchanged", () => {
    const skills = [makeSkill("alpha", "alpha description")];
    const messages: ChatMessage[] = [
      { role: "user", text: "use $alpha" },
      { role: "assistant", text: "ok" },
    ];

    const mapped = mapMessagesForModel(messages, skills);
    expect(mapped).toEqual(messages);
  });
});

describe("buildSkillInstructionBlock", () => {
  it("returns empty string when there are no skills", () => {
    const output = buildSkillInstructionBlock(
      {
        explicitMentions: [],
        explicit: [],
        autoMatched: [],
        combined: [],
      },
      [],
    );
    expect(output).toBe("");
  });

  it("renders skills catalog and explicit skill payloads", () => {
    const alpha = makeSkill("alpha", "alpha description");
    const beta = makeSkill("beta", "beta description");

    const output = buildSkillInstructionBlock(
      {
        explicitMentions: ["alpha"],
        explicit: [alpha],
        autoMatched: [],
        combined: [alpha],
      },
      [alpha, beta],
    );

    expect(output).toContain("## Skills");
    expect(output).toContain("### Available skills");
    expect(output).toContain("- alpha: alpha description (file: /tmp/alpha/SKILL.md)");
    expect(output).toContain("- beta: beta description (file: /tmp/beta/SKILL.md)");
    expect(output).toContain("<skill>");
    expect(output).toContain("<name>alpha</name>");
    expect(output).toContain("alpha instructions");
    expect(output).not.toContain("<name>beta</name>");
  });
});

describe("buildSkillPromptContext and hasSkillMentions", () => {
  it("supports explicit mentions and keeps model prompt unchanged", () => {
    const alpha = makeSkill("alpha", "alpha automation helper");

    const context = buildSkillPromptContext("please use $alpha", [alpha]);

    expect(hasSkillMentions("please use $alpha")).toBe(true);
    expect(hasSkillMentions("plain prompt")).toBe(false);
    expect(context.modelPrompt).toBe("please use $alpha");
    expect(context.selection.explicit.map((skill) => skill.name)).toEqual(["alpha"]);
    expect(context.selection.autoMatched).toEqual([]);
    expect(context.instructionBlock).toContain("### Available skills");
    expect(context.instructionBlock).toContain("<name>alpha</name>");
  });

  it("does not resolve ambiguous explicit mentions", () => {
    const alphaA = makeSkill("alpha", "alpha A", "/tmp/a/alpha/SKILL.md");
    const alphaB = makeSkill("alpha", "alpha B", "/tmp/b/alpha/SKILL.md");

    const context = buildSkillPromptContext("please use $alpha", [alphaA, alphaB]);

    expect(context.selection.explicit).toEqual([]);
    expect(context.selection.combined).toEqual([]);
    expect(context.instructionBlock).not.toContain("<skill>");
    expect(context.instructionBlock).toContain("### Available skills");
  });
});
