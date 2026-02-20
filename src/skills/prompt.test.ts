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

function makeSkill(name: string, description: string): SkillDefinition {
  return {
    name,
    nameLower: name.toLowerCase(),
    description,
    descriptionPreview: description,
    content: `${name} instructions`,
    sourcePath: `/tmp/${name}/SKILL.md`,
    directoryPath: `/tmp/${name}`,
  };
}

describe("transformPromptMentionsForModel", () => {
  it("replaces known mentions and leaves unknown mentions unchanged", () => {
    const skills = [makeSkill("alpha", "alpha description")];
    const output = transformPromptMentionsForModel("please run $alpha then $unknown", skills);
    expect(output).toBe("please run use $alpha skill then $unknown");
  });

  it("preserves punctuation after transformed mentions", () => {
    const skills = [makeSkill("alpha", "alpha description")];
    const output = transformPromptMentionsForModel("please run $alpha.", skills);
    expect(output).toBe("please run use $alpha skill.");
  });
});

describe("mapMessagesForModel", () => {
  it("transforms only user messages", () => {
    const skills = [makeSkill("alpha", "alpha description")];
    const messages: ChatMessage[] = [
      { role: "user", text: "use $alpha" },
      { role: "assistant", text: "ok" },
    ];

    const mapped = mapMessagesForModel(messages, skills);

    expect(mapped[0]).toEqual({ role: "user", text: "use use $alpha skill" });
    expect(mapped[1]).toEqual(messages[1]);
  });
});

describe("buildSkillInstructionBlock", () => {
  it("returns empty string with no selected skills", () => {
    const empty = buildSkillInstructionBlock({
      explicitMentions: [],
      explicit: [],
      autoMatched: [],
      combined: [],
    });
    expect(empty).toBe("");
  });

  it("renders explicit and auto-matched skill sections", () => {
    const alpha = makeSkill("alpha", "alpha description");
    const beta = makeSkill("beta", "beta description");

    const output = buildSkillInstructionBlock({
      explicitMentions: ["alpha"],
      explicit: [alpha],
      autoMatched: [beta],
      combined: [alpha, beta],
    });

    expect(output).toContain("explicit skills: alpha");
    expect(output).toContain("auto-matched skills: beta");
    expect(output).toContain("[skill: alpha]");
    expect(output).toContain("alpha instructions");
    expect(output).toContain("[skill: beta]");
  });

  it("renders explicit none when only auto-matched skills exist", () => {
    const beta = makeSkill("beta", "beta description");
    const output = buildSkillInstructionBlock({
      explicitMentions: [],
      explicit: [],
      autoMatched: [beta],
      combined: [beta],
    });
    expect(output).toContain("explicit skills: none");
    expect(output).toContain("auto-matched skills: beta");
  });
});

describe("buildSkillPromptContext and hasSkillMentions", () => {
  it("builds combined prompt context and mention detection", () => {
    const alpha = makeSkill("alpha", "alpha automation helper");

    const context = buildSkillPromptContext("please use $alpha", [alpha]);

    expect(hasSkillMentions("please use $alpha")).toBe(true);
    expect(hasSkillMentions("plain prompt")).toBe(false);
    expect(context.modelPrompt).toBe("please use use $alpha skill");
    expect(context.selection.explicit.map((skill) => skill.name)).toEqual(["alpha"]);
    expect(context.instructionBlock).toContain("[skill: alpha]");
  });
});
