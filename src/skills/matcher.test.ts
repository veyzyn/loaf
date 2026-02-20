import { describe, expect, it } from "vitest";
import { parseExplicitSkillMentions, selectSkillsForPrompt } from "./matcher.js";
import type { SkillDefinition } from "./types.js";

function makeSkill(name: string, description: string): SkillDefinition {
  return {
    name,
    nameLower: name.toLowerCase(),
    description,
    descriptionPreview: description,
    content: `${name} content`,
    sourcePath: `/tmp/${name}/SKILL.md`,
    directoryPath: `/tmp/${name}`,
  };
}

describe("parseExplicitSkillMentions", () => {
  it("parses unique mentions and ignores numeric-only tokens", () => {
    const mentions = parseExplicitSkillMentions("Use $alpha and $alpha again, price is $25, maybe $beta.");
    expect(mentions).toEqual(["alpha", "beta"]);
  });

  it("returns empty list when no mentions are present", () => {
    expect(parseExplicitSkillMentions("no skill mention here")).toEqual([]);
  });
});

describe("selectSkillsForPrompt", () => {
  const skills = [
    makeSkill("alpha", "generic automation helper"),
    makeSkill("frontend-responsive", "responsive layout breakpoints touch targets and mobile design"),
    makeSkill("database", "postgres indexes and query optimization"),
  ];

  it("includes explicit mentions and additional auto-matched skills", () => {
    const result = selectSkillsForPrompt(
      "please use $alpha and help with responsive mobile layout and touch targets",
      skills,
    );

    expect(result.explicit.map((skill) => skill.name)).toEqual(["alpha"]);
    expect(result.autoMatched.map((skill) => skill.name)).toEqual(["frontend-responsive"]);
    expect(result.combined.map((skill) => skill.name)).toEqual(["alpha", "frontend-responsive"]);
  });

  it("returns empty auto matches below threshold", () => {
    const result = selectSkillsForPrompt("completely unrelated sentence", skills);
    expect(result.explicit).toEqual([]);
    expect(result.autoMatched).toEqual([]);
    expect(result.combined).toEqual([]);
  });

  it("prioritizes exact skill name matches even without explicit mentions", () => {
    const result = selectSkillsForPrompt("alpha can help here", skills);
    expect(result.autoMatched.map((skill) => skill.name)).toContain("alpha");
  });

  it("sorts auto matches by score then by name when scores tie", () => {
    const tiedSkills = [
      makeSkill("aaa-layout", "responsive mobile design"),
      makeSkill("bbb-layout", "responsive mobile design"),
    ];

    const result = selectSkillsForPrompt("responsive mobile design layout", tiedSkills);
    expect(result.autoMatched.map((skill) => skill.name)).toEqual(["aaa-layout", "bbb-layout"]);
  });

  it("caps description scoring after six matching description tokens", () => {
    const denseSkill = makeSkill(
      "dense",
      "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
    );
    const result = selectSkillsForPrompt(
      "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
      [denseSkill],
    );
    expect(result.autoMatched.map((skill) => skill.name)).toEqual(["dense"]);
  });

  it("sorts by score before falling back to name order", () => {
    const scoredSkills = [
      makeSkill("high-score", "responsive touch targets mobile layout"),
      makeSkill("low-score", "touch targets"),
    ];
    const result = selectSkillsForPrompt(
      "high score low score responsive touch targets mobile layout",
      scoredSkills,
    );
    expect(result.autoMatched.map((skill) => skill.name)).toEqual(["high-score", "low-score"]);
  });
});
