import { describe, expect, it } from "vitest";
import { parseExplicitSkillMentions } from "./matcher.js";

describe("parseExplicitSkillMentions", () => {
  it("parses unique mentions and ignores numeric-only tokens", () => {
    const mentions = parseExplicitSkillMentions("Use $alpha and $alpha again, price is $25, maybe $beta.");
    expect(mentions).toEqual(["alpha", "beta"]);
  });

  it("returns empty list when no mentions are present", () => {
    expect(parseExplicitSkillMentions("no skill mention here")).toEqual([]);
  });
});
