import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildDescriptionPreview,
  extractSkillDescription,
  getSkillsDirectory,
  loadSkillsCatalog,
} from "./loader.js";

const tempDirs: string[] = [];

afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

describe("loadSkillsCatalog", () => {
  it("returns empty catalog when directory does not exist", () => {
    const missingDir = path.join(os.tmpdir(), `loaf-missing-${Date.now()}`);
    const catalog = loadSkillsCatalog(missingDir);

    expect(catalog.directory).toBe(missingDir);
    expect(catalog.skills).toEqual([]);
    expect(catalog.errors).toEqual([]);
  });

  it("loads valid skills, ignores missing SKILL.md, and reports empty files", () => {
    const rootDir = fs.mkdtempSync(path.join(os.tmpdir(), "loaf-skills-"));
    tempDirs.push(rootDir);

    const alphaDir = path.join(rootDir, "alpha-skill");
    fs.mkdirSync(alphaDir, { recursive: true });
    fs.writeFileSync(
      path.join(alphaDir, "SKILL.md"),
      "# Alpha\n\nalpha skill helps with command-line automation and debugging workflows.",
      "utf8",
    );

    const zetaDir = path.join(rootDir, "zeta-skill");
    fs.mkdirSync(zetaDir, { recursive: true });
    fs.writeFileSync(path.join(zetaDir, "SKILL.md"), "# Zeta\n\nshort desc", "utf8");

    const missingSkillFileDir = path.join(rootDir, "missing-file");
    fs.mkdirSync(missingSkillFileDir, { recursive: true });

    fs.writeFileSync(path.join(rootDir, "README.txt"), "not a skill folder", "utf8");

    const emptySkillDir = path.join(rootDir, "empty-skill");
    fs.mkdirSync(emptySkillDir, { recursive: true });
    fs.writeFileSync(path.join(emptySkillDir, "SKILL.md"), "\n\n", "utf8");

    const catalog = loadSkillsCatalog(rootDir);

    expect(catalog.skills.map((skill) => skill.name)).toEqual(["alpha-skill", "zeta-skill"]);
    expect(catalog.skills[0]?.description).toBe("alpha skill helps with command-line automation and debugging workflows.");
    expect(catalog.skills[0]?.descriptionPreview).toBe("alpha skill helps with command-line automation and debugging...");
    expect(catalog.skills[1]?.descriptionPreview).toBe("short desc");
    expect(catalog.errors).toHaveLength(1);
    expect(catalog.errors[0]).toContain("empty skill content");
  });

  it("reports read failures when SKILL.md exists as a directory", () => {
    const rootDir = fs.mkdtempSync(path.join(os.tmpdir(), "loaf-skills-read-error-"));
    tempDirs.push(rootDir);

    const brokenDir = path.join(rootDir, "broken-skill");
    fs.mkdirSync(path.join(brokenDir, "SKILL.md"), { recursive: true });

    const catalog = loadSkillsCatalog(rootDir);
    expect(catalog.skills).toEqual([]);
    expect(catalog.errors).toHaveLength(1);
    expect(catalog.errors[0]).toContain("failed reading");
  });

  it("reports directory read failures", () => {
    const filePath = path.join(os.tmpdir(), `loaf-skill-file-${Date.now()}.txt`);
    fs.writeFileSync(filePath, "not a directory", "utf8");
    tempDirs.push(filePath);

    const catalog = loadSkillsCatalog(filePath);

    expect(catalog.skills).toEqual([]);
    expect(catalog.errors).toHaveLength(1);
    expect(catalog.errors[0]).toContain("failed reading skills directory");
  });

  it("handles non-Error throws from directory reads", () => {
    const rootDir = fs.mkdtempSync(path.join(os.tmpdir(), "loaf-skills-non-error-"));
    tempDirs.push(rootDir);
    vi.spyOn(fs, "readdirSync").mockImplementation(() => {
      throw "boom";
    });

    const catalog = loadSkillsCatalog(rootDir);
    expect(catalog.errors).toHaveLength(1);
    expect(catalog.errors[0]).toContain("boom");
  });

  it("handles non-Error throws from skill file reads", () => {
    const rootDir = fs.mkdtempSync(path.join(os.tmpdir(), "loaf-skills-file-non-error-"));
    tempDirs.push(rootDir);

    const alphaDir = path.join(rootDir, "alpha");
    fs.mkdirSync(alphaDir, { recursive: true });
    fs.writeFileSync(path.join(alphaDir, "SKILL.md"), "content", "utf8");

    vi.spyOn(fs, "readFileSync").mockImplementation(() => {
      throw "read failed";
    });

    const catalog = loadSkillsCatalog(rootDir);
    expect(catalog.skills).toEqual([]);
    expect(catalog.errors).toHaveLength(1);
    expect(catalog.errors[0]).toContain("read failed");
  });
});

describe("extractSkillDescription", () => {
  it("returns first non-heading paragraph and strips markdown decorators", () => {
    const description = extractSkillDescription(
      "# Skill\n\n**Build** reliable `tools` quickly.\n\n## Next\n\nOther content.",
    );

    expect(description).toBe("Build reliable tools quickly.");
  });

  it("falls back when only headings are present", () => {
    const description = extractSkillDescription("# Title\n\n## Subtitle\n\n### More");
    expect(description).toBe("no description provided");
  });

  it("falls back when markdown cleanup removes all content", () => {
    const description = extractSkillDescription("** **");
    expect(description).toBe("no description provided");
  });
});

describe("helpers", () => {
  it("returns a default preview for empty descriptions", () => {
    expect(buildDescriptionPreview("")).toBe("no description provided");
  });

  it("returns full preview for short descriptions", () => {
    expect(buildDescriptionPreview("small summary")).toBe("small summary");
  });

  it("resolves ~/.loaf/skills directory", () => {
    const skillsDirectory = getSkillsDirectory();
    expect(skillsDirectory.endsWith(path.join(".loaf", "skills"))).toBe(true);
  });
});
