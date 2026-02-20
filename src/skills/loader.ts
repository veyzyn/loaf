import fs from "node:fs";
import path from "node:path";
import { getLoafDataDir } from "../persistence.js";
import type { SkillCatalog, SkillDefinition } from "./types.js";

const SKILL_FILE_NAME = "SKILL.md";
const DESCRIPTION_PREVIEW_WORDS = 8;

export function getSkillsDirectory(): string {
  return path.join(getLoafDataDir(), "skills");
}

export function loadSkillsCatalog(skillsDirectory = getSkillsDirectory()): SkillCatalog {
  const skills: SkillDefinition[] = [];
  const errors: string[] = [];

  if (!fs.existsSync(skillsDirectory)) {
    return {
      directory: skillsDirectory,
      skills,
      errors,
    };
  }

  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(skillsDirectory, { withFileTypes: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      directory: skillsDirectory,
      skills,
      errors: [`failed reading skills directory ${skillsDirectory}: ${message}`],
    };
  }

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }

    const directoryPath = path.join(skillsDirectory, entry.name);
    const sourcePath = path.join(directoryPath, SKILL_FILE_NAME);
    if (!fs.existsSync(sourcePath)) {
      continue;
    }

    let rawContent = "";
    try {
      rawContent = fs.readFileSync(sourcePath, "utf8");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`failed reading ${sourcePath}: ${message}`);
      continue;
    }

    const content = rawContent.trim();
    if (!content) {
      errors.push(`skipped ${sourcePath}: empty skill content`);
      continue;
    }

    const description = extractSkillDescription(content);
    skills.push({
      name: entry.name,
      nameLower: entry.name.toLowerCase(),
      description,
      descriptionPreview: buildDescriptionPreview(description),
      content,
      sourcePath,
      directoryPath,
    });
  }

  skills.sort((left, right) => left.name.localeCompare(right.name));

  return {
    directory: skillsDirectory,
    skills,
    errors,
  };
}

export function extractSkillDescription(content: string): string {
  const normalized = content.replace(/\r\n/g, "\n");
  const paragraphs = normalized
    .split(/\n\s*\n/g)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

  for (const paragraph of paragraphs) {
    const cleaned = cleanMarkdownParagraph(paragraph);
    if (cleaned) {
      return cleaned;
    }
  }

  return "no description provided";
}

export function buildDescriptionPreview(description: string): string {
  const words = description.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) {
    return "no description provided";
  }

  if (words.length <= DESCRIPTION_PREVIEW_WORDS) {
    return words.join(" ");
  }

  return `${words.slice(0, DESCRIPTION_PREVIEW_WORDS).join(" ")}...`;
}

function cleanMarkdownParagraph(paragraph: string): string {
  const lines = paragraph
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !line.startsWith("#"))
    .filter((line) => line !== "```");

  if (lines.length === 0) {
    return "";
  }

  const compact = lines.join(" ").replace(/\s+/g, " ").trim();
  const withoutDecorators = compact
    .replace(/^[*-]\s+/, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .trim();

  return withoutDecorators || "";
}
