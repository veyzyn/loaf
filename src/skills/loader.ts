import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { getLoafDataDir } from "../persistence.js";
import type { SkillCatalog, SkillDefinition } from "./types.js";

const SKILL_FILE_NAME = "SKILL.md";
const DESCRIPTION_PREVIEW_WORDS = 8;
const REPO_MARKER_NAME = ".git";
const AGENTS_SKILLS_PATH_SEGMENTS = [".agents", "skills"] as const;

export function getSkillsDirectory(): string {
  return path.join(getLoafDataDir(), "skills");
}

export function getRepoSkillsDirectories(cwd = process.cwd()): string[] {
  const resolvedCwd = path.resolve(cwd);
  const projectRoot = findProjectRootFromPath(resolvedCwd);
  if (!projectRoot) {
    return [];
  }

  const candidates = getDirectoriesBetween(projectRoot, resolvedCwd)
    .map((dirPath) => path.join(dirPath, ...AGENTS_SKILLS_PATH_SEGMENTS))
    .filter((skillsPath) => {
      try {
        return fs.statSync(skillsPath).isDirectory();
      } catch {
        return false;
      }
    });

  return dedupePathList(candidates);
}

export function getSkillsDirectories(): string[] {
  const candidates = [
    ...getRepoSkillsDirectories(),
    getSkillsDirectory(),
    path.join(os.homedir(), ".agents", "skills"),
  ];

  return dedupePathList(candidates);
}

export function loadSkillsCatalog(skillsDirectories: string | string[] = getSkillsDirectories()): SkillCatalog {
  const directories = Array.isArray(skillsDirectories)
    ? skillsDirectories.map((entry) => entry.trim()).filter(Boolean)
    : [skillsDirectories.trim()].filter(Boolean);
  const skills: SkillDefinition[] = [];
  const errors: string[] = [];
  const seenSkillNames = new Set<string>();

  for (const skillsDirectory of directories) {
    if (!fs.existsSync(skillsDirectory)) {
      continue;
    }

    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(skillsDirectory, { withFileTypes: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`failed reading skills directory ${skillsDirectory}: ${message}`);
      continue;
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

      const nameLower = entry.name.toLowerCase();
      if (seenSkillNames.has(nameLower)) {
        errors.push(`skipped ${sourcePath}: duplicate skill name "${entry.name}"`);
        continue;
      }
      seenSkillNames.add(nameLower);

      const description = extractSkillDescription(content);
      skills.push({
        name: entry.name,
        nameLower,
        description,
        descriptionPreview: buildDescriptionPreview(description),
        content,
        sourcePath,
        directoryPath,
      });
    }
  }

  skills.sort((left, right) => left.name.localeCompare(right.name));
  const directoryLabel = directories.join(", ");

  return {
    directory: directoryLabel,
    directories,
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

function findProjectRootFromPath(cwd: string): string | null {
  let current = cwd;
  while (true) {
    if (fs.existsSync(path.join(current, REPO_MARKER_NAME))) {
      return current;
    }

    const parent = path.dirname(current);
    if (parent === current) {
      return null;
    }
    current = parent;
  }
}

function getDirectoriesBetween(root: string, cwd: string): string[] {
  const stack: string[] = [];
  let current = cwd;

  while (true) {
    stack.push(current);
    if (current === root) {
      break;
    }

    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }

  return stack.reverse();
}

function dedupePathList(candidates: string[]): string[] {
  const deduped: string[] = [];
  for (const candidate of candidates) {
    const normalized = candidate.trim();
    if (!normalized || deduped.includes(normalized)) {
      continue;
    }
    deduped.push(normalized);
  }
  return deduped;
}
