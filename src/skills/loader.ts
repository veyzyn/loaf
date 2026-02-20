import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { getLoafDataDir } from "../persistence.js";
import type { SkillCatalog, SkillDefinition } from "./types.js";

const SKILL_FILE_NAME = "SKILL.md";
const DESCRIPTION_PREVIEW_WORDS = 8;
const REPO_MARKER_NAME = ".git";
const AGENTS_SKILLS_PATH_SEGMENTS = [".agents", "skills"] as const;
const MAX_SCAN_DEPTH = 6;
const MAX_SKILLS_DIRS_PER_ROOT = 2_000;

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
  const seenSkillPaths = new Set<string>();

  for (const skillsDirectory of directories) {
    if (!fs.existsSync(skillsDirectory)) {
      continue;
    }

    const sourcePaths = discoverSkillFiles(skillsDirectory, errors);
    for (const sourcePath of sourcePaths) {
      if (seenSkillPaths.has(sourcePath)) {
        continue;
      }
      seenSkillPaths.add(sourcePath);

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

      const directoryPath = path.dirname(sourcePath);
      const name = path.basename(directoryPath);
      const nameLower = name.toLowerCase();
      const description = extractSkillDescription(content);
      skills.push({
        name,
        nameLower,
        description,
        descriptionPreview: buildDescriptionPreview(description),
        content,
        sourcePath,
        directoryPath,
      });
    }
  }

  skills.sort((left, right) => left.name.localeCompare(right.name) || left.sourcePath.localeCompare(right.sourcePath));
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

function discoverSkillFiles(skillsDirectory: string, errors: string[]): string[] {
  const queue: Array<{ directoryPath: string; depth: number }> = [];
  const visitedDirectories = new Set<string>();
  const seenSkillPaths = new Set<string>();
  const sourcePaths: string[] = [];
  let truncated = false;

  const enqueueDirectory = (directoryPath: string, depth: number) => {
    if (depth > MAX_SCAN_DEPTH || truncated) {
      return;
    }
    const resolvedPath = resolvePath(directoryPath);
    if (!resolvedPath) {
      return;
    }
    if (visitedDirectories.size >= MAX_SKILLS_DIRS_PER_ROOT) {
      truncated = true;
      return;
    }
    if (visitedDirectories.has(resolvedPath)) {
      return;
    }
    visitedDirectories.add(resolvedPath);
    queue.push({ directoryPath: resolvedPath, depth });
  };

  enqueueDirectory(skillsDirectory, 0);

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current) {
      break;
    }

    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(current.directoryPath, { withFileTypes: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`failed reading skills directory ${current.directoryPath}: ${message}`);
      continue;
    }

    for (const entry of entries) {
      if (entry.name.startsWith(".")) {
        continue;
      }

      const entryPath = path.join(current.directoryPath, entry.name);
      if (entry.isDirectory()) {
        enqueueDirectory(entryPath, current.depth + 1);
        continue;
      }

      if (entry.isSymbolicLink()) {
        let stats: fs.Stats;
        try {
          stats = fs.statSync(entryPath);
        } catch {
          continue;
        }

        if (stats.isDirectory()) {
          enqueueDirectory(entryPath, current.depth + 1);
          continue;
        }

        if (!stats.isFile() || entry.name !== SKILL_FILE_NAME) {
          continue;
        }
      } else if (!entry.isFile() || entry.name !== SKILL_FILE_NAME) {
        continue;
      }

      const resolvedSkillPath = resolvePath(entryPath) ?? path.resolve(entryPath);
      if (seenSkillPaths.has(resolvedSkillPath)) {
        continue;
      }

      seenSkillPaths.add(resolvedSkillPath);
      sourcePaths.push(resolvedSkillPath);
    }
  }

  if (truncated) {
    errors.push(`skills scan truncated under ${skillsDirectory} after ${MAX_SKILLS_DIRS_PER_ROOT} directories`);
  }

  return sourcePaths;
}

function resolvePath(targetPath: string): string | null {
  try {
    return fs.realpathSync(targetPath);
  } catch {
    return null;
  }
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
