import type { ChatMessage } from "../chat-types.js";
import { parseExplicitSkillMentions } from "./matcher.js";
import type { SkillDefinition, SkillSelection } from "./types.js";

const SKILLS_USAGE_RULES = `- Discovery: The list above is the skills available in this session (name + description + file path). Skill bodies live on disk at the listed paths.
- Trigger rules: If the user names a skill (with \`$SkillName\` or plain text) OR the task clearly matches a skill's description shown above, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, open its \`SKILL.md\`. Read only enough to follow the workflow.
  2) When \`SKILL.md\` references relative paths (e.g., \`scripts/foo.py\`), resolve them relative to the skill directory listed above first, and only consider other paths if needed.
  3) If \`SKILL.md\` points to extra folders such as \`references/\`, load only the specific files needed for the request; don't bulk-load everything.
  4) If \`scripts/\` exist, prefer running or patching them instead of retyping large code blocks.
  5) If \`assets/\` or templates exist, reuse them instead of recreating from scratch.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Keep context small: summarize long sections instead of pasting them; only load extra files when needed.
  - Avoid deep reference-chasing: prefer opening only files directly linked from \`SKILL.md\` unless you're blocked.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue.`;

export type SkillPromptContext = {
  selection: SkillSelection;
  modelPrompt: string;
  instructionBlock: string;
};

export function buildSkillPromptContext(prompt: string, skills: SkillDefinition[]): SkillPromptContext {
  const explicitMentions = parseExplicitSkillMentions(prompt);
  const explicit = selectExplicitSkillsForPrompt(explicitMentions, skills);
  const selection: SkillSelection = {
    explicitMentions,
    explicit,
    autoMatched: [],
    combined: explicit,
  };

  return {
    selection,
    modelPrompt: transformPromptMentionsForModel(prompt, skills),
    instructionBlock: buildSkillInstructionBlock(selection, skills),
  };
}

export function transformPromptMentionsForModel(prompt: string, _skills: SkillDefinition[]): string {
  return prompt;
}

export function mapMessagesForModel(messages: ChatMessage[], _skills: SkillDefinition[]): ChatMessage[] {
  return messages;
}

export function buildSkillInstructionBlock(selection: SkillSelection, allSkills: SkillDefinition[] = []): string {
  const sections: string[] = [];
  const catalog = buildSkillCatalogBlock(allSkills);
  if (catalog) {
    sections.push(catalog);
  }

  for (const skill of selection.explicit) {
    sections.push(renderExplicitSkillMessage(skill));
  }

  return sections.join("\n\n");
}

export function hasSkillMentions(prompt: string): boolean {
  return parseExplicitSkillMentions(prompt).length > 0;
}

function selectExplicitSkillsForPrompt(explicitMentions: string[], skills: SkillDefinition[]): SkillDefinition[] {
  const skillsByName = new Map<string, SkillDefinition[]>();
  for (const skill of skills) {
    const existing = skillsByName.get(skill.nameLower) ?? [];
    existing.push(skill);
    skillsByName.set(skill.nameLower, existing);
  }

  const selected: SkillDefinition[] = [];
  const seenPaths = new Set<string>();
  for (const mention of explicitMentions) {
    const matches = skillsByName.get(mention) ?? [];
    if (matches.length !== 1) {
      continue;
    }

    const skill = matches[0];
    if (!skill || seenPaths.has(skill.sourcePath)) {
      continue;
    }

    seenPaths.add(skill.sourcePath);
    selected.push(skill);
  }

  return selected;
}

function buildSkillCatalogBlock(allSkills: SkillDefinition[]): string {
  if (allSkills.length === 0) {
    return "";
  }

  const lines: string[] = [
    "## Skills",
    "A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and file path so you can open the source for full instructions when using a specific skill.",
    "### Available skills",
  ];

  for (const skill of allSkills) {
    lines.push(`- ${skill.name}: ${skill.description} (file: ${normalizePath(skill.sourcePath)})`);
  }

  lines.push("### How to use skills");
  lines.push(SKILLS_USAGE_RULES);
  return lines.join("\n");
}

function renderExplicitSkillMessage(skill: SkillDefinition): string {
  return `<skill>\n<name>${skill.name}</name>\n<path>${normalizePath(skill.sourcePath)}</path>\n${skill.content}\n</skill>`;
}

function normalizePath(filePath: string): string {
  return filePath.replace(/\\/g, "/");
}
