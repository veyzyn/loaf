import type { ChatMessage } from "../chat-types.js";
import { parseExplicitSkillMentions, selectSkillsForPrompt } from "./matcher.js";
import type { SkillDefinition, SkillSelection } from "./types.js";

const SKILL_MENTION_PATTERN = /(^|[^\w$])\$([a-zA-Z0-9][a-zA-Z0-9._:-]*)/g;

export type SkillPromptContext = {
  selection: SkillSelection;
  modelPrompt: string;
  instructionBlock: string;
};

export function buildSkillPromptContext(prompt: string, skills: SkillDefinition[]): SkillPromptContext {
  const selection = selectSkillsForPrompt(prompt, skills);
  return {
    selection,
    modelPrompt: transformPromptMentionsForModel(prompt, skills),
    instructionBlock: buildSkillInstructionBlock(selection),
  };
}

export function transformPromptMentionsForModel(prompt: string, skills: SkillDefinition[]): string {
  const knownNames = new Set(skills.map((skill) => skill.nameLower));

  return prompt.replace(SKILL_MENTION_PATTERN, (full, prefix: string, rawName: string) => {
    const { mentionName, suffix } = splitMentionToken(rawName);
    const nameLower = mentionName.toLowerCase();
    if (!knownNames.has(nameLower)) {
      return full;
    }
    return `${prefix}use $${mentionName} skill${suffix}`;
  });
}

export function mapMessagesForModel(messages: ChatMessage[], skills: SkillDefinition[]): ChatMessage[] {
  return messages.map((message) => {
    if (message.role !== "user") {
      return message;
    }
    return {
      ...message,
      text: transformPromptMentionsForModel(message.text, skills),
    };
  });
}

export function buildSkillInstructionBlock(selection: SkillSelection): string {
  if (selection.combined.length === 0) {
    return "";
  }

  const explicitNames = selection.explicit.map((skill) => skill.name);
  const autoNames = selection.autoMatched.map((skill) => skill.name);

  const sections: string[] = [
    "skills context: evaluate and apply the selected skills for this request.",
    `explicit skills: ${explicitNames.length > 0 ? explicitNames.join(", ") : "none"}`,
    `auto-matched skills: ${autoNames.length > 0 ? autoNames.join(", ") : "none"}`,
  ];

  for (const skill of selection.combined) {
    sections.push(`[skill: ${skill.name}]\n${skill.content}`);
  }

  return sections.join("\n\n");
}

export function hasSkillMentions(prompt: string): boolean {
  return parseExplicitSkillMentions(prompt).length > 0;
}

function splitMentionToken(rawName: string): { mentionName: string; suffix: string } {
  const token = rawName.trim();
  const mentionName = token.replace(/[.,!?;:]+$/g, "");
  const suffix = token.slice(mentionName.length);
  return {
    mentionName,
    suffix,
  };
}
