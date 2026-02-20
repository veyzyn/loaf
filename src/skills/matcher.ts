import type { SkillDefinition, SkillSelection } from "./types.js";

const MAX_AUTO_SKILLS = 4;
const AUTO_MATCH_THRESHOLD = 6;
const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "if",
  "in",
  "into",
  "is",
  "it",
  "of",
  "on",
  "or",
  "the",
  "to",
  "with",
  "you",
  "your",
]);

const SKILL_MENTION_PATTERN = /(^|[^\w$])\$([a-zA-Z0-9][a-zA-Z0-9._:-]*)/g;

export function parseExplicitSkillMentions(prompt: string): string[] {
  const mentions: string[] = [];
  const seen = new Set<string>();

  for (const match of prompt.matchAll(SKILL_MENTION_PATTERN)) {
    const name = normalizeMentionToken(match[2] as string);
    if (!name || /^\d+$/.test(name) || seen.has(name)) {
      continue;
    }
    mentions.push(name);
    seen.add(name);
  }

  return mentions;
}

export function selectSkillsForPrompt(prompt: string, skills: SkillDefinition[]): SkillSelection {
  const explicitMentions = parseExplicitSkillMentions(prompt);
  const byNameLower = new Map(skills.map((skill) => [skill.nameLower, skill]));
  const explicit = explicitMentions
    .map((name) => byNameLower.get(name))
    .filter((skill): skill is SkillDefinition => Boolean(skill));

  const explicitNameSet = new Set(explicit.map((skill) => skill.nameLower));
  const promptLower = prompt.toLowerCase();
  const promptTokens = new Set(tokenize(promptLower));

  const scoredAuto = skills
    .filter((skill) => !explicitNameSet.has(skill.nameLower))
    .map((skill) => ({
      skill,
      score: scoreSkill(skill, promptLower, promptTokens),
    }))
    .filter((row) => row.score >= AUTO_MATCH_THRESHOLD)
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      return left.skill.name.localeCompare(right.skill.name);
    })
    .slice(0, MAX_AUTO_SKILLS)
    .map((row) => row.skill);

  return {
    explicitMentions,
    explicit,
    autoMatched: scoredAuto,
    combined: [...explicit, ...scoredAuto],
  };
}

function scoreSkill(skill: SkillDefinition, promptLower: string, promptTokens: Set<string>): number {
  let score = 0;

  if (promptLower.includes(skill.nameLower)) {
    score += 12;
  }

  for (const token of tokenize(skill.nameLower)) {
    if (token.length >= 2 && promptTokens.has(token)) {
      score += 4;
    }
  }

  let descriptionPoints = 0;
  for (const token of tokenize(skill.description.toLowerCase())) {
    if (token.length < 4 || STOP_WORDS.has(token)) {
      continue;
    }
    if (promptTokens.has(token)) {
      descriptionPoints += 1;
      if (descriptionPoints >= 6) {
        break;
      }
    }
  }

  return score + descriptionPoints;
}

function tokenize(value: string): string[] {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function normalizeMentionToken(raw: string): string {
  return raw
    .trim()
    .replace(/[.,!?;:]+$/g, "")
    .toLowerCase();
}
