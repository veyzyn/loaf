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

function normalizeMentionToken(raw: string): string {
  return raw
    .trim()
    .replace(/[.,!?;:]+$/g, "")
    .toLowerCase();
}
