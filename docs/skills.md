# loaf skills guide

Skills are local instruction bundles stored in `SKILL.md` files.

## What skills do

- Add domain-specific instructions to a single prompt.
- Let you invoke repeatable workflows by name.
- Keep your normal prompt short while providing richer context.

## How to use a skill

Mention it in your prompt with a leading `$`:

```text
$frontend-design create a pricing page with 3 tiers
```

Tips:

- Skill matching is case-insensitive.
- Skill names support letters, numbers, `.`, `_`, `:`, and `-`.
- The first token beginning with `$` is treated as the skill mention token.
- Use `/skills` to list currently discoverable skills.

## Skill discovery locations

`loaf` scans these directories for `SKILL.md` files:

1. Repo-local `.agents/skills` directories from repo root to current working directory.
2. User data directory: `~/.loaf/skills`.
3. Global user directory: `~/.agents/skills`.

If multiple skill files resolve to the same name token, mention resolution can be ambiguous. In practice, prefer unique skill names.

## TUI skill autocomplete

- Start input with `$` to show skill suggestions.
- Use `Up/Down` to navigate suggestions.
- Use `Tab` to autocomplete.
- `Enter` also autocompletes when you are still completing the skill token.

## Troubleshooting

- Skill not listed in `/skills`: confirm file path ends with `SKILL.md` and folder is under one of the discovery roots.
- Mention not applied: verify exact token (`$skill-name`) and avoid duplicate skill names with the same lowercase form.
- Empty skill ignored: ensure `SKILL.md` has non-empty content.

## Related docs

- [usage.md](usage.md)
- [tools.md](tools.md)
