---
name: git-release
description: Create consistent releases and changelogs from git history
license: MIT
compatibility: opencode
metadata:
  audience: maintainers
  workflow: github
---

## What I do

- Analyze recent commits and merged PRs to draft release notes
- Propose a semantic version bump based on commit types (feat/fix/breaking)
- Provide a copy-pasteable `gh release create` command

## When to use me

Use this when you are preparing a tagged release. Ask clarifying questions if the target versioning scheme is unclear.

## Workflow

1. Run `git log --oneline --no-decorate <last_tag>..HEAD` to gather changes
2. Categorize commits: Features, Bug Fixes, Breaking Changes, Documentation
3. Propose version bump (major/minor/patch) with reasoning
4. Generate release notes in Markdown
5. Provide final command: `gh release create vX.Y.Z --title "vX.Y.Z" --notes "..."`

## Usage

```
/load git-release
```

Then: "Prepare a release from the last tag v1.2.0"
