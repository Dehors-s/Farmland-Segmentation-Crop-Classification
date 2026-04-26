---
name: pr-review
description: Review pull requests for code quality, security, and performance
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: github
---

## What I do

- Review PR diffs line-by-line for correctness and style
- Check for common security vulnerabilities (injection, auth, secrets)
- Identify performance bottlenecks and suggest optimizations
- Verify test coverage and suggest missing edge cases
- Check API compatibility and breaking changes

## Review checklist

### Functionality
- Does the code do what the PR description says?
- Are edge cases handled (empty states, errors, timeouts)?
- Are there any logical errors or race conditions?

### Security
- Are user inputs validated/sanitized?
- Are secrets, tokens, or credentials exposed?
- Is authentication/authorization correctly implemented?
- Are there any SQL injection or XSS vectors?

### Performance
- Are there N+1 queries or excessive loops?
- Are large datasets handled efficiently (streaming, pagination)?
- Are caching opportunities missed?

### Maintainability
- Is the code readable and well-structured?
- Are there appropriate error messages and logging?
- Are magic numbers replaced with named constants?
- Is the change backward-compatible?

### Testing
- Are there unit tests for new logic?
- Do tests cover edge cases and error paths?
- Are integration tests needed for this change?

## Output format

```
## Summary
<1-2 sentence overview>

## Issues Found
- **Severity: High/Medium/Low** — Description with file:line reference

## Suggestions
- Optional improvements or refactoring ideas

## Verdict
Approve / Changes requested / Comment
```
