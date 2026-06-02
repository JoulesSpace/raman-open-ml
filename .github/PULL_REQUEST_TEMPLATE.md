<!-- Thanks for contributing! Keep PRs focused and small where possible. -->

## What and why
<!-- What does this change, and what problem does it solve? Link any issue: Closes #123 -->

## How it was verified
<!-- Commands you ran and what you observed. Metrics come from actually running, not assumptions. -->
- [ ] `pytest` passes
- [ ] `ruff check .` is clean
- [ ] `bash scripts/folderinfo.sh` passes (new folders carry a `.folderinfo`)
- [ ] `agent-memory/` updated if a decision/insight changed (index updated in the same change)
- [ ] No em-dash characters added (repo convention: spaced hyphen `-`)
- [ ] Commit messages follow `type(scope): summary` (feat/fix/docs/build/chore/refactor/test/perf)

## Notes
<!-- Trade-offs, follow-ups, or anything a reviewer should know. -->
