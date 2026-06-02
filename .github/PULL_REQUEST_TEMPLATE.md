<!-- Template adapted from stevemao/github-issue-templates (simple + checklist). -->

Fixes #

## Proposed Changes

  -
  -
  -

## Checklist

### All submissions
* [ ] Followed the guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md)
* [ ] Checked there are no other open [Pull Requests](../../../pulls) for the same change

### Verified locally
* [ ] `pytest` passes
* [ ] `ruff check .` is clean
* [ ] `bash scripts/folderinfo.sh` passes (new folders carry a `.folderinfo`)
* [ ] `agent-memory/` updated if a decision/insight changed (index updated in the same change)
* [ ] No em-dash characters added; commit messages follow `type(scope): summary`
