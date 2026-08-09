# Publishing (trusted publishing via OIDC)

> Status: **live.** Trusted publishing is set up and has shipped releases
> (`0.1.0`, `0.1.1`). The workflow is
> [`.github/workflows/publish.yml`](../.github/workflows/publish.yml), which
> is the source of truth — this document records the one-time setup and how
> to cut a release.

Local builds work too (`uv build` produces sdist + wheel; hatch-vcs derives
the version from git tags, writing `src/zarr_vlen_ndarray/_version.py`).

## One-time setup (done)

1. **PyPI**: on pypi.org, a *pending publisher* was added for project
   `zarr-vlen-ndarray`: owner `espg`, repository `zarr-vlen-ndarray`,
   workflow `publish.yml`, environment `pypi`.
2. **GitHub**: the `pypi` environment exists in the repo settings
   (Settings → Environments); reviewers may optionally be required on it.

## Cutting a release

Push a `*.*.*` tag — the version comes from the tag via hatch-vcs, so no file
needs editing:

```sh
git tag 0.1.2
git push --tags
```

The workflow then runs the test matrix, builds sdist + wheel, publishes to
PyPI via trusted publishing (OIDC — no token secrets), and creates a GitHub
release with the artifacts attached. Watch it with
`gh run watch $(gh run list --workflow=publish.yml --limit 1 --json databaseId -q '.[0].databaseId')`.

PyPI's JSON API caches for a few minutes, so a freshly published version may
not show up as `info.version` immediately; `https://pypi.org/pypi/zarr-vlen-ndarray/<version>/json`
reflects it right away.

Note: `test.yml` declares `workflow_call`, so `publish.yml` reuses it via
`uses:` rather than duplicating the matrix.
