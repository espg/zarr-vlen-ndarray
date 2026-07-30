# Publishing (trusted publishing via OIDC)

> Status: **not yet set up** — publishing is an espg action
> (englacial/zagg#210 ruling). This documents the exact steps, mirroring
> moczarr's working setup (`espg/moczarr`, `.github/workflows/publish.yml`).

Local builds already work (`uv build` produces sdist + wheel; hatch-vcs
derives the version from git tags, writing `src/zarr_vlen_ndarray/_version.py`).

## One-time setup

1. **PyPI**: on pypi.org, add a *pending publisher* for project
   `zarr-vlen-ndarray`: owner `espg`, repository `zarr-vlen-ndarray`,
   workflow `publish.yml`, environment `pypi`.
2. **TestPyPI**: same on test.pypi.org, environment `testpypi`.
3. **GitHub**: create the two environments (`testpypi`, `pypi`) in the repo
   settings (Settings → Environments); optionally require reviewers on `pypi`.
4. Add the workflow below as `.github/workflows/publish.yml` (deliberately
   not committed yet, so no accidental publish path exists before the PyPI
   side is registered).
5. **Release**: push a `*.*.*` tag (e.g. `0.1.0`). The workflow runs tests,
   builds, publishes to TestPyPI then PyPI, and creates a GitHub release with
   the artifacts attached.

## `publish.yml` (copy verbatim once step 1–3 are done)

```yaml
name: Publish

# Tag-driven release: pushing a *.*.* tag builds sdist+wheel, publishes to
# TestPyPI then PyPI via trusted publishing (OIDC — no token secrets; the
# one-time PyPI-side registration binds project `zarr-vlen-ndarray` to this
# repo, workflow `publish.yml`, environments `testpypi`/`pypi`), then creates
# the GitHub release with the artifacts attached.

on:
  push:
    tags:
      - "*.*.*"

jobs:
  test:
    uses: ./.github/workflows/test.yml

  build:
    name: Build distribution
    needs: test
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v5
      with:
        fetch-depth: 0  # hatch-vcs derives the version from the tag

    - name: Install uv
      uses: astral-sh/setup-uv@v6

    - name: Build sdist and wheel
      run: uv build

    - name: Store distribution packages
      uses: actions/upload-artifact@v4
      with:
        name: python-package-distributions
        path: dist/

  publish-to-testpypi:
    name: Publish to TestPyPI
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: testpypi
      url: https://test.pypi.org/p/zarr-vlen-ndarray
    permissions:
      id-token: write  # trusted publishing (OIDC)

    steps:
    - name: Download dists
      uses: actions/download-artifact@v4
      with:
        name: python-package-distributions
        path: dist/

    - name: Publish to TestPyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        repository-url: https://test.pypi.org/legacy/
        verbose: true

  publish-to-pypi:
    name: Publish to PyPI
    needs: publish-to-testpypi
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/zarr-vlen-ndarray
    permissions:
      id-token: write  # trusted publishing (OIDC)

    steps:
    - name: Download dists
      uses: actions/download-artifact@v4
      with:
        name: python-package-distributions
        path: dist/

    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1

  github-release:
    name: Create GitHub Release
    needs: publish-to-pypi
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
    - name: Download dists
      uses: actions/download-artifact@v4
      with:
        name: python-package-distributions
        path: dist/

    - name: Create GitHub Release + attach artifacts
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: |
        TAG="${GITHUB_REF#refs/tags/}"
        cat > release_notes.md <<EOF
        ## Installation

        \`\`\`sh
        pip install zarr-vlen-ndarray==$TAG
        \`\`\`
        EOF
        gh release create "$TAG" \
          --repo '${{ github.repository }}' \
          --title "Release $TAG" \
          --notes-file release_notes.md || echo "Release already exists"
        gh release upload "$TAG" dist/* \
          --repo '${{ github.repository }}' --clobber
```

Note: `test.yml` already declares `workflow_call`, so the `uses:` reuse above
works as-is.
