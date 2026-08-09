# DRAFT: zarr-extensions registry submission

> Status: **draft — not submitted.** The PR to
> [zarr-developers/zarr-extensions](https://github.com/zarr-developers/zarr-extensions)
> is an espg action (englacial/zagg#210 ruling: the agent prepares drafts; the
> registry PR and PyPI publish are human actions).

## How to submit

Steps 1–4 are **done**: the fork is `espg/zarr-extensions`, branch
`vlen-array`, pushed, holding one commit ("Register vlen-ndarray data type
and codec") on top of an upstream-synced `main`. Only step 5 — opening the
PR — remains, and that is an espg action.

1. ~~Fork `zarr-developers/zarr-extensions`, create a branch.~~
2. ~~Copy this repository's registry files into the fork, preserving paths:~~
   - `registry/data-types/vlen-ndarray/README.md` → `data-types/vlen-ndarray/README.md`
   - `registry/data-types/vlen-ndarray/schema.json` → `data-types/vlen-ndarray/schema.json`
   - `registry/codecs/vlen-ndarray/README.md` → `codecs/vlen-ndarray/README.md`
   - `registry/codecs/vlen-ndarray/schema.json` → `codecs/vlen-ndarray/schema.json`
     (added during prep: the registry README asks for a `schema.json` per
     extension and every sibling codec has one)
3. ~~Fix the two cross-repo relative links to the reference implementation if
   desired~~ — left pointing at https://github.com/espg/zarr-vlen-ndarray,
   which resolves publicly.
4. ~~Run `npx prettier -w **/schema.json`~~ — both schema files were already
   prettier-clean.
5. Open as a **draft PR** first (the registry README recommends this for
   soliciting feedback), with the description below; mark ready when settled:

       gh pr create --repo zarr-developers/zarr-extensions \
         --head espg:vlen-array --base main --draft \
         --title "Register vlen-ndarray data type and codec"

Review is by the Zarr Format Working Group; the registry README notes review
focuses on name clarity/conflicts and formal requirements. All registry
documents are CC-BY-3.0 — opening the PR accepts that license for these files.

## Suggested PR title

    Register vlen-ndarray data type and codec

## Suggested PR description

---

This PR registers `vlen-ndarray`, a data type for arrays whose elements are
**variable-length ndarrays** — each element has shape `(n, *inner_shape)`
with `n` varying per element and the trailing `inner_shape` plus the scalar
`dtype` fixed in the configuration — together with its paired parameter-free
`array -> bytes` codec:

```json
{
  "data_type": {
    "name": "vlen-ndarray",
    "configuration": {"dtype": "float32", "inner_shape": [2]}
  },
  "codecs": [{"name": "vlen-ndarray"}]
}
```

**Wire format.** The codec serializes each element as its raw little-endian
C-order bytes and frames chunks exactly as the registered `vlen-bytes` codec
(u32le count; per element u32le length + payload). Encoded chunks are
therefore byte-identical to `bytes` + `vlen-bytes` chunks holding each
element's raw bytes: existing stores using that common ragged-data convention
upgrade to the typed form with a metadata-only rewrite, and implementations
without `vlen-ndarray` support can read the data as `bytes` + `vlen-bytes`
after metadata substitution.

**Relation to existing discussions.** This is a concrete, fixed-inner-shape
point in the "generic container data types" design space discussed in #57,
scoped to fixed-size numeric/boolean core scalar types so every element has a
well-defined item size (element length is implied by encoded byte length —
no per-element shape storage). It does not preclude a more general
`variable_length[<base>]` family later.

**Implementation.** A working implementation (zarr-python >= 3.1, registered
via `zarr.data_type` / `zarr.codecs` entry points) with round-trip,
sharding, and byte-identity tests:
https://github.com/espg/zarr-vlen-ndarray

**Motivating use.** The [zagg](https://github.com/englacial/zagg)
aggregation pipeline (and its `moczarr` reader) stores per-cell t-digest
centroid sets (`(n, 2)` float32) and location lists (`(n,)` uint64) on
HEALPix grids; the store spec's `/2` revision cites this name
(englacial/zagg#340, englacial/zagg#210).

---

## Pre-submission checklist

- [x] No conflicting name in `data-types/` or `codecs/` (re-checked
      2026-08-08 against upstream `4da7b37`: none; closest neighbors are
      `bytes`, `vlen-bytes`, `vlen-utf8`).
- [x] `schema.json` formatted with prettier (both files already clean under
      default settings; the registry has no prettier config).
- [x] Reference-implementation links resolve (package pushed, README
      rendered). All external links in both READMEs return 200 —
      `zarr-vlen-ndarray`, `zagg`, numcodecs, zarr-specs — which matters
      because the registry's only CI job is a lychee link check with
      `fail: true`. All relative links resolve within the fork.
- [x] Schemas validate every example in the two READMEs, and reject a
      non-core `dtype`, a missing `inner_shape`, a bare-string data type
      name, and an unknown codec configuration key.
- [x] Consider linking zarr-extensions#57 in the PR body (done above) so the
      ZFWG sees the relationship to the container-types discussion.

## Note on the codec schema shape

`codecs/vlen-ndarray/schema.json` uses the plain object form (as `reshape`
and `packbits` do), not the `oneOf` with a bare-string alternative that
`vlen-bytes` and `vlen-utf8` carry. That alternative exists for the legacy
v2 string-name spelling of those codecs; a newly registered v3 codec has no
such history, and the codec README already requires the `name` member.
`configuration` is permitted but must be empty.
