# DRAFT: zarr-extensions registry submission

> Status: **draft — not submitted.** The PR to
> [zarr-developers/zarr-extensions](https://github.com/zarr-developers/zarr-extensions)
> is an espg action (englacial/zagg#210 ruling: the agent prepares drafts; the
> registry PR and PyPI publish are human actions).

## How to submit

1. Fork `zarr-developers/zarr-extensions`, create a branch.
2. Copy this repository's registry files into the fork, preserving paths:
   - `registry/data-types/vlen-ndarray/README.md` → `data-types/vlen-ndarray/README.md`
   - `registry/data-types/vlen-ndarray/schema.json` → `data-types/vlen-ndarray/schema.json`
   - `registry/codecs/vlen-ndarray/README.md` → `codecs/vlen-ndarray/README.md`
3. Fix the two cross-repo relative links to the reference implementation if
   desired (they already point at https://github.com/espg/zarr-vlen-ndarray).
4. Run `npx prettier -w **/schema.json` (registry house rule for schema files).
5. Open as a **draft PR** first (the registry README recommends this for
   soliciting feedback), with the description below; mark ready when settled.

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

- [ ] No conflicting name in `data-types/` or `codecs/` (last checked
      2026-07-30: none; closest neighbors are `bytes`, `vlen-bytes`,
      `vlen-utf8`).
- [ ] `schema.json` formatted with prettier.
- [ ] Reference-implementation links resolve (package pushed, README
      rendered).
- [ ] Consider linking zarr-extensions#57 in the PR body (done above) so the
      ZFWG sees the relationship to the container-types discussion.
