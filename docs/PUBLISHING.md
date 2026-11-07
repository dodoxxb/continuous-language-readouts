# Publishing the prepared repository

This folder is the prepared first version of the public repository `https://github.com/dodoxxb/continuous-language-readouts`. The source tree is pushed from here; the five fitted-checkpoint bundles are attached separately to the `v0.1.0` GitHub Release because they exceed GitHub's ordinary file limit.

The Git index contains source, documentation, scalar records and the CALM document-level aggregates. Large `.pt` files and the five `weights/bundles/*.tar` archives are intentionally ignored by Git. The full manuscript evidence ZIP, raw XSum articles and the original CALM source-token caches are not part of the Git index.

After reviewing this folder, a repository owner with GitHub access can publish the indexed files from this directory, using their own configured Git identity. The prepared folder already has a Git index and `origin` is configured. If working from the separate code-only archive, first run `git init --initial-branch=main`, `git add --all`, and `git remote add origin https://github.com/dodoxxb/continuous-language-readouts.git` in the extracted folder.

```bash
git status --short
git commit -m "Initial release: readout study, reproducibility tools and selected evidence"
git push -u origin main
```

Use `git remote -v` to inspect the configured destination. Confirm that the repository and README are publicly accessible after the push.

The checkpoint terms are finalized in `WEIGHTS_NOTICE.txt` and mirrored in `weights/README.md`, `weights/manifest.json` and the release notes; confirm these four statements agree before publishing. Then create a GitHub Release with the exact tag `v0.1.0` and upload the five independently verified files in `weights/bundles/`. The direct links in `weights/README.md` use this exact tag and filenames. Release notes are prepared in `docs/RELEASE_NOTES_v0.1.0.md`. Sizes, SHA-256 values and contained checkpoint counts are in `weights/BUNDLES.json`; individual checkpoints and validation selections are in `weights/manifest.json`. After uploading, download each asset and verify the recorded hash. The pretrained upstream generators and datasets remain external.

The manuscript lists this repository for code and its Releases page for selected checkpoints; these locations must be populated before submission. Publish the code and release assets before submitting, then check that the repository, five asset URLs and arXiv server preview work.
