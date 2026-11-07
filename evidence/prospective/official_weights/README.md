# Official checkpoint inventory, checked 2026-10-13

Sources: https://huggingface.co/api/models?author=embedded-language-flows&full=true&sort=lastModified&direction=-1&limit=100 and the per-model API (`?blobs=true`); official code https://github.com/lillian039/ELF, including its distillation branch.

The API returned 13 model repositories. The most recently modified repository is ELF-B-PD-owt-torch (2026-06-26), containing checkpoint_19017. No later repository update was returned in this official organization at retrieval time. Repository modification dates are not training dates. The separate ELF-B-owt-teacher repository is included in the inventory; it is not asserted to be identical to the ELF-B-owt-torch checkpoint. The distillation README currently points to ELF-B-owt-torch as its teacher.

The three checkpoints used in the frozen prospective experiment (B, M, PD) exactly match the official current LFS SHA-256 values. `used_weights_comparison.json` compares official values with the local checkpoint hashes frozen before generation. The newest released PD student is therefore already represented in the new 512-sequence, two-generation-seed PD8 condition, without changing the frozen primary protocol. This inventory is provenance, not a new trained-model result. Third-party forks and conditional checkpoints are not represented as new unconditional ELF releases.
