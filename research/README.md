# Original research scripts

This directory preserves selected original mechanism, context, conditional-task and routing scripts and recorded protocols. Additional original source files are alongside the relevant [`evidence/`](../evidence/README.md) records.

These scripts were run in the original ELF workspace. Some paths and helpers are environment-specific; they depend on large cached states, pretrained models, third-party implementations and datasets that are not included here. They are supplied as provenance and research starting points, not advertised as turnkey reproduction commands. Use `tools/check_evidence.py` for the supported portable CPU checks.

The residual-context experiments reuse previously inspected tests and include focal and shuffled controls; the global-scale round follows the first AURORA transfer failure. The XSum selector was fitted on a separate development set and tested on fresh documents at a fixed call budget. Neither experiment is evidence of a standalone generation algorithm or an end-to-end acceleration.
