# Training data

Place the complete training corpora in this directory:

- `gensyntax_cpt.jsonl`: continued pre-training records with a `text` field;
- `gensyntax_sft.jsonl`: supervised records with `instruction`, `input` and `output` fields.

The files are intentionally not included in this code repository. Publish them in a versioned data repository or provide a deterministic retrieval and preprocessing workflow. Record row counts, source accessions, split rules, deduplication and leakage checks, licenses and SHA-256 checksums.
