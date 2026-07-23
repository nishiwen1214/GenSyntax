# Data documentation

## Public data locations

- repository examples and selected phenotype tables: `Data/`;
- released Hugging Face dataset: <https://huggingface.co/datasets/ShiwenNi/GenSyntax-data>;
- released evaluation files: <https://huggingface.co/datasets/ShiwenNi/GenSyntax-data/tree/main/Data>.

The GitHub repository includes the Task 1 test file locally. The public Hugging Face `Data/` directory contains the released test sets for all four evaluation tasks:

| Task | Public test file | Records |
|---|---|---:|
| Task 1: plasmid host identification | `gene_task1_test_1000_format.json` | 1,000 |
| Task 2: four-option gene-product disambiguation | `gene_task2_test_500_opts.json` | 500 |
| Task 2: eight-option gene-product disambiguation | `gene_task2_test_500_opts_8.json` | 500 |
| Task 3: three-contig ordering | `gene_task3_test_500_contig3_format.json` | 500 |
| Task 3: four-contig ordering | `gene_task3_test_500_contig4_format.json` | 500 |
| Task 3: five-contig ordering | `gene_task3_test_500_contig5_format.json` | 500 |
| Task 4: gene essentiality | `gene_task4_test_1000_format.json` | 1,000 |

The Task 1 evaluator additionally uses `Data/genus_taxonomy.csv` to map each
reference genus to class, order and family. The table contains the columns
`Genus`, `Class`, `Order` and `Family`; header matching in the evaluator is
case-insensitive.

Clone the released dataset with:

```bash
git clone https://huggingface.co/datasets/ShiwenNi/GenSyntax-data
```

The evaluation scripts should then receive files from `GenSyntax-data/Data/` through their `--references` arguments. The dataset revision used for reported manuscript results should be pinned in the final release. A complete dataset card, checksums and an artifact inventory are still recommended for archival publication.

## Inference schema

The root Python scripts expect:

```json
[
  {
    "instruction": "complete task prompt"
  }
]
```

The public dataset uses:

```json
[
  {
    "Input": "complete task prompt",
    "Output": "reference answer"
  }
]
```

The release should standardize one schema or provide a deterministic converter. Every record should also have a stable identifier, task name, split, source accession, label and provenance fields.

## Required provenance by task

### Pre-training corpus

Provide the NCBI assembly accession, RefSeq release/retrieval date, replicon accession, replicon type, taxonomic identifiers, product count, source GBFF checksum and split/exclusion status. Include the exact extraction script and feature/product selection rules.

### Plasmid host identification

Provide stable plasmid and host accessions; taxonomy database/version; order, family, genus, species and strain labels; number of annotated products; pre-training/fine-tuning/test membership; and external-benchmark provenance. Release the list of the 1,000 held-out plasmids and all temporal or CRISPR-supported external sets used in the manuscript.

### Gene-product disambiguation

Provide replicon accession, masked feature identifier and coordinates, original PGAP product, candidate products, correct option, distractor-sampling seed, option order and split membership.

### Genome contig ordering

Provide chromosome accession, ordered feature identifiers, split points, shuffled order, ground-truth order, random seed and the cyclic-equivalence rule. Release the CAMI-287 accession/exclusion list.

### Gene essentiality

Provide organism and assembly accession, DEG identifier/version, feature identifier, genomic coordinates, product, essentiality label, evidence type, mapping status and gene-level/genome-level split assignments. Release the two independent *Streptococcus suis* mappings separately.

### Phenotype prediction

For each BacDive-derived table, document the BacDive download/query date, permitted redistribution status, raw-to-cleaned transformations, species-to-RefSeq mapping, excluded records, class conversion thresholds and split seeds. Access-controlled or redistribution-restricted source fields should be represented by a retrieval script and stable identifiers rather than copied without permission.

## Data integrity

Every released data archive should include:

- row counts before and after filtering;
- duplicate and leakage checks;
- SHA-256 checksums;
- a data dictionary;
- a machine-readable license/provenance manifest;
- scripts that regenerate derived inputs from the permitted source data.
