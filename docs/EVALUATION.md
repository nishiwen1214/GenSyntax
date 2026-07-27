# Evaluation definitions

This document records the deterministic evaluation rules used by the four
primary GenSyntax tasks. Runnable inference and evaluation commands remain in
the repository [README](../README.md).

## Shared conventions

- Reference files and prediction files must contain the same number of
  records.
- Prediction files contain one model response per line.
- Empty, unresolved or invalid predictions remain in the applicable
  denominator and are counted as errors.
- Tasks 1–4 use 100 sample-level percentile-bootstrap replicates, a two-sided
  90% confidence interval and random seed 42 by default.
- The CSV files contain compact result tables. The JSON files also record the
  input paths and evaluation settings.

## Task 1: plasmid host identification

The evaluator parses genus, species and optional strain from either plain text
or a bracketed answer. `Data/genus_taxonomy.csv` supplies the corresponding
class, order and family and must contain `genus,class,order,family` columns.

Accuracy at class, order, family, genus and species uses every test record as
the denominator. Strain accuracy uses only records whose reference answer
contains strain information. Missing or unparseable predictions remain in the
relevant denominator and count as incorrect.

Outputs:

- rank-specific accuracy;
- correct, total and unresolved counts;
- percentile-bootstrap lower and upper confidence bounds.

## Task 2: gene-product disambiguation

The four-option and eight-option test sets are evaluated independently.
Official released Task 2 records expose option markers that permit
`--num-options auto`; `--num-options 4` and `--num-options 8` select the
settings explicitly.

The parser accepts bracketed labels, numbered labels, explicit `Answer: X`
text and standalone option letters. Each reference must resolve to exactly one
valid answer. Empty, ambiguous, unparseable or out-of-range predictions remain
in the denominator and count as incorrect.

Outputs:

- overall accuracy;
- correct, total, resolved and unresolved counts;
- percentile-bootstrap lower and upper confidence bounds.

## Task 3: circular contig ordering

Tasks with three, four and five contigs are evaluated separately. A prediction
is correct when it is a cyclic rotation of the reference order. Reversed
orders are not accepted for manuscript evaluation; `--allow-reverse` is an
optional exploratory setting and must not be used to reproduce the reported
results.

A resolved prediction must contain every identifier from `Contig 1` through
`Contig N` exactly once. Missing, duplicated, unexpected or additional
identifiers remain in the denominator and count as errors.

Outputs:

- circular-order accuracy;
- correct, total, resolved and unresolved counts;
- percentile-bootstrap lower and upper confidence bounds;
- an optional error CSV containing record-level failures.

## Task 4: gene essentiality

The evaluator parses `non-essential` before `essential` because the former
contains the latter as a substring. It reports:

- accuracy;
- essential-class precision, recall and F1;
- non-essential-class precision, recall and F1;
- macro precision, recall and F1;
- percentile-bootstrap confidence bounds for every metric.

Empty or invalid responses remain in the accuracy denominator and count as
false negatives for the corresponding reference class. Because the released
test set is class-imbalanced, reports must identify whether precision, recall
and F1 refer to the essential class, non-essential class or macro average.
These quantities are not interchangeable.

## Changing statistical settings

The evaluators expose the same statistical arguments:

```text
--bootstrap-replicates 100
--confidence-level 0.90
--seed 42
```

Any deviation from these defaults should be reported together with the code
commit, model revision and dataset revision.
