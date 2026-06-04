# PMO/QED Post Hoc Analysis Results

This directory contains machine-readable outputs used for the PMO/QED post hoc analysis of f-RAG-generated molecules with M-FRAG region scores.

## Files

- `PMO_QED_mfrag_decile_analysis_online_region_3_aggregate.csv`
  - Aggregate decile analysis using the final cumulative fragment-region score, `online_region_3`.
  - Reports mean oracle score, median oracle score, top-score hit rate, enrichment, and mean molecule count for each decile.
- `PMO_QED_mfrag_example_molecules.csv`
  - Representative low-, middle-, and high-region-score molecules used for the qualitative SI figure.
  - The `type` column records f-RAG provenance metadata from the original run and is not used as a visual label in the manuscript figure.
- `figure/pmo_qed_mfrag_examples_part1.png`
  - Representative molecule panel for amlodipine, fexofenadine, osimertinib, and perindopril.
- `figure/pmo_qed_mfrag_examples_part2.png`
  - Representative molecule panel for ranolazine, sitagliptin, zaleplon, and QED.

Large generated streams, checkpoints, and full region-score annotation files are excluded from the Git repository because of file size. They can be regenerated from the scripts in this repository or distributed separately as archival release assets.
