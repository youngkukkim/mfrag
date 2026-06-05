# PMO/QED Post Hoc Analysis Results

This directory contains machine-readable outputs used for the PMO/QED post hoc analysis of f-RAG-generated molecules with M-FRAG region scores.

## Files

- `PMO_QED_mfrag_decile_analysis_online_region_3_aggregate.csv`
  - Aggregate decile analysis using the final cumulative fragment-region score, `online_region_3`.
  - Reports mean oracle score, median oracle score, top-score hit rate, enrichment, and mean molecule count for each decile.
- `PMO_QED_mfrag_example_molecules.csv`
  - Representative low-, middle-, and high-region-score molecules used for the qualitative SI figure.
  - The `type` column records f-RAG provenance metadata from the original run and is not used as a visual label in the manuscript figure.
- `PMO_QED_mfrag_fragment_trajectory_ablation_s2_runq09.csv`
  - Per-oracle AUC top-10, top-10, and top-100 values for the PMO/QED fragment-region control analysis using the running \(q=0.9\) replay setting.
  - Includes the full f-RAG stream, score-only, trajectory-only, score-plus-trajectory, and same-count random-control rows.
- `PMO_QED_mfrag_fragment_trajectory_ablation_s2_w1000_q09.csv`
  - Warm-up variant of the fragment-region control analysis where the first 1000 generated molecules are retained before applying the running \(q=0.9\) replay filter.
- `figure/pmo_qed_mfrag_examples_part1.pdf`
  - Vector representative molecule panel for amlodipine, fexofenadine, osimertinib, and perindopril.
- `figure/pmo_qed_mfrag_examples_part2.pdf`
  - Vector representative molecule panel for ranolazine, sitagliptin, zaleplon, and QED.
- `figure/pmo_qed_mfrag_examples_part1.png`
  - PNG preview of the corresponding vector PDF panel.
- `figure/pmo_qed_mfrag_examples_part2.png`
  - PNG preview of the corresponding vector PDF panel.

Large generated streams, checkpoints, and full region-score annotation files are excluded from the Git repository because of file size. They can be regenerated from the scripts in this repository or distributed separately as archival release assets.
