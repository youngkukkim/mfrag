# M-FRAG

M-FRAG aligns molecular and fragment representations in a shared embedding space and uses the resulting region score for property-aware fragment selection.

## Installation

```bash
conda create -n mfrag python=3.10
conda activate mfrag
pip install -r requirements.txt
```

## Data Preprocessing

```bash
python data_generate.py
```

## Training

```bash
python train_mfrag.py -t parp1
```

M-FRAG training supports docking targets, QED, SA, and MPO property targets.

## Generation

```bash
python run.py -t parp1 -v data/parp1.txt
```

Generation supports docking targets, QED, and SA.

## Evaluation

```bash
python eval.py results/file_name.csv -t parp1
```

Generated-molecule evaluation supports docking targets, QED, and SA.

## Paper Results

Machine-readable PMO/QED post hoc analysis outputs and representative molecule panels are available in `paper_results/pmo_qed/`.
The PMO/QED files include the aggregate decile analysis, representative molecule selections, and the PDF/PNG figure panels used in the Supporting Information.
Large generated streams, full region-score annotations, docking-score outputs, random-control indices, and trained checkpoints are not stored directly in Git because of file size.
These files can be regenerated from the released scripts; versioned release assets accompanying the manuscript will provide generated SMILES, target scores, QED/SA values, novelty annotations, seeds, and evaluation summaries.
