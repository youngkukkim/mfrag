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
