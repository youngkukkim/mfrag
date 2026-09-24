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
python train_mfrag.py -t parp1 --model_arch shared --train_mode joint
```

To use the epoch, batch-size, and early-stopping settings listed in the
Supporting Information:

```bash
python train_mfrag.py -t parp1 --model_arch shared --train_mode joint \
  --epochs 50 --batch_size 512 --early_stop_patience 5
```

Use the respective docking target, `qed`, or `sa` with `-t` to train each
property-specific model. Generic training defaults remain unchanged.

The paper configuration uses one shared graph encoder for molecule and fragment
inputs. The property-prediction and molecule--fragment alignment objectives
jointly optimize this encoder. `shared` and `joint` are the defaults; they are
shown explicitly above to make the reported configuration unambiguous.

M-FRAG training supports docking targets, QED, SA, and MPO property targets.

## Generation

```bash
python run.py -t parp1 -v data/parp1.txt
```

Generation supports docking targets, QED, and SA.

`run.py` retains the single-property selection rule. Each run writes its resolved
options to
`results/*_settings.json` alongside the generated-molecule records.

## Docking Benchmark

`run_benchmark.py` extends the single-property rule using separately trained
docking, QED, and normalized-SA M-FRAG spaces.

Prepare the QED/SA reference embeddings before generation:

```bash
python prepare_benchmark_references.py --device cpu
python run_benchmark.py -t parp1 -v data/my2_parp1.txt -g 0 -s 0
```

The preparation script uses the fixed 8,000 ZINC training-row indices in
`data/benchmark_reference_indices.json` and stores references under
`mfrag_region_cache/benchmark/`. It uses only training rows and checks for
overlap with the validation indices.

Docking targets are `parp1`, `fa7`, `5ht1b`, `braf`, and `jak2`, with respective
`data/my2_<target>.txt` vocabularies. The benchmark uses raw ECFP fragment
descriptors and 40-neighbor region scores. Online encoder fine-tuning remains
off unless explicitly enabled. Generation budget and seed are set with
`--num_mols` and `--seed`. All other shared generation options remain available.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover fragment pooling safeguards, selection modes, sampling options,
benchmark region conditions, GA parent eligibility, and benchmark evaluation
criteria without running docking.

## Evaluation

For the docking benchmark and fragment-selection/noise comparisons:

```bash
python eval_benchmark.py results/file_name.csv -t parp1 --num_mols 3000
```

The evaluator uses the first `--num_mols` outputs, including warmup and both
SAC and GA sources. Novel Hit Ratio counts unique generated SMILES satisfying
actual `QED > 0.5`, normalized `SA > 5/9`, maximum ECFP4 Tanimoto similarity to
the ZINC training molecules `< 0.4`, and the target docking threshold. Its
denominator is the full output count before filtering. No predictor screening
is applied. SMILES deduplication follows the recorded generation strings;
canonical SMILES are used for the novelty calculation.

Novel Top-5% DS averages the best 5% of the full output count among unique
novel molecules satisfying the QED/SA criteria (150 molecules for 3,000
outputs). The input CSV stores the higher-is-better `-DS`; the evaluator
reports negative DS in kcal/mol, so more negative is better. If fewer molecules
qualify, it reports the available count and warns rather than silently changing
the requested count. Internal diversity is mean pairwise ECFP4 Tanimoto distance
over all unique valid outputs, before the QED/SA/novelty criteria. Fingerprints
use radius 2 and 1,024 bits. `--output metrics.csv` also saves the metrics.

The evaluator uses `data/zinc250k_novelty.pt`, building it from the ZINC CSV
and validation-index file if absent. This first-time preparation and the
similarity calculations run on CPU; they do not require new docking or generation.

The existing `eval.py` command is a separate, unqualified docking summary: its
hit ratio and top-5% score do not apply the joint QED/SA/novelty criteria and
are not the benchmark NHR and Novel Top-5% DS. `eval_generation_batch.py` retains
the legacy summaries, including QED/SA-only objectives; use `eval_benchmark.py`
for the joint docking-benchmark metrics.

## Paper Results

Machine-readable PMO/QED post hoc analysis outputs and representative molecule panels are available in `paper_results/pmo_qed/`.
The PMO/QED files include the aggregate decile analysis, representative molecule selections, and the PDF/PNG figure panels used in the Supporting Information.
Large generated streams, full region-score annotations, docking-score outputs, and random-control indices are not stored directly in Git because of file size.
