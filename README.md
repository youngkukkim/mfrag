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

Regression training supports `--regression_loss huber|mse`. The default is
Huber loss with `--delta 1.0`, retaining the existing training setting. To use
MSE, add `--regression_loss mse`; `--delta` does not affect MSE.
For optional generation-time fine-tuning, both `run.py` and `run_benchmark.py`
accept `--mfrag_finetune_loss huber|mse` and `--mfrag_finetune_delta` (defaults:
`huber` and `1.0`). These options take effect only with
`--enable_mfrag_finetune` and do not change existing checkpoint weights at load
time. Classification training continues to use binary cross-entropy with logits.

The paper configuration uses one shared graph encoder for molecule and fragment
inputs. The property-prediction and molecule--fragment alignment objectives
jointly optimize this encoder. `shared` and `joint` are the defaults; they are
shown explicitly above to make the reported configuration unambiguous.

Training saves regression checkpoints to `ckpt/reg/<property>/`, and generation
loads `ckpt/reg/<property>/best.pt` by default. For a different checkpoint root,
set `--out_root` during training and `--mfrag_root` during generation to the same
directory. Checkpoint files are not included in this repository.

M-FRAG training supports docking targets, QED, SA, and MPO property targets.

## Generation

```bash
python run.py -t parp1 -v data/parp1.txt
```

Generation supports docking targets, QED, and SA.

`run.py` retains the single-property selection rule. Optional controls are
`--fragment_selection_mode hybrid|sac_only|mfrag_only` and
`--gumbel_noise_scale` (default `0.001`; `1.0` uses unscaled noise).
These selection modes apply after the common random warmup:

- `hybrid` uses region-score improvements, with SAC fragment selection when no
  eligible candidate is available.
- `sac_only` uses SAC fragment selection. `--disable_region_guidance` remains an
  alias for this mode.
- `mfrag_only` uses the same region rule but no SAC fragment fallback. If no
  eligible candidate exists, it uses region-score changes among viable
  candidates, including non-improving ones. If encoding fails, viable candidates
  receive uniform weights.

SAC attachment-site selection and GA remain active in all three modes. The
noise multiplier applies to all three action stages. Replay minibatches retain
the existing SAC learning path; region guidance is applied to online selection
of one molecule at a time. Each run writes its resolved options to
`results/*_settings.json` alongside the generated-molecule records.

## Docking Benchmark

`run_benchmark.py` extends the single-property rule using separately trained
docking, QED, and normalized-SA M-FRAG spaces. It keeps candidates whose mixed
embeddings improve the docking region score while meeting QED and SA region
thresholds of `0.75` and `0.70`. Eligible candidates are weighted by their
docking region-score improvements; Hybrid uses SAC fallback when none qualifies.
These region thresholds do not guarantee the actual properties of the assembled
molecule.

The same benchmark configuration admits completed molecules with actual
`QED > 0.5` and normalized `SA > 5/9` to the GA parent pool. It retains up to
100 parents ranked by the docking objective and samples parents in proportion
to that objective. QED and SA are already returned by the molecule evaluation
step. Parent eligibility does not remove generated outputs from the result
file or generation count. The SAC reward remains the docking objective.

Prepare the QED/SA reference embeddings once for the checkpoints being used:

```bash
python prepare_benchmark_references.py --device cpu
python run_benchmark.py -t parp1 -v data/my2_parp1.txt -g 0 -s 0
```

The preparation script uses the fixed 8,000 ZINC training-row indices in
`data/benchmark_reference_indices.json` and stores references under
`mfrag_region_cache/benchmark/`. It uses only training rows and checks for
overlap with the validation indices. Reference files carry their checkpoint
hashes; regenerate them when changing the QED/SA checkpoints. Models are read
from `ckpt/reg/<property>/best.pt` by default, or from the explicit
`--qed_mfrag_ckpt` and `--sa_mfrag_ckpt` paths supplied to both commands.

Docking targets are `parp1`, `fa7`, `5ht1b`, `braf`, and `jak2`, with respective
`data/my2_<target>.txt` vocabularies. The benchmark uses raw ECFP fragment
descriptors and 40-neighbor region scores. Online encoder fine-tuning remains
off unless explicitly enabled. Generation budget and seed are set with
`--num_mols` and `--seed`. All other shared generation options remain available.

The selection and noise options also work with `run_benchmark.py`. In
`sac_only`, QED/SA region guidance is disabled, while the same GA parent
eligibility rule remains active. In `mfrag_only`, the no-eligible-candidate rule
relaxes the QED/SA region conditions as well as positive docking improvement.

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
Large generated streams, full region-score annotations, docking-score outputs, random-control indices, and trained checkpoints are not stored directly in Git because of file size.
