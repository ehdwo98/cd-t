# CD-T Experiments: What to Run, Why, and How

This repository accompanies the paper **“Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition (CD‑T)”** (ICLR 2025). The codebase is organized around a set of Jupyter notebooks under `notebooks/` that reproduce the paper-style circuit discovery and evaluation workflows.

This document explains:
- what each experiment is trying to establish (**why**),
- the high-level procedure (**how**),
- and which notebooks to run (and in what order).

---

## 1) Environment & prerequisites

### Install

```bash
pip install -r requirements.txt
```

### GPU
Most notebooks are significantly faster on a CUDA GPU (TransformerLens + decomposition batching).

### Common notebook pitfalls / version mismatches

- **`torchtyping` vs. PyTorch incompatibility**
  - Some environments throw: `RuntimeError: Cannot subclass _TensorBase directly` when importing `torchtyping`.
  - This repo now includes a defensive fallback in `im_utils/prompts.py` to allow the docstring notebooks to run even when `torchtyping` is incompatible (type hints become no-ops).

- **TransformerLens `generate()` API**
  - Some notebooks use `return_type='tensor'`, which is rejected by modern TransformerLens.
  - Use `return_type='tokens'` and then convert the last token to string.

- **IPython display import**
  - If you see: `ImportError: cannot import name 'display' from IPython.core.display`
  - Change to: `from IPython.display import display, HTML`

---

## 2) Recommended execution order (end-to-end)

If your goal is to reproduce the “main” circuit pipelines plus the paper’s faithfulness-style plots:

1. **Sanity checks of CD-T implementation**
   - `notebooks/correctness_tests/GPT_correctness_tests.ipynb`
   - `notebooks/correctness_tests/BERT_tests.ipynb`
   - (optional) `notebooks/correctness_tests/GPT_exploration.ipynb`

2. **IOI (Indirect Object Identification) circuit discovery**
   - `notebooks/IOI_automated_circuit_discovery.ipynb`
   - (optional visualization / deeper inspection) `notebooks/IOI_analysis_visualization.ipynb`

3. **Docstring completion circuit analysis**
   - `notebooks/docstring_task_analysis.ipynb`

4. **Greater-than task circuit analysis**
   - `notebooks/greater_than_task.ipynb`

5. **Faithfulness-style evaluation & plots**
   - `notebooks/faithfulness_eval.ipynb`

Notes:
- Some notebooks contain placeholder paths named `REDACTED`. You must replace these with real paths on your machine.
- `faithfulness_eval.ipynb` expects you to already have saved circuit/head-ranking artifacts from earlier runs (CD‑T outputs and, optionally, EAP outputs). The repo does **not** consistently auto-save those artifacts by default; you may need to add a small “save results” cell.

---

## 3) Experiment: Correctness / Sanity checks

### Why
CD‑T is a decomposition method. Before trusting circuit discovery results, you need to ensure the decomposition math is implemented correctly and is numerically stable. These notebooks act as regression tests and sanity checks.

### How (high-level)
- Load a model (GPT-like or BERT-like).
- Run decomposition functions (e.g., CD‑T propagation) on controlled inputs.
- Verify shapes, invariants, and that recombining relevant/irrelevant parts reproduces the original activations/logits.

### Notebooks
- `notebooks/correctness_tests/GPT_correctness_tests.ipynb`
- `notebooks/correctness_tests/BERT_tests.ipynb`
- `notebooks/correctness_tests/GPT_exploration.ipynb` (optional)

---

## 4) Experiment: IOI automated circuit discovery

### Why
IOI is a standard mechanistic interpretability benchmark. The goal is to automatically identify a small set of internal components (typically attention heads at specific positions) that explain most of the model’s performance on IOI—i.e., a “circuit”.

### How (high-level)
The typical flow inside the IOI notebooks is:
- Load `gpt2-small` via TransformerLens `HookedTransformer`.
- Build an IOI dataset and a corresponding “ABC” corruption dataset used for **mean ablation** baselines.
- Compute **mean attention head outputs** (often `attn.hook_z`) over the corruption dataset; this is what ablation patches in.
- Run CD‑T decomposition in batches:
  - **Step A**: decompose contributions **to the logits** (or to a logit-difference metric).
  - **Step B**: pick top “outlier” nodes and treat them as targets; decompose contributions **to those targets**; iterate.
- Collect candidate nodes into an initial circuit.
- Optionally apply a **greedy pruning heuristic**: remove nodes that improve the circuit score under the task metric.
- Evaluate the resulting circuit using mean-ablation hooks (keep circuit nodes; ablate the rest).

### Notebooks
- `notebooks/IOI_automated_circuit_discovery.ipynb`
  - Produces a circuit as a list of `Node(layer_idx, sequence_idx, attn_head_idx)`.
  - Does **not** reliably export “head rankings” to JSON by default.
- `notebooks/IOI_analysis_visualization.ipynb` (optional)
  - Used for deeper inspection/visualization; may require small import fixes depending on IPython version.

### Outputs you may want to save (for later `faithfulness_eval.ipynb`)
`faithfulness_eval.ipynb` needs a **head order** (ranking) for CD‑T on IOI. IOI discovery often works at node granularity (layer, position, head), so you typically need to aggregate node scores into a head-level score:
- A common aggregation is `max` or `mean` score over positions for a given `(layer, head)`.
- Save the resulting `[(f"{layer}.{head}", importance), ...]` list as JSON.

---

## 5) Experiment: Docstring task analysis (attn-only-4l)

### Why
Docstring completion is another standard circuit benchmark (especially used with TransformerLens toy models). The goal is to show CD‑T can recover meaningful circuits beyond IOI and beyond GPT‑2 small.

### How (high-level)
- Load `attn-only-4l` (TransformerLens).
- Generate synthetic docstring prompts (from `im_utils/prompts.py`).
- Filter to “good” prompts where the model predicts the correct next token (so we study the mechanism when the model is actually performing the task).
- Build a corruption dataset (e.g., `random_answer_doc`) and compute mean activations / patch values.
- Run CD‑T decomposition and/or targeted decomposition to identify relevant heads.
- Evaluate circuits using mean ablation: keep circuit heads (often all positions for those heads) and ablate others.

### Notebook
- `notebooks/docstring_task_analysis.ipynb`

### Known compatibility fixes
- If `torchtyping` is incompatible with your PyTorch build, the repo’s `im_utils/prompts.py` now falls back to no-op type hints.
- If `HookedTransformer.generate()` rejects `return_type='tensor'`, switch to `return_type='tokens'`.

---

## 6) Experiment: Greater-than task (GPT‑2 small)

### Why
The “Greater-than” benchmark tests whether GPT‑2 can compare numbers/years and assign higher probability mass to “greater” completions. The circuit discovery goal is the same: identify a small set of heads whose retention preserves task performance.

### How (high-level)
- Load `gpt2-small`.
- Build the dataset from `greater_than_task/`.
- Define the metric (probability mass on correct vs incorrect year tokens).
- Build a mean-ablation baseline by computing mean head outputs (`attn.hook_z`) over a reference set.
- Evaluate circuits by keeping selected heads and mean-ablating the rest.

### Notebook
- `notebooks/greater_than_task.ipynb`

---

## 7) Experiment: Faithfulness evaluation (IOI / Docstring / Greater-than)

### Why
Circuit discovery is only useful if the discovered subgraph is **faithful**: it should preserve the original model’s behavior on the task. The paper emphasizes:
- **Faithfulness vs. circuit size**: how quickly performance is recovered as more components are added.
- **Better-than-random tests**: the discovered circuit should be more faithful than random circuits of comparable size.
- Comparisons to baselines like **EAP** (when available).

### How (high-level)
`faithfulness_eval.ipynb` performs:

1) **Define a clean score and a null score**
- `MODEL_CLEAN`: the model’s task metric without ablation.
- `MODEL_NULL`: the metric when essentially everything is ablated (mean-ablation baseline).

2) **Define faithfulness**
For a circuit with score `circuit_score`:

\[
\text{faithfulness}=\frac{\text{circuit\_score}-\text{MODEL\_NULL}}{\text{MODEL\_CLEAN}-\text{MODEL\_NULL}}
\]

3) **Faithfulness vs. node/head count**
- Add components (heads) one by one according to an ordering:
  - random order,
  - CD‑T ranking order,
  - EAP ranking order (if provided),
  and plot the resulting faithfulness curve.

4) **Hypothesis testing vs random**
- For a fixed “our circuit” faithfulness value, sample random circuits of different sizes (10%…100% of model heads).
- Estimate the probability that our circuit is more faithful than random circuits of that size.

### Notebook
- `notebooks/faithfulness_eval.ipynb`

### Why it often errors “as-is”
This notebook uses multiple `REDACTED` placeholders and assumes specific file formats:
- It may assume CD‑T head rankings are stored as an iterable of `(head, importance)` pairs.
- In practice, you may have a JSON dict like `{index: score}` or a different keying scheme.
- For EAP, it may assume a specific key like `'0.077'` exists; your EAP JSON might store different keys or a different structure.

To run it successfully, you must:
- replace `REDACTED` with real paths, and
- ensure the loaded artifacts match the notebook’s expected structure (or add small adapter code).

---

## 8) Where each notebook pulls utilities from

- Core CD‑T decomposition utilities: `pyfunctions/`
  - `pyfunctions/cdt_basic.py`
  - `pyfunctions/cdt_source_to_target.py`
  - `pyfunctions/cdt_from_source_nodes.py`
  - `pyfunctions/faithfulness_ablations.py` (mean-ablation hooks, scoring helpers)
  - `pyfunctions/wrappers.py` (`Node`, `AblationSet`, etc.)

- Greater-than dataset/tools: `greater_than_task/`

- Docstring prompt utilities: `im_utils/`

---

## 9) Practical advice

- Run notebooks top-to-bottom **in a fresh kernel**. Many cells depend on global state (e.g., `pos_labels`, `seq_len`, cached activations).
- For any notebook that contains `REDACTED`, treat it as: “you must provide a real path”.
- If you want a fully reproducible pipeline, add small “export results” cells at the end of discovery notebooks so that `faithfulness_eval.ipynb` can load them without manual copy/paste.


