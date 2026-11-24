# DistilClean: A Data Cleaning Framework Based on LLM Knowledge Distillation

DistilClean combines LLM knowledge distillation with data constraints to detect and correct cell-level errors in tabular data. It uses rules and contextual evidence to guide teacher-model annotation, then trains lightweight student models for error detection and repair-candidate ranking.

The shared workflow supports **Hospital, Flights, Beers, Rayyan, Adult, and Soccer** through an external experiment configuration.

## Installation

Use Python **3.9+**:

```bash
git clone https://github.com/JackAvril/DistilClean.git
cd DistilClean
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PY="$PWD/.venv/bin/python"
```

## Prepare the inputs

Download the dataset package from [Google Drive](https://drive.google.com/file/d/14vQuk0lV54QrfTzY8tvQAjgi6Vu4VV0y/view?usp=sharing). It contains Dirty and Clean CSVs for all six datasets. Extract it into a directory containing the six dataset folders.

Provide an external rule-pool JSON and a complete experiment configuration matching [configs/config.schema.json](configs/config.schema.json). The schema lists required settings and their allowed types and ranges. Set all training, inference, annotation and runtime parameters explicitly in your configuration.

```bash
DATA_ROOT=/absolute/path/to/datasets
DATASET=hospital
CONFIG=/absolute/path/to/experiment.json
RULE_POOL=/absolute/path/to/hospital_rule_pool.json
RUN_DIR="runs/${DATASET}-$(date -u +%Y%m%dT%H%M%SZ)"
```

The configuration contains:

| Section | Contents |
|---|---|
| `dataset`, `dirty` | Dataset identifier and Dirty CSV path relative to `DATA_ROOT` |
| `missing`, `fields`, `rules` | Missing-value semantics, column definitions and declared constraints |
| `detection` | Sampling, propagation, feedback, thresholds, student and verifier settings |
| `repair` | Candidate generation, preference sampling, ranking and repair decisions |
| `teacher` | Request timeout and retry policy |
| `runtime` | Random seed, device (`cpu` or `cuda`) and thread count |

Column definitions may specify `kind`, `required`, `target`, `pattern`, `bounds`, `domain`, `aliases` and missing-value policies. Dataset-specific context and repair settings can be supplied in the corresponding sections. Missing required settings cause an error; configurations are not merged with experiment presets.

## Configure the LLM

Use an OpenAI-compatible Chat Completions API supporting JSON-object responses. The teacher labels candidate error cells during detection and candidate-pair preferences during correction.

```bash
MODEL=your-model-name
ENDPOINT=https://your-provider.example/v1
CACHE_DIR="$PWD/cache/$DATASET"
read -r -p "Maximum LLM requests: " MAX_REQUESTS
read -r -s -p "LLM API key: " CLEANING_LLM_API_KEY
export CLEANING_LLM_API_KEY
printf '\n'

TEACHER_ARGS=(
  --teacher live --model "$MODEL" --endpoint "$ENDPOINT"
  --max-requests "$MAX_REQUESTS" --cache "$CACHE_DIR"
)
```

Choose a positive request cap for your experiment. It includes retries and is shared by detection and correction in an `all` invocation. Responses are cached automatically. An existing `OPENAI_API_KEY` is also supported.

## Run data cleaning

Set shared arguments:

```bash
COMMON_ARGS=(
  --dataset "$DATASET" --data-root "$DATA_ROOT" --config "$CONFIG"
  --rule-pool "$RULE_POOL" --legacy-rule-semantics archived
)
```

### End-to-end

```bash
"$PY" -B -m cleaning all "${COMMON_ARGS[@]}" "${TEACHER_ARGS[@]}" \
  --output "$RUN_DIR/all"
```

### Detection only

```bash
"$PY" -B -m cleaning detection "${COMMON_ARGS[@]}" "${TEACHER_ARGS[@]}" \
  --output "$RUN_DIR/detection"
```

### Correction only

```bash
"$PY" -B -m cleaning correction "${COMMON_ARGS[@]}" "${TEACHER_ARGS[@]}" \
  --detections "$RUN_DIR/detection" --output "$RUN_DIR/correction"
```

Use the same input data, rules, configuration and code version in both stages. Each run requires a new output directory. Cell coordinates use zero-based data-row positions and CSV column names.

### Run with a complete cache

For a repeat run with a complete response cache:

```bash
"$PY" -B -m cleaning all "${COMMON_ARGS[@]}" \
  --teacher cache --model "$MODEL" --endpoint "$ENDPOINT" \
  --cache "$CACHE_DIR" --output "$RUN_DIR/cached"
```

Keep the model, endpoint, inputs and configuration unchanged. A missing cached response stops the run.

## Outputs

| File | Contents |
|---|---|
| `detections.csv` | Candidate cells, prediction scores and detection decisions |
| `repairs.csv` | Proposed and final repair values, acceptance decisions and reasons |
| `repaired.csv` | Cleaned table |
| `run.json` | Configuration, input/code fingerprints, environment and execution status |
| `*_training.json` | Training and iteration records |
| `detector.pt`, `ranker.pt` | Student models, when fitted |

## Project layout

```text
cleaning/          Shared detection, correction, evidence and model modules
configs/           External configuration schema
```

```bash
"$PY" -B -m cleaning --help
```
