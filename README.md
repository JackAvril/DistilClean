# DistilClean: A Data Cleaning Framework Based on LLM Knowledge Distillation

DistilClean combines LLM knowledge distillation with data constraints to automatically detect and correct cell-level errors in tabular data. It uses rules and contextual evidence to guide teacher-model annotation, then trains lightweight student models for error detection and repair-candidate ranking.

DistilClean provides a unified workflow for cell-level error detection and correction on **Hospital, Flights, Beers, Rayyan, Adult, and Soccer**. Dataset configurations select the input tables, rules and experiment parameters.

## Installation

Use Python **3.9+**. Run the following from the directory containing `cleaning/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PY="$PWD/.venv/bin/python"
```

CPU is supported. To use a compatible CUDA installation, add `--device cuda` to a run command.

## Prepare the inputs

Download the dataset package from [Google Drive](https://drive.google.com/file/d/14vQuk0lV54QrfTzY8tvQAjgi6Vu4VV0y/view). It contains Dirty and Clean CSVs for all six datasets. Extract the archive and set `DATA_ROOT` to the directory containing the six dataset folders.

Set the project path, select a dataset and provide its rule pool. Rule pools are prepared separately.

```bash
DATA_ROOT=/absolute/path/to/Splittree
DATASET=hospital
RULE_POOL=/absolute/path/to/hospital_rule_pool.json
RUN_DIR="runs/${DATASET}-$(date -u +%Y%m%dT%H%M%SZ)"
```

Dirty-table paths are relative to `DATA_ROOT`:

| Dataset | Dirty CSV |
|---|---|
| `hospital` | `hospital/hospital_dirty_no_address23.csv` |
| `flights` | `flights/flights_dirty.csv` |
| `beers` | `beers/beers_dirty.csv` |
| `rayyan` | `rayyan/rayyan_dirty.csv` |
| `adult` | `adult/adult_dirty.csv` |
| `soccer` | `soccer/soccer_dirty.csv` |

For another dataset, update `DATASET` and `RULE_POOL`. Table paths and field settings are defined in `configs/<dataset>.json`. Use a new output directory for each run.

## Configure the LLM

For a first run, configure a teacher model through an OpenAI-compatible Chat Completions API with JSON-object responses. The LLM labels candidate error cells during detection and candidate-pair preferences during correction. The framework then trains local student models for detection and repair ranking.

Replace the model name and API base URL with those supplied by your provider, then enter your API key:

```bash
MODEL=your-model-name
ENDPOINT=https://your-provider.example/v1
MAX_REQUESTS=2500
CACHE_DIR="$PWD/cache/$DATASET"
read -r -s -p "LLM API key: " CLEANING_LLM_API_KEY
export CLEANING_LLM_API_KEY
printf '\n'

TEACHER_ARGS=(
  --teacher live --model "$MODEL" --endpoint "$ENDPOINT"
  --max-requests "$MAX_REQUESTS" --cache "$CACHE_DIR"
)
```

`MAX_REQUESTS` is an example request cap; set it for your experiment. It includes retries and is shared by detection and correction in an `all` invocation. Annotation budgets and feedback rounds are set in the configuration files. Responses are cached automatically. An existing `OPENAI_API_KEY` is also supported.

## Run data cleaning

Set the shared arguments once:

```bash
COMMON_ARGS=(
  --dataset "$DATASET" --data-root "$DATA_ROOT"
  --rule-pool "$RULE_POOL" --legacy-rule-semantics archived
  --seed 42 --threads 1
)
```

### End-to-end

```bash
"$PY" -B -m cleaning all "${COMMON_ARGS[@]}" "${TEACHER_ARGS[@]}" \
  --output "$RUN_DIR/all"
```

The command performs LLM annotation, student training, detection and correction, then writes the cleaned table.

### Detection only

```bash
"$PY" -B -m cleaning detection "${COMMON_ARGS[@]}" "${TEACHER_ARGS[@]}" \
  --output "$RUN_DIR/detection"
```

### Correction only

```bash
"$PY" -B -m cleaning correction "${COMMON_ARGS[@]}" "${TEACHER_ARGS[@]}" \
  --detections "$RUN_DIR/detection" \
  --output "$RUN_DIR/correction"
```

Use the same Dirty table, rule pool and settings in both stages. Cell coordinates use zero-based data-row positions and CSV column names.

### Run with a complete cache

For a repeat run with a complete response cache, use `--teacher cache` to run without LLM requests. Keep the model, endpoint, inputs and configuration unchanged; a missing response stops the run.

```bash
"$PY" -B -m cleaning all "${COMMON_ARGS[@]}" \
  --teacher cache --model "$MODEL" --endpoint "$ENDPOINT" \
  --cache "$CACHE_DIR" --output "$RUN_DIR/cached"
```

## Outputs

| File | Contents |
|---|---|
| `detections.csv` | Candidate cells, prediction scores and detection decisions |
| `repairs.csv` | Proposed and final repair values, acceptance decisions and reasons |
| `repaired.csv` | Cleaned table |
| `run.json` | Configuration, environment and execution status |
| `*_training.json` | Training and iteration records |
| `detector.pt`, `ranker.pt` | Student models, when fitted |

## Configuration and tools

```text
cleaning/          Shared detection, correction, evidence and model modules
configs/          Default parameters and six dataset configurations
tools/            Rule import and intermediate-output comparison utilities
tests/            Tests
runs/             Run outputs
```

Edit `configs/defaults.json` for shared defaults or `configs/<dataset>.json` for dataset settings. The main workflow uses the supplied rule pool.

```bash
"$PY" -B -m cleaning --help
"$PY" -B tools/stage_diff.py --help
"$PY" -B -m unittest discover -s tests -v
```

