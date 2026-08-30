# DeepD Inference API (`apiexample`)

The `apiexample` directory contains the **inference gateway client** for DeepD.
All DeepD features are served through the online inference gateway and driven
with `cli.py` in this directory.

```text
apiexample/
├── cli.py                      # gateway client: submit → poll → download
├── species.py                  # scientific-name → internal species-id lookup
├── data/species_name_to_id.json
└── etc/submit_job.json         # default submit request body template
```

## Configuration

Configure your API key with the `INFERENCE_API_TOKEN` environment variable, or
place it in the git-ignored `API-Key.txt` at the repository root. You may also
pass the token directly with `--token`.

```bash
export INFERENCE_API_TOKEN="your_api_token"
```

Each invocation submits an inference job, polls the gateway until it completes,
downloads the result JSON (`{task_id}.json`) into `--output-dir` (default
`./results`), and prints a task-type summary. To fetch an existing job by its
task id instead of submitting a new one, pass `--task-id`.

The examples below use the `apiexample/` directory as the working directory:

```bash
cd apiexample
```

(from the repository root you can run the same commands as
`python apiexample/cli.py ...` instead).

## Task types

| Task type   | Description                                                    |
|-------------|----------------------------------------------------------------|
| `embedding` | Extract a nucleotide-resolution sequence representation.       |
| `logits`    | Score a sequence: per-position logits, mean loss, and perplexity. |
| `generate`  | Generate a DNA sequence conditioned on a prompt.               |

## Embeddings

Extract nucleotide-resolution representations of a sequence for downstream
prediction or similarity tasks:

```bash
python cli.py --prompt ATGCATGC --task-type embedding \
  --species "Homo sapiens" --output-dir results
```

The result JSON holds the embedding `values` and its `shape`.

## Log-likelihood scoring and perplexity

Score how likely a sequence is under the model - the basis for variant-effect
and zero-shot prediction:

```bash
python cli.py --prompt ACGTACGT --task-type logits \
  --species "Homo sapiens" --output-dir results
```

The result JSON holds the raw per-position `logits.values` (L x V), the mean
negative log-likelihood (`loss`), and the sequence perplexity (`ppl`):

```python
import json

with open("results/<task_id>.json") as f:
    result = json.load(f)["result"]
logits = result["logits"]["values"]
loss, ppl = result["loss"], result["ppl"]
```

## Sequence generation

Generate a controllable DNA sequence from a prompt:

```bash
python cli.py --prompt ATG --task-type generate \
  --species "Caenorhabditis elegans" \
  --max-tokens 100 \
  --output-dir results
```

## Fetch an existing job by task id

Skip submission and only poll + download the result of a previously submitted
job:

```bash
python cli.py --task-id <task_id> --output-dir results
```

## Parameter reference

**Job submission**

| Parameter | Description |
|-----------|-------------|
| `--prompt <seq>` | DNA prompt sequence (max 9997 characters). |
| `--prompt-file <path>` | Read the prompt from a file instead of the command line (same length limit). |
| `--task-type <type>` | Inference task type: `embedding`, `generate`, or `logits` (required when submitting a new job). |
| `--species <name>` | Species scientific name (e.g. `"Homo sapiens"`); resolved by the client to the internal id the gateway expects. If omitted, the job runs without a specific species. Unknown names fail before submit (exit `2`). |
| `--request-id <id>` | Optional request id. |
| `--token <token>` | API token for the `Authorization: Bearer` header. |

**Generation control**

| Parameter | Description |
|-----------|-------------|
| `--max-tokens <n>` | Maximum number of tokens to generate. |
| `--min-tokens <n>` | Minimum number of tokens to generate. |
| `--top-k <n>` | Top-k sampling. |
| `--top-p <p>` | Nucleus (top-p) sampling. |
| `--min-p <p>` | Min-p sampling threshold. |
| `--temperature <t>` | Sampling temperature (default: 1.0). |
| `--repetition-penalty <p>` | Penalty for repeated tokens (>= 1.0; > 1.0 discourages repeats). |
| `--presence-penalty <p>` | Presence penalty (OpenAI-style). |
| `--frequency-penalty <p>` | Frequency penalty (OpenAI-style). |
| `--generate-params-json <json>` | Raw generate parameters JSON merged into `parameters.generate_params_json`; the flat flags above take precedence. |

**Gateway and polling**

| Parameter | Description |
|-----------|-------------|
| `--gateway-url <url>` | Inference gateway URL (default: `https://ideal.biomap.com/api/online/inference`). |
| `--submit-file <path>` | Submit request body template (`modelname`, `namespace`, `parameters`, ...). |
| `--output-dir <dir>` | Directory where the downloaded `{task_id}.json` is saved (default: `./results`). |
| `--task-id <id>` | Fetch an existing job: skip submit, poll, and download its result. |
| `--poll-interval <s>` | Polling interval in seconds (default: 5). |
| `--timeout <s>` | Maximum time to wait for job completion (default: 3600). |
| `--http-timeout <s>` | Per-request HTTP timeout in seconds (default: 120). |
| `--save-submit-json <path>` | Save the submit response JSON to a file. |
| `--save-query-json <path>` | Save the last poll response JSON to a file. |
| `--success-status <list>` | Comma-separated terminal success statuses. |
| `--failure-status <list>` | Comma-separated terminal failure statuses. |

## Exit codes

`0` success · `1` other errors · `2` unknown species name