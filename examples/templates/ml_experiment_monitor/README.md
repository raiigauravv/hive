# ML Experiment Monitor Agent

An agent template for monitoring [Weights & Biases](https://wandb.ai) ML experiments. Fetches runs and metrics via the W&B GraphQL API, analyzes for regressions, and delivers a rich HTML report with insights and recommendations.

## What the Agent Does

1. **Intake** — Asks you for your W&B entity and project name, and confirms what to monitor (latest runs, metric trends, regression detection, artifact tracking).
2. **Fetch** — Pulls run metadata, metrics history, summary metrics, and artifacts from W&B via the `wandb_tool` integration.
3. **Analysis** — Compares runs, detects metric regressions (loss going up, accuracy going down, overfitting signals), identifies the best-performing run, and surfaces key insights.
4. **Report** — Writes a multi-section HTML report with an experiment summary table, metric trends, regression alerts, best-run highlight, and actionable recommendations — then serves it directly in your browser.

## Flow Diagram

```
intake → fetch → analysis → report
  ↑                            |
  |    (new_project)           |
  +----------------------------+
  |    (dig_deeper)            |
  +--------→ fetch ←-----------+
```

- **intake → fetch**: On successful intake, proceed to data fetching.
- **fetch → analysis**: On successful data fetch, proceed to analysis.
- **analysis → report**: On successful analysis, generate and deliver the report.
- **report → intake**: If the user wants to monitor a *new project*, restart from intake.
- **report → fetch**: If the user wants to *dig deeper* into the same project, re-fetch with updated focus.

## Prerequisites

- A [Weights & Biases](https://wandb.ai) account with at least one project containing runs.
- Your W&B API key set as an environment variable:

```bash
export WANDB_API_KEY=your_api_key_here
```

Get your API key at: https://wandb.ai/authorize

## How to Run

**Interactive TUI (recommended):**
```bash
cd hive
uv run python -m examples.templates.ml_experiment_monitor tui
```

**CLI run (single project):**
```bash
uv run python -m examples.templates.ml_experiment_monitor run \
  --entity my-org \
  --project my-project
```

**Validate agent structure:**
```bash
uv run python -m examples.templates.ml_experiment_monitor validate
```

**Show agent info:**
```bash
uv run python -m examples.templates.ml_experiment_monitor info
```

**Interactive shell:**
```bash
uv run python -m examples.templates.ml_experiment_monitor shell
```

## Example Inputs

When prompted, provide:

| Field | Example |
|---|---|
| W&B Entity | `my-username` or `my-org` |
| W&B Project | `image-classification` |
| Monitor scope | `latest runs, metric trends, regression detection` |

The agent will then:
- List all runs in the project (up to 25)
- Fetch metrics (`loss`, `accuracy`, `val_loss`, `val_accuracy`) for the most recent 3–5 runs
- Detect regressions across consecutive runs
- Generate an HTML report saved locally and opened in your browser

## Output

The agent produces an HTML report (`ml_report.html`) containing:

- **Experiment Summary Table** — run IDs, states, timestamps, key final metrics
- **Metric Trends** — per-run metric direction (↑ improving, ↓ degrading, → stable)
- **Regression Alerts** — highlighted warnings for any detected metric regressions
- **Best Run Highlight** — the top-performing run with its config
- **Recommendations** — actionable next steps based on the analysis

## Architecture

| Node | Type | Description |
|---|---|---|
| `intake` | `event_loop`, client-facing | Gathers monitoring requirements from user |
| `fetch` | `event_loop`, background | Fetches W&B API data (runs, metrics, artifacts) |
| `analysis` | `event_loop`, background | Detects regressions, compares runs, surfaces insights |
| `report` | `event_loop`, client-facing | Writes HTML report, serves to user, handles follow-ups |

This template uses the `wandb_tool` integration merged in [#6963](https://github.com/adenhq/hive/pull/6963) and follows the exact same structure as the `deep_research_agent` template.
