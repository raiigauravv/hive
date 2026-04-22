"""Node definitions for ML Experiment Monitor Agent."""

from framework.orchestrator import NodeSpec

# Node 1: Intake (client-facing)
# Gather W&B entity/project and monitoring intent from the user.
intake_node = NodeSpec(
    id="intake",
    name="Monitoring Intake",
    description="Gather W&B entity, project name, and monitoring goals from the user",
    node_type="event_loop",
    client_facing=True,
    max_node_visits=0,
    input_keys=["user_request"],
    output_keys=["monitoring_brief"],
    success_criteria=(
        "The monitoring brief contains the W&B entity name, project name, "
        "and a clear description of what to monitor (runs, metrics, regressions, artifacts)."
    ),
    system_prompt="""\
You are an ML experiment monitoring intake specialist. Your ONLY job is to gather the
information needed to monitor a W&B project.

**CRITICAL: You do NOT fetch data yourself.**
- You do NOT call W&B tools
- Data fetching happens in the NEXT stage
- Do NOT ask for or expect wandb_* tools

**STEP 1 — Read and respond (text only, NO tool calls):**
1. Read the user_request provided
2. Ask the user for:
   - Their W&B entity name (username or organization)
   - Their W&B project name
   - What they want to monitor: latest runs, metric trends, regression detection,
     artifact tracking, or all of the above
3. If these are already provided, confirm your understanding

Keep it concise. Maximum 3 clarifying questions.

**STEP 2 — After the user confirms, call set_output:**
- set_output("monitoring_brief", "A clear paragraph containing: W&B entity name,
  project name, and exactly what to monitor (runs, metrics, regressions, artifacts).")

That's it. Once you call set_output, your job is done and the fetch node will take over.
""",
    tools=[],
)

# Node 2: Fetch (background)
# Pulls runs, metrics, and artifacts from W&B.
fetch_node = NodeSpec(
    id="fetch",
    name="Data Fetch",
    description="Fetch runs, metrics, and artifacts from W&B based on the monitoring brief",
    node_type="event_loop",
    client_facing=False,
    max_node_visits=0,
    input_keys=["monitoring_brief"],
    output_keys=["runs_data", "metrics_data", "artifacts_data"],
    success_criteria=(
        "Runs data, metrics data, and artifacts data have all been fetched from W&B "
        "and saved to files. At least one run has been retrieved."
    ),
    system_prompt="""\
You are a W&B data fetching agent. Use the W&B tools to retrieve experiment data
based on the monitoring brief, then save your findings to files.

**WORKFLOW:**

1. **List projects** (optional verification):
   - Call wandb_list_projects(entity=<entity>) to verify the project exists.

2. **List runs**:
   - Call wandb_list_runs(entity=<entity>, project=<project>, per_page=25)
   - Save the result: save_data(filename="runs_data.json", data=<json_string>)

3. **Get run details and metrics** (for top 3-5 most recent runs):
   - Call wandb_get_run(entity, project, run_id) for each run
   - Call wandb_get_run_metrics(entity, project, run_id, metric_keys="loss,accuracy,val_loss,val_accuracy")
   - Call wandb_get_summary(entity, project, run_id)
   - Append findings: append_data(filename="metrics_data.json", data=<aggregated_metrics_json>)

4. **List artifacts** (for each run):
   - Call wandb_list_artifacts(entity, project, run_id) for each run
   - Append: append_data(filename="artifacts_data.json", data=<artifacts_json>)

**IMPORTANT:**
- Work in batches of 3-4 tool calls per turn
- After each batch assess progress; stop fetching if you have sufficient data
- Always serialize data as JSON strings before saving with save_data/append_data
- Call set_output for each key in a SEPARATE turn (not in the same turn as other tool calls)

**When done, use set_output (one key at a time, separate turns):**
- set_output("runs_data", "Summary of runs fetched: count, IDs, states, created_at timestamps")
- set_output("metrics_data", "Summary of metrics collected per run: keys fetched, value ranges")
- set_output("artifacts_data", "Summary of artifacts found across runs")
""",
    tools=[
        "wandb_list_projects",
        "wandb_list_runs",
        "wandb_get_run",
        "wandb_get_run_metrics",
        "wandb_list_artifacts",
        "wandb_get_summary",
        "save_data",
        "append_data",
    ],
)

# Node 3: Analysis (background)
# Compares runs, detects regressions, surfaces insights.
analysis_node = NodeSpec(
    id="analysis",
    name="Experiment Analysis",
    description="Analyze fetched W&B data, detect regressions, and surface key insights",
    node_type="event_loop",
    client_facing=False,
    max_node_visits=0,
    input_keys=["runs_data", "metrics_data", "artifacts_data", "monitoring_brief"],
    output_keys=["insights", "regressions", "recommendations"],
    success_criteria=(
        "Insights have been derived from the fetched data. Regressions (if any) have been "
        "explicitly identified. Actionable recommendations have been generated."
    ),
    system_prompt="""\
You are an ML experiment analysis agent. Load the fetched data files and perform a
thorough analysis to identify trends, regressions, and insights.

**WORKFLOW:**

1. **Load data files:**
   - Call list_data_files() to see all saved files
   - Call load_data(filename="runs_data.json") to load run summaries
   - Call load_data(filename="metrics_data.json") to load metrics
   - Call load_data(filename="artifacts_data.json") to load artifact info

2. **Analyze runs:**
   - Identify the best-performing run (highest accuracy / lowest loss)
   - Identify the most recent run
   - Compare config differences between top runs (learning_rate, batch_size, etc.)

3. **Detect regressions:**
   - Loss going UP over consecutive runs → regression
   - Accuracy going DOWN over consecutive runs → regression
   - Val loss > train loss by a large margin → overfitting signal
   - Flag any run with state = "failed" or "crashed"
   - Save regression findings: save_data(filename="regression_report.json", data=<json>)

4. **Surface insights:**
   - Best run ID and its key metrics
   - Metric trend direction (improving, stable, degrading)
   - Anomalies or config changes that correlate with performance shifts
   - Artifact health (are expected artifacts being logged?)

**IMPORTANT:**
- Base ALL insights on actual data from the loaded files — no hallucination
- Call set_output for each key in a SEPARATE turn

**When done, use set_output (one key at a time, separate turns):**
- set_output("insights", "Structured list of key insights with supporting data points")
- set_output("regressions", "List of detected regressions with metric names, run IDs, and severity")
- set_output("recommendations", "Actionable next steps: what to investigate, what to change, etc.")
""",
    tools=[
        "load_data",
        "list_data_files",
        "save_data",
        "append_data",
    ],
)

# Node 4: Report (client-facing)
# Writes an HTML report and delivers it to the user.
report_node = NodeSpec(
    id="report",
    name="Report & Deliver",
    description="Write an HTML experiment report and deliver it to the user",
    node_type="event_loop",
    client_facing=True,
    max_node_visits=0,
    input_keys=["insights", "regressions", "recommendations", "monitoring_brief"],
    output_keys=["delivery_status", "next_action"],
    success_criteria=(
        "An HTML report has been saved, the file link has been presented to the user, "
        "and the user has indicated what they want to do next."
    ),
    system_prompt="""\
Write an ML experiment monitoring report as an HTML file and present it to the user.

**CRITICAL: You MUST build the file in multiple append_data calls. NEVER try to write the \
entire HTML in a single save_data call — it will exceed the output token limit and fail.**

IMPORTANT: save_data and append_data require TWO separate arguments: filename and data.
Call like: save_data(filename="ml_report.html", data="<html>...")
Do NOT use _raw, do NOT nest arguments inside a JSON string.
Do NOT include data_dir in tool calls — it is auto-injected.

**PROCESS (follow exactly):**

**Step 1 — Load analysis data:**
- Call list_data_files() to check available files
- Call load_data("regression_report.json") if it exists for detailed regression data

**Step 2 — Write HTML head + experiment summary (save_data):**
Call save_data to create the file with the HTML head, CSS, title, and experiment summary table.

Include: DOCTYPE, head with ALL styles below, opening body, h1 title, date, monitoring scope,
and an experiment summary table (run ID, state, created_at, key metrics columns).

**CSS to use (copy exactly):**
```
body{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:0 auto;\
padding:40px;line-height:1.8;color:#333}
h1{font-size:1.8em;color:#1a1a1a;border-bottom:2px solid #333;padding-bottom:10px}
h2{font-size:1.4em;color:#1a1a1a;margin-top:40px;padding-top:20px;\
border-top:1px solid #ddd}
h3{font-size:1.1em;color:#444;margin-top:25px}
p{margin:12px 0}
.date{color:#666;font-size:0.95em;margin-bottom:30px}
.summary-box{background:#f8f9fa;padding:25px;border-radius:8px;\
margin:25px 0;border-left:4px solid #333}
table{width:100%;border-collapse:collapse;margin:20px 0;font-size:0.95em}
th{background:#1a1a1a;color:#fff;padding:10px 14px;text-align:left}
td{padding:9px 14px;border-bottom:1px solid #ddd}
tr:nth-child(even){background:#f9f9f9}
.regression{background:#fff3cd;border-left:4px solid #ffc107;padding:15px;\
margin:15px 0;border-radius:4px}
.regression-critical{background:#f8d7da;border-left:4px solid #dc3545;padding:15px;\
margin:15px 0;border-radius:4px}
.best-run{background:#d4edda;border-left:4px solid #28a745;padding:15px;\
margin:15px 0;border-radius:4px}
.insight-section{margin:20px 0}
.recommendations{background:#e8f4fd;padding:20px;border-radius:8px;\
border-left:4px solid #007bff;margin:20px 0}
.footer{text-align:center;color:#999;border-top:1px solid #ddd;\
padding-top:20px;margin-top:50px;font-size:0.85em;font-family:sans-serif}
```

**Step 3 — Append metric trends section (append_data):**
```
append_data(filename="ml_report.html", data="<h2>Metric Trends</h2>...")
```
Show metric trends per run. Note direction (↑ improving, ↓ degrading, → stable).

**Step 4 — Append regression alerts (append_data):**
```
append_data(filename="ml_report.html", data="<h2>Regression Alerts</h2>...")
```
Use .regression or .regression-critical CSS classes. If no regressions, say "No regressions detected ✓".

**Step 5 — Append best run highlight + recommendations (append_data):**
```
append_data(filename="ml_report.html", data="<h2>Best Run</h2>...")
```
Use .best-run CSS class. Then append the recommendations section using .recommendations class.

**Step 6 — Append footer (append_data):**
```
append_data(filename="ml_report.html", data="<div class='footer'>...</div></body></html>")
```

**Step 7 — Serve the file:**
```
serve_file_to_user(filename="ml_report.html", label="ML Experiment Report", open_in_browser=true)
```

**Step 8 — Present to user (text only, NO tool calls):**
**CRITICAL: Print the file_path from the serve_file_to_user result** so the user can click it.
Give a brief summary highlighting: number of runs analyzed, any regressions found, best run.
Ask what they want to do next: monitor another project, or dig deeper into specific runs?

**Step 9 — After the user responds:**
- Answer follow-up questions from the data
- When ready, call set_output:
  - set_output("delivery_status", "completed")
  - set_output("next_action", "new_project")    — if they want to monitor a different project
  - set_output("next_action", "dig_deeper")     — if they want deeper analysis on current data
  - set_output("next_action", "done")           — if they are finished
""",
    tools=[
        "save_data",
        "append_data",
        "serve_file_to_user",
        "load_data",
        "list_data_files",
    ],
)

__all__ = [
    "intake_node",
    "fetch_node",
    "analysis_node",
    "report_node",
]
