# Rayda Fleet Copilot

An agentic AI copilot that lets IT administrators ask questions about device
telemetry, get grounded insights, and propose actions under human approval.

> **Status: complete.** Data pipeline, tools, agent, evaluation suite and provider switch.
> A LangGraph agent plans, calls tenant-scoped tools in an explicit loop,
> writes cited answers that are verified against the tool results, and
> proposes actions that pass an evidence gate and wait for human approval.
> Everything is logged and audited, and a 54-case eval suite scores it against
> ground truth computed independently from the raw data. The configured model
> is OpenAI GPT-5.5; Claude Opus 5.5 and Google Gemini can be selected in config
> or per command.
>
> Verified live on GPT-5.5 and Gemini 3.5 Flash. Not yet run on Claude: the
> Anthropic key's account has no credits. Gemini's agent evals are incomplete
> because the key hit its usage quota. See [Running evaluations](#running-evaluations).

---

## Quick start

Requires Python 3.11 (pinned in `.python-version`) and [uv](https://docs.astral.sh/uv/).
If Python 3.11 is not installed, `uv python install 3.11` fetches it.

```bash
make setup      # create .venv on Python 3.11 and install the project with dev tools
make ingest     # validate the dataset and build data/processed/fleet.db
make profile    # write reports/data_profile.md
make test       # run the test suite
make eval       # run the eval suite (no API key needed), writes reports/eval_scorecard.md
```

Try the tools directly (each call is audited like an agent call):

```bash
.venv/bin/fleet-copilot tools list
.venv/bin/fleet-copilot tools run detect_battery_end_of_life --tenant acme
.venv/bin/fleet-copilot tools run find_devices_by_metric --tenant acme-001 \
    --args '{"metric": "disk_free_pct", "comparison": "lt", "value": 15}'
```

Ask the copilot (uses the configured model, GPT-5.5: copy `.env.example` to
`.env` and set `OPENAI_API_KEY`):

```bash
.venv/bin/fleet-copilot ask "Which devices at Acme are low on disk space?" --tenant acme
.venv/bin/fleet-copilot ask "Which batteries are failing? Flag them for replacement." \
    --tenant acme --actor alice@acme
.venv/bin/fleet-copilot chat --tenant globex --actor bob@globex
```

Proposed actions are shown with their justification and evidence, and you
approve or reject each one before anything runs.

Use another provider for one command (key in `.env`: `OPENAI_API_KEY` or
`GOOGLE_API_KEY`), or set `llm.provider` and `llm.model` in `config/settings.yaml`:

```bash
.venv/bin/fleet-copilot --provider openai --model <openai-model> ask "..." --tenant acme
.venv/bin/fleet-copilot --provider google --model <gemini-model> chat --tenant initech
```

Without `make` or `uv`:

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev,openai,google]"
.venv/bin/fleet-copilot ingest
.venv/bin/fleet-copilot profile
.venv/bin/fleet-copilot audit-verify
.venv/bin/pytest
.venv/bin/fleet-copilot eval
```

## Project structure

```
rayda_copilot_assessment/
├── config/
│   └── settings.yaml              # paths, ingestion limits, device classes, logging
├── data/
│   ├── raw/                       # source dataset (committed, never modified)
│   ├── processed/fleet.db         # generated: fleet database, rebuilt by `ingest`
│   ├── audit/audit.db             # generated: append-only audit trail
│   ├── actions/actions.db         # generated: executed (approved) actions
│   ├── checkpoints/               # generated: agent state, so approvals can resume
│   └── quarantine/                # generated: rejected records with reasons
├── docs/                          # assessment brief
├── reports/                       # generated: data profile, eval scorecards and runs
├── logs/                          # generated: JSON log files
├── src/fleet_copilot/
│   ├── cli.py                     # `fleet-copilot`: ingest, profile, tools, ask, chat,
│   │                              #   eval, eval-compare, audit-verify
│   ├── exceptions.py              # app error hierarchy
│   ├── evidence.py                # Evidence model: snapshot id + cited values
│   ├── config/settings.py         # typed settings + fallback to defaults
│   ├── agents/
│   │   └── fleet_copilot_agent.py # session API: ask(), decide() on approvals
│   ├── graphs/
│   │   ├── fleet_copilot_graph.py # LangGraph StateGraph wiring (diagram inside)
│   │   ├── state.py               # CopilotState
│   │   ├── routing.py             # conditional edges
│   │   ├── dependencies.py        # services injected into nodes
│   │   └── nodes/                 # one file per node:
│   │       ├── input_guard_node.py        planner_node.py      executor_node.py
│   │       ├── tool_node.py               responder_node.py    verifier_node.py
│   │       ├── action_proposer_node.py    evidence_gate_node.py
│   │       ├── human_approval_node.py     action_execution_node.py
│   │       └── finalize_node.py           node_utils.py
│   ├── prompts/                   # one file per prompt, each with name + version
│   │   ├── system_prompt.py       planner_prompt.py    executor_prompt.py
│   │   ├── responder_prompt.py    action_proposer_prompt.py
│   │   └── output_schemas.py      # Plan, GroundedAnswer, ActionPlan
│   ├── llm/
│   │   ├── llm_client.py          # LLMClient interface, LangChain implementation, refusals
│   │   ├── llm_factory.py         # Claude / OpenAI / Gemini client from settings
│   │   └── recording_llm_client.py # record / replay responses for evals
│   ├── guardrails/
│   │   ├── tenant_context.py      # session tenant, resolved by id or name
│   │   ├── input_guard.py         # injection / cross-tenant / size checks
│   │   ├── answer_verifier.py     # citations and numbers vs tool results
│   │   ├── evidence_gate.py       # is each proposed action justified?
│   │   └── approval_policy.py     # explicit approval, default deny
│   ├── tools/
│   │   ├── schemas.py             # input schema per tool (no tenant field)
│   │   ├── read_tools.py          # current-state questions
│   │   ├── insight_tools.py       # time-series detectors as tools
│   │   ├── action_tools.py        # state-changing actions + ActionExecutor
│   │   ├── registry.py            # read/insight tool catalog
│   │   ├── executor.py            # validate → scope → run → truncate → audit
│   │   ├── context.py             # what a handler may use
│   │   └── results.py             # ToolResult: one shape for every tool
│   ├── insights/                  # deterministic detectors (pure functions)
│   │   ├── battery_detector.py    memory_detector.py
│   │   ├── storage_detector.py    compliance_detector.py
│   │   └── models.py              # Insight: finding, evidence, explanation
│   ├── db/
│   │   ├── schema.sql             # tables, constraints, indexes
│   │   ├── rows.py                # one typed row model per table
│   │   ├── connection.py          # connect (read-only for consumers), create schema
│   │   ├── writer.py              # bulk inserts from row models
│   │   ├── fleet_repository.py    # tenant-scoped read queries
│   │   └── action_store.py        # executed actions (idempotent)
│   ├── ingestion/                 # raw_models, loader, transformer, pipeline
│   ├── profiling/data_profiler.py # dataset coverage and quality report
│   ├── audit/audit_logger.py      # hash-chained, append-only audit log
│   └── utils/                     # logging, time, OS versions, units, stats, hashing, schema
└── tests/                         # unit + end-to-end tests (real dataset, scripted LLM)
```

Evaluation:

```
evals/
├── cases/                         # eval cases (YAML): grounded_qa, insights, actions,
│                                  #   guardrails, gate_redteam
└── recordings/                    # recorded model responses for deterministic replay
src/fleet_copilot/evaluation/
├── ground_truth.py                # expected answers from the raw NDJSON (independent of the app)
├── cases.py                       # case models and loader
├── tool_suite.py                  # reference tool calls vs ground truth (no LLM)
├── gate_suite.py                  # red-team proposals vs the evidence gate (no LLM)
├── agent_suite.py                 # full agent turns, scored check by check
├── metrics.py  scorecard.py       # scoring helpers, Markdown/JSON scorecard
├── comparison.py                  # latest agent run per provider/model, side by side
├── eval_context.py                # per-run settings, ground truth, scratch stores
└── runner.py                      # modes, suites, reports
```

## Tests

```bash
make test        # 226 tests, about 20 s, no API key or network
```

Tests run on the real dataset (ingested once into a temporary folder) and a
scripted LLM, so they are deterministic and free.

| File | Tests | Covers |
|---|---|---|
| `test_loader.py`, `test_transformer.py`, `test_pipeline.py` | 29 | Validation, quarantine, reject limit, atomic rebuild, tenant integrity, real-dataset shape |
| `test_settings.py`, `test_utils.py` | 20 | Settings fallback and strict validation; version parsing, units, time, slopes |
| `test_audit_logger.py` | 4 | Hash chain, append-only triggers, tamper detection |
| `test_tools.py` | 33 | Tool answers vs known facts, every tool cites evidence, truncation, error fallbacks, audit of every call |
| `test_tenant_isolation.py` | 43 | Every repository query and every tool × every tenant returns only that tenant's data; cross-tenant access denied and audited |
| `test_insights.py` | 13 | Detector logic on synthetic series: end of life, consistent vs intermittent, projections, drift |
| `test_guardrails.py` | 17 | Input guard, answer verifier, approval policy |
| `test_agent_graph.py` | 19 | Full graph with a scripted LLM: tool loop and cap, retry and fallback, guard refusals, evidence gate, approval interrupt/resume, idempotency |
| `test_action_store.py` | 2 | Re-worded duplicate actions stored once |
| `test_llm_providers.py` | 24 | Provider switch, client options per provider, tool definitions, refusal detection, comparison report |
| `test_evaluation.py` | 14 | Eval harness: cases valid, ground truth, scoring catches wrong answers, record/replay |
| `test_cli.py` | 8 | Commands end to end: tools, ask without a key, provider validation, offline eval, audit verify |

## Running evaluations

```bash
make eval            # = fleet-copilot eval --mode replay   (no API key)
make eval-offline    # = fleet-copilot eval --mode offline  (no model at all)
make eval-record     # = fleet-copilot eval --mode record   (configured model; saves responses)
make eval-compare    # side-by-side table of the latest run per provider/model
.venv/bin/fleet-copilot eval --mode live --case action_      # configured model, nothing saved
.venv/bin/fleet-copilot eval --layer tools --layer gate     # pick suites
```

Each run writes `reports/eval_runs/<run_id>/` with its own audit log, action
store and checkpoints (evals never touch the app's stores), plus
`eval_scorecard.md` and `eval_results.json`; the latest scorecard is copied to
`reports/eval_scorecard.md`. The command exits 1 when the pass rate is below
`--min-pass-rate` (default 1.0), so it can gate CI.

| Mode | Model | Agent cases | Deterministic |
|---|---|---|---|
| `offline` | none | only guardrail cases decided before any model call | yes |
| `replay` (default) | recorded responses | cases with recordings; others skipped, never guessed | yes |
| `record` | configured model (or `--provider`/`--model`) | all; responses saved to `evals/recordings/` | no (records once) |
| `live` | configured model | all; nothing saved | no |

**Why record/replay:** model output varies from run to run (and current
Claude models do not accept `temperature` at all), so the agent suite gets
determinism from recordings rather than from sampling settings. Tools are deterministic and
"now" is the dataset's as-of time, so the same question produces the same
model requests; each response is stored under a hash of the exact request.
Changing a prompt or tool changes the requests, and replay then reports those
cases as not recorded rather than replaying stale answers.

### What is evaluated

54 cases in `evals/cases/`, across three layers:

| Layer | Cases | Needs model | Checks |
|---|---|---|---|
| **tools** | 15 | no | Reference tool call per question vs ground truth: device precision/recall (or exact count), every item cites evidence present in the result, all evidence belongs to the tenant |
| **gate** | 12 | no | Red-team proposals (healthy device, wrong component, fabricated or missing evidence, cross-tenant device or employee, passing check, oversized message, and justified ones) must be blocked or allowed as expected |
| **agent** | 27 | 22 of 27 | Full turns: answer status, tool selection, answer names exactly the ground-truth devices (recall 1.0, precision ≥ 0.8) or contains the count, "cannot answer" when data is missing, citations verified, no other tenant's device named, proposals justified, nothing runs before approval or after a decline, guardrail refusals with zero model calls |

**Ground truth** (`evaluation/ground_truth.py`) is computed from the raw NDJSON
with its own code: no database, repository, tools or detectors. Only the
threshold definitions are shared, because they are the specification. A bug in
ingestion, a query or a detector shows up as a disagreement.

Case mix: the brief's three example questions (for several tenants), current
state (screen lock, software), insights (battery end of life, memory pressure
constant vs occasional, storage trend, compliance drift), an unanswerable
question (GPU temperature), four action requests (one deliberately
unjustified), and seven guardrail cases (cross-tenant by name and id, prompt
injection, approval bypass, system-prompt leak, out of scope, another
tenant's device id).

### Current results

GPT-5.5, recorded once and replayed by `make eval` (dataset as of 2026-06-12T11:00:00Z):

| Layer / suite | Cases | Passed | Failed | Skipped |
|---|---|---|---|---|
| tools / grounded_qa + insight | 15 | 15 | 0 | 0 |
| gate / red team | 12 | 12 | 0 | 0 |
| agent / grounded_qa | 9 | 9 | 0 | 0 |
| agent / insight | 7 | 7 | 0 | 0 |
| agent / action | 4 | 4 | 0 | 0 |
| agent / guardrail | 7 | 7 | 0 | 0 |
| **Total** | **54** | **54** | **0** | **0** |

| Metric | Value |
|---|---|
| Tool device precision / recall | 1.0 / 1.0 |
| Agent answer precision / recall | 1.0 / 1.0 |
| Answers verified on the first draft | 95% (the rest after one re-draft) |
| LLM calls per question | 3.4 |
| Live latency per question (recording run) | 20 s |
| Guardrail refusals made with zero model calls | 5 / 5 |

**What the evals caught on the way there** (each fixed, then re-recorded):

- An approved action was stored twice when the model re-worded its reason;
  duplicates are now matched on the target, not the free text.
- The model wrote a 70-character upgrade `spec`; the gate rightly blocked it,
  but the model had never been told the limit. Argument limits are now in the
  action catalog the model sees.
- "No battery findings" carried no evidence, so the model refused to answer
  without citations. Empty insight results now cite the checked snapshots.
- First drafts put thresholds in claims without evidence; thresholds now go in
  caveats.
- An eval bug: a correct "none" answer that listed passing devices as
  evidence was scored wrong. "None" questions now check for a negative answer.

**Other providers:** Gemini 3.5 Flash answered the live checks correctly, but
its agent eval run was cut short by the key's usage quota (429), so it has no
complete recording yet; provider errors are reported as errors, not as wrong
answers. Claude Opus 5.5 is not recorded: the Anthropic key's account has no
credits. `make eval-compare` shows recorded models side by side in
`reports/eval_comparison.md`.

The harness is tested too (`tests/test_evaluation.py`): a wrong tool, an
incomplete answer and a flipped gate expectation are each detected; record →
replay reproduces a run exactly with no model calls.

## Model providers

The configured default is OpenAI GPT-5.5 (`config/settings.yaml`); the
built-in default, used when no settings file exists, is Claude Opus 5.5.
All three providers sit behind the same `LLMClient` interface, so prompts,
tools, guardrails and the graph are identical for all three; only
`llm/llm_factory.py` knows the differences.

| | Claude (`anthropic`) | OpenAI (`openai`) | Gemini (`google`) |
|---|---|---|---|
| Key in `.env` | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` | `GOOGLE_API_KEY` |
| Package | installed | extra `.[openai]` | extra `.[google]` |
| Model used | `claude-opus-5-5` (built-in default) | `gpt-5.5` (configured default) | set `--model`, e.g. `gemini-3.5-flash` |
| Reasoning control | `llm.effort` (default `medium`) | `llm.reasoning_effort` | `llm.reasoning_effort` |
| `llm.temperature` | ignored (rejected by current Claude models) | passed through | passed through |
| Declined requests | server-side fallback model, then safe reply | `refusal` / `content_filter` → safe reply | safety finish reasons → safe reply |

How to switch:

- **Config:** set `llm.provider` and `llm.model` in `config/settings.yaml`.
- **Per command:** `fleet-copilot --provider openai --model <id> ask ...`
  (or `FLEET_COPILOT_LLM_PROVIDER` / `FLEET_COPILOT_LLM_MODEL`).
- Switching provider without a model is refused, and so is a Claude model id
  with another provider, so a run never silently uses the wrong model.

One tool format (OpenAI-style function definitions) is accepted by all three
integrations, and structured outputs use JSON-schema mode everywhere. Tests
check, without network calls, that each provider's client is built with the
right options, accepts the tool definitions (and never the action tools), and
that each provider's refusal signals are caught.

**Comparing providers.** Record the agent suite once per model, then compare:

```bash
.venv/bin/fleet-copilot eval --mode record                                  # configured (GPT-5.5)
.venv/bin/fleet-copilot --provider anthropic --model claude-opus-5-5 eval --mode record
.venv/bin/fleet-copilot --provider google --model <id> eval --mode record
.venv/bin/fleet-copilot eval-compare      # writes reports/eval_comparison.md
```

The comparison shows, per provider and model: agent cases passed, answer
recall and precision, answers verified on the first draft, right tool chosen,
proposals justified, LLM calls per question and latency. Recordings are keyed
by model, so each model replays its own responses. For each model it uses the
latest run that called the model (record or live), falling back to the latest
replay; latency is only shown for runs that called the model. Runs with
skipped cases or provider errors are labelled as such. The terminal shows a
compact table; `reports/eval_comparison.md` has every metric.

## Agent architecture

The agent is an explicit LangGraph `StateGraph`, not a prompt wrapper.

```
START → input_guard ──(blocked)───────────────────────────────┐
           │                                                  │
        planner ──(out of scope / model error)────────────────┤
           │                                                  │
        executor ◄──┐  explicit tool loop                     │
           │        │  (≤ agent.max_tool_iterations rounds)   │
           ├─► tools┘                                         │
           │                                                  │
        responder ◄──┐                                        │
           │         │ citations failed, retry left           │
        verifier ────┘                                        │
           │ verified + user asked for actions                │
           ├─► action_proposer → evidence_gate ─(none ready)──┤
           │                         │                        │
           │                   human_approval  ⏸ interrupt    │
           │                         │                        │
           │                   execute_actions                │
           └──────────────────────► finalize ◄────────────────┘ → END
```

| Node | LLM? | Does |
|---|---|---|
| `input_guard` | no | Blocks empty/oversized input, prompt-injection phrasing, and requests naming another company |
| `planner` | yes (structured) | Classifies intent (question / insight / action / out of scope), lists the tool calls, sets `wants_actions` |
| `executor` ⇄ `tools` | yes (tool calling) | Runs the plan through read/insight tools only; the loop is graph edges, capped by config |
| `responder` | yes (structured) | Writes the answer from tool results only, citing snapshot ids, listing every claim |
| `verifier` | no | Checks every claim cites ids from this turn and every number appears in the tool results; one retry, then a deterministic fallback answer |
| `action_proposer` | yes (structured) | Proposes actions with arguments, justification and evidence ids (only when asked and the answer verified) |
| `evidence_gate` | no | Refuses proposals the evidence does not support (rules below) |
| `human_approval` | no | `interrupt()`: the graph pauses, state is checkpointed in SQLite |
| `execute_actions` | no | Runs only approved proposals; re-checks tenant ownership; idempotent |
| `finalize` | no | Composes the reply and writes one audit event for the whole turn |

**Model:** GPT-5.5 as configured in `config/settings.yaml`. Claude Opus 5.5
(`claude-opus-5-5`, effort `medium`, server-side fallback enabled) and Gemini
can be selected instead (see [Model providers](#model-providers)).
Current Claude models do not accept `temperature`, so reproducibility comes
from deterministic tools, structured outputs, a fixed as-of time and recorded
responses. Nodes depend on a small `LLMClient` interface, so tests run on a
scripted model.

**Sessions:** each question runs in its own graph thread. A thread waiting for
approval can be resumed later, even from another process
(`FleetCopilotAgent.decide(thread_id, decisions)`); only the same tenant can
resume it.

## Action catalog and approval

The model never calls these directly. They are not in the tool loop's
registry: if the model tries, it gets "unknown tool".

| Action | Arguments | Evidence the gate requires (from this turn) |
|---|---|---|
| `flag_device_for_replacement` | `device_id`, `reason` | Battery at end of life, or ≥ 2 high-severity findings on the device |
| `create_upgrade_order` | `device_id`, `component` (ram/storage/battery), `spec` | Matching finding: memory pressure / storage pressure or low disk / battery end of life |
| `open_remediation_ticket` | `device_id`, `check_id`, `note` | The check is failing in the device's latest snapshot |
| `notify_employee` | `employee_id`, `message` (≤ 500 chars) | A finding or failing check on one of the employee's devices |

Every proposal must also: match the action schema, target a device or
employee of the session's company, and cite evidence ids that exist in this
turn's tool results and concern the target. Duplicates in one turn are refused.

Approval policy: explicit per-proposal decision; anything undecided is
rejected (default deny); the approver is recorded. Approved actions are
written to `data/actions/actions.db` (stand-in for ticketing, procurement and
messaging integrations); the same approved action is never stored twice.

## Data model

| Table | One row per | Notes |
|---|---|---|
| `companies` | tenant | name derived from id (`acme-001` → `Acme`) |
| `employees` | employee | belongs to exactly one company |
| `devices` | device | type (`laptop`/`desktop`), model, RAM, first/last seen |
| `snapshots` | device × collection | OS version split into major/minor/release, memory %, battery |
| `disk_volumes` | volume per snapshot | free %, encryption, system-volume flag |
| `compliance_results` | check per snapshot | `check_id`, severity, pass/fail |
| `installed_software` | app per snapshot | name, version, publisher |
| `dataset_metadata` | key | `as_of`, source SHA-256, ingestion run id |

**Evidence ids.** Each snapshot has a readable, deterministic id such as
`1LYSSFD074BB@20260514T0954Z` (device + UTC time). The agent cites these.

**Tenant column everywhere.** Every row reaches `company_id` directly or via
its snapshot, so tenant isolation becomes one `WHERE` clause in the tool layer.

## Tool catalog

All tools are read-only and scoped to the session's tenant. Thresholds come
from `config/settings.yaml` and are restated in every result's `assumptions`.

| Tool | Kind | Answers | Key arguments |
|---|---|---|---|
| `get_fleet_overview` | read | Device counts by type, OS, model; failing devices per check | — |
| `list_devices` | read | Inventory, optionally filtered | `device_type`, `os_name`, `model_name`, `employee_id` |
| `get_device_details` | read | Full latest state of one device | `device_id` |
| `find_devices_by_metric` | read | "Which devices are low on disk now?" | `metric`, `comparison`, `value` |
| `find_devices_by_os_version` | read | "How many devices run an OS older than macOS 15?" | `os_name`, `comparison`, `version` |
| `get_compliance_status` | read | "Laptops failing high-severity checks" | `check_id`, `severity`, `status`, `device_type` |
| `search_installed_software` | read | Devices with a given application | `name` |
| `get_device_history` | read | How a device's metrics changed | `device_id`, `metrics`, `days` |
| `detect_battery_end_of_life` | insight | Batteries at / approaching end of life | `days` |
| `detect_memory_pressure` | insight | Devices consistently or intermittently RAM-constrained | `days` |
| `detect_storage_pressure` | insight | Devices low on storage, with days until critical | `days` |
| `detect_compliance_drift` | insight | Checks that started failing; fleet failing counts start → end | `check_id`, `days` |

Default thresholds: low disk < 15% free (critical ≤ 5%), high memory ≥ 85%,
battery cycle limit 1000 (warning at 80%), capacity decline ≥ 3%;
"consistent" = ≥ 70% of snapshots, "intermittent" = ≥ 30%, at least 7 snapshots.

### Every result has the same shape

```json
{
  "tool": "find_devices_by_metric",
  "status": "ok | not_found | error",
  "summary": "2 of 10 devices have system disk free % below 15.",
  "tenant_id": "acme-001",
  "as_of": "2026-06-12T11:00:00Z",
  "items": [{"device_id": "MT7PJB7N5LRE", "disk_free_pct": 2.0, "evidence_ids": ["MT7PJB7N5LRE@20260612T0839Z"]}],
  "evidence": [{"snapshot_id": "MT7PJB7N5LRE@20260612T0839Z", "values": {"disk_free_pct": 2.0}, "...": "..."}],
  "assumptions": ["Values come from each device's latest snapshot (dataset as of ...)."],
  "coverage": {"devices_checked": 10}
}
```

Insight items add `finding`, `explanation`, `severity` and `metrics`, which is
the brief's "finding, supporting evidence, brief explanation".

## Grounding strategy

- **Code computes, the model explains.** Counts, filters, trends, slopes and
  projections are deterministic Python/SQL; the model is told to copy numbers,
  and the verifier rejects any number not present in the tool results.
- **Every item cites evidence.** Each item carries `evidence_ids`, and each
  evidence record holds the exact values used. A test checks that every cited
  id is present in the same result.
- **Empty answers are grounded too.** "0 of 9 laptops fail high-severity
  checks" comes with the passing snapshots that were checked, and a detector
  that finds nothing cites the latest snapshot of every device it checked.
- **"Now" means the dataset's as-of time** (latest snapshot), not the wall
  clock, so answers are reproducible.
- **Coverage and assumptions travel with the answer:** how many devices were
  checked or skipped (e.g. no battery data), and which thresholds applied.
- **Answers are verified before they are shown.** The responder must list
  every factual statement as a claim with evidence ids. The verifier rejects
  unknown ids and numbers not present in the tool results, asks for one
  re-draft, then falls back to a deterministic answer built from tool
  summaries. Actions are never proposed from an unverified answer.
- **Tool output is data, not instructions**, stated in the system prompt.

## Tenant isolation

- The tenant is fixed when a session starts and passed to the data layer by
  code. **No tool has a tenant argument**; an extra `company_id` argument is
  rejected by the schema.
- Every query in `FleetRepository` filters `company_id = :company_id`.
- The input guard refuses requests that name another company before any
  model call, and audits them as `denied`.
- Actions re-check that the target belongs to the tenant right before running,
  and approval threads can only be resumed by the same company.
- Asking for another company's device returns the same "not found" as a
  device that does not exist, so nothing leaks, but the audit log records it
  as `denied` with reason `cross_tenant_device`.
- Tests run every tool for every tenant and assert that no item or evidence
  belongs to another company.

## Reliability: fallbacks, logs and audit

### Fallbacks

| Situation | Behaviour |
|---|---|
| `settings.yaml` missing | Run on built-in defaults, log a warning |
| `settings.yaml` invalid (bad value, unknown key, bad YAML) | Stop with a clear error; never run on silently corrected settings |
| Malformed JSON or schema-invalid record | Reject that line, write it to `data/quarantine/` with the reason, continue |
| Duplicate snapshot (same device and time) | Reject the duplicate |
| Reject ratio above `max_reject_ratio` (5%) | Abort; the previous database is kept |
| Any failure while building the database | Built in a temp file and swapped in atomically, so the previous database is kept |
| Device or employee under two companies | Abort: tenant integrity violation |
| Unknown OS version format | Keep the raw string, leave parsed fields empty, log a warning |
| Missing battery section | Stored as "not reported", never as zero |
| Log file cannot be opened | Continue with console logging |
| Audit write fails | The operation fails (fail closed) |
| Unknown tool or invalid tool arguments | `error` result explaining what is valid; nothing reaches the database |
| Tool handler raises unexpectedly | `error` result with a generic message (details in the log); the agent loop continues |
| Fleet database missing | `error` result telling the user to run `ingest` |
| Result larger than `tools.max_rows` (50) | Truncated, flagged, evidence trimmed to the kept items |
| Too few snapshots for a trend (< 7) | Device skipped and listed in `coverage`; no trend is claimed |
| API key for the chosen provider missing | Clear error at session start; tools, ingestion and offline evals still work |
| Provider package not installed | Clear error with the `pip install -e ".[openai]"` / `".[google]"` command |
| Provider switched without a model, or Claude model with another provider | Refused at startup |
| Rate limit, timeout, 5xx from the model | SDK retries (`llm.max_retries`) |
| Model declines or blocks a request | Claude: server-side fallback model; any provider: if still declined, a safe reply |
| Planner fails | Safe reply, no tools run, no actions |
| Executor fails mid-loop | Answer from the evidence already gathered |
| Answer fails citation checks twice | Deterministic answer from tool summaries; no actions proposed |
| Proposer fails | Answer is returned; "actions could not be generated" |
| Model requests an action tool | "Unknown tool" result; nothing runs |
| No approval decision for a proposal | Rejected (default deny) |
| Same action approved twice | Stored once; second run reported as duplicate |

### Logs

- Console: readable lines. File: `logs/fleet_copilot.log`, one JSON object per
  line, rotated at 5 MB.
- Every line carries a `run_id` (e.g. `ingest-21118ac3f4f9`) that also appears
  in the audit log and the database metadata, so one run can be traced end to end.

### Audit

- Stored in its own database (`data/audit/audit.db`), so it survives rebuilds.
- **Append-only:** database triggers block `UPDATE` and `DELETE`.
- **Tamper-evident:** each entry holds the SHA-256 of its content and of the
  previous entry. `fleet-copilot audit-verify` recomputes the chain.
- Each entry records actor, tenant, event type, action, status, details and `run_id`.
- Ingestion writes `started`, then `succeeded` or `failed`.
- Every tool call writes one event: actor, tenant, tool, arguments, result
  status, item count, cited evidence ids, duration; status `succeeded`,
  `failed` or `denied`.
- The agent adds: `input_guard` (denied), `action_proposal` (proposed /
  denied, with gate reasons), `action_approval` (approved / rejected, with
  approver), `action` (succeeded / failed) and one `agent_turn` per request
  (intent, tools used, evidence ids, verification result, proposal outcomes).
- Logs carry the thread id as `run_id`, so a request's log lines, tool calls
  and audit events can be joined.

## What the data profile shows

From `reports/data_profile.md` (run `make profile`):

- 3 companies, 25 devices (22 laptops, 3 Mac mini desktops), 14 employees,
  750 snapshots: one per device per day for 30 days (2026-05-14 → 2026-06-12).
- macOS 13.6 / 14.5 / 15.3 / 15.4 and Windows 10 22H2 / 11 22H2 / 11 23H2;
  no device changed OS version in the window.
- **No snapshot fails the only high-severity check (`disk_encryption`).** The
  correct answer to "laptops failing high-severity checks" is "none". This
  becomes an eval case.
- 10 laptop snapshots have no battery data; trend logic must skip them.
- Battery data has no design capacity, so battery health comes from cycle
  count, condition and the full-charge capacity trend.

## Design decisions and trade-offs

- **Typed SQL storage instead of RAG.** The telemetry is fully structured;
  exact filters and counts need SQL, and rows give citable evidence ids.
- **Validate at the edge.** Pydantic models define what a valid record is;
  everything downstream can trust the database.
- **Fail loud on configuration and tenant problems; degrade gracefully on bad
  data lines.** A wrong setting or mixed tenant is a systemic issue, while a
  single bad record is not.
- **Read-only connections for consumers.** Only ingestion writes to the fleet
  database; tools open it read-only.
- **Narrow, typed tools instead of text-to-SQL.** Enums and schemas make bad
  calls fail clearly, tenant scoping cannot be bypassed, and results are
  reproducible. Trade-off: a question outside the catalog needs a new tool.
- **Filter in Python after one scoped query.** With 25 devices this keeps the
  SQL fixed and simple; at real scale filters would move into SQL.
- **Battery health without design capacity.** Judged from service condition,
  cycle count against a rated limit, and capacity decline, and stated as such.
- **Explicit graph over a prebuilt ReAct agent.** Planning, the tool loop,
  verification, the evidence gate and approval are separate, testable nodes.
  Trade-off: more code than `create_react_agent`, and each node is a place to
  maintain.
- **Deterministic guardrails, LLM for language.** Input guard, verifier,
  evidence gate and approval policy are plain code, so they cannot be talked
  out of their rules. Trade-off: the input guard's patterns are simple and can
  miss novel phrasings; the data layer's tenant scoping is the real boundary.
- **Propose, then approve.** The model can only propose; actions run after an
  explicit human decision, with the evidence shown. Trade-off: one extra step
  for the administrator on every action.
- **Structured outputs via JSON schema, not forced tool calls.** Required for
  Claude Opus 5.5, and it keeps the plan, answer and proposals typed.
- **One thread per question.** Simple, auditable turns; follow-up questions do
  not see earlier answers yet.
- **Evaluate in layers.** Tool and gate suites run without a model, so most
  correctness and safety properties are checked deterministically and for free;
  the agent suite then checks what only the model can get wrong (tool choice,
  answer completeness, proposal quality). Trade-off: agent cases need a
  recording session with a real API key before they count.
- **Strict answer scoring.** An answer must name every ground-truth device;
  one extra device in five is tolerated. Trade-off: a correct answer that
  also lists near-threshold devices for context can fail, which is preferable
  to passing answers that leave devices out.
- **Provider switch, not per-request routing.** One model per session, chosen
  in config or per command, keeps every turn traceable to one model and keeps
  evals comparable. Routing simple questions to a cheaper model is possible on
  top of the same interface but is not built: at 25 devices the cost does not
  justify the extra failure modes.
- **Explicit provider classes over `init_chat_model`.** Each provider needs
  different options (Claude: effort and fallback, no temperature; others:
  temperature and reasoning effort), so the factory builds each class
  directly. Integrations are optional extras, imported only when selected.

