"""Command line entry point: `fleet-copilot <command>`."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from fleet_copilot.agents import AgentTurn, FleetCopilotAgent
from fleet_copilot.audit import AuditLogger
from fleet_copilot.config import Settings, load_settings, with_llm_overrides
from fleet_copilot.db.connection import connect
from fleet_copilot.exceptions import FleetCopilotError
from fleet_copilot.guardrails.approval_policy import ApprovalDecision
from fleet_copilot.guardrails.tenant_context import resolve_tenant
from fleet_copilot.ingestion.pipeline import run_ingestion
from fleet_copilot.profiling.data_profiler import build_profile, render_markdown
from fleet_copilot.tools import TOOL_REGISTRY, ToolExecutor
from fleet_copilot.utils.logging_utils import configure_logging, new_run_id, set_run_id

app = typer.Typer(help="Rayda Fleet Copilot.", no_args_is_help=True, add_completion=False)
tools_app = typer.Typer(help="Inspect and run copilot tools directly.", no_args_is_help=True)
app.add_typer(tools_app, name="tools")
console = Console()

T = TypeVar("T")


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None, typer.Option("--config", help="Path to a settings YAML file.")
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            envvar="FLEET_COPILOT_LLM_PROVIDER",
            help="LLM provider for this command: anthropic | openai | google.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model", envvar="FLEET_COPILOT_LLM_MODEL", help="Model id for the provider."
        ),
    ] = None,
) -> None:
    load_dotenv()  # API keys from .env, if present
    settings = _run_or_exit(lambda: with_llm_overrides(load_settings(config), provider, model))
    configure_logging(settings.logging, settings.paths.log_dir)
    ctx.obj = settings


@app.command()
def ingest(ctx: typer.Context) -> None:
    """Validate the raw dataset and (re)build the fleet database."""
    settings: Settings = ctx.obj
    set_run_id(new_run_id("ingest"))
    summary = _run_or_exit(lambda: run_ingestion(settings, AuditLogger(settings.paths.audit_db)))

    table = Table(title="Ingestion summary", show_header=False)
    for key, value in asdict(summary).items():
        table.add_row(key, str(value))
    console.print(table)
    if summary.records_rejected:
        console.print(
            f"[yellow]{summary.records_rejected} record(s) quarantined in "
            f"{settings.paths.quarantine_file}[/yellow]"
        )


@app.command()
def profile(
    ctx: typer.Context,
    output: Annotated[Path | None, typer.Option("--output", help="Markdown report path.")] = None,
) -> None:
    """Profile the fleet database and write a Markdown report."""
    settings: Settings = ctx.obj
    set_run_id(new_run_id("profile"))
    data_profile = _run_or_exit(lambda: build_profile(settings.paths.fleet_db))

    report_path = output or settings.paths.reports_dir / "data_profile.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(data_profile), encoding="utf-8")

    for section in data_profile.sections:
        table = Table(title=section.title)
        for header in section.headers:
            table.add_column(header)
        for row in section.rows:
            table.add_row(*("—" if v is None else str(v) for v in row))
        console.print(table)
    for note in data_profile.observations:
        console.print(f"• {note}")
    console.print(f"\n[green]Report written to {report_path}[/green]")


@app.command("audit-verify")
def audit_verify(ctx: typer.Context) -> None:
    """Verify the audit log hash chain."""
    settings: Settings = ctx.obj
    set_run_id(new_run_id("audit"))
    count = _run_or_exit(lambda: AuditLogger(settings.paths.audit_db).verify_integrity())
    console.print(f"[green]Audit log intact: {count} entries verified.[/green]")


@tools_app.command("list")
def tools_list() -> None:
    """Show the tool catalog."""
    table = Table(title="Tools")
    table.add_column("Name", no_wrap=True)  # keep names copyable
    for column in ("Category", "Arguments", "Description"):
        table.add_column(column)
    for spec in TOOL_REGISTRY.values():
        arguments = ", ".join(spec.input_model.model_fields) or "—"
        table.add_row(spec.name, spec.category, arguments, spec.description)
    console.print(table)


@tools_app.command("run")
def tools_run(
    ctx: typer.Context,
    tool_name: Annotated[str, typer.Argument(help="Tool name, see `tools list`.")],
    tenant: Annotated[str, typer.Option("--tenant", help="Company id or name, e.g. acme-001.")],
    args: Annotated[str, typer.Option("--args", help="Tool arguments as JSON.")] = "{}",
    actor: Annotated[str, typer.Option("--actor", help="Who is running the tool.")] = "cli-admin",
) -> None:
    """Run one tool for one tenant and print its JSON result (audited like agent calls)."""
    settings: Settings = ctx.obj
    set_run_id(new_run_id("tool"))
    try:
        arguments = json.loads(args)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Error:[/red] --args is not valid JSON: {exc}")
        raise typer.Exit(code=1) from exc

    def run():
        with closing(connect(settings.paths.fleet_db, read_only=True)) as conn:
            tenant_ctx = resolve_tenant(conn, tenant, actor=actor)
        executor = ToolExecutor(settings, tenant_ctx, AuditLogger(settings.paths.audit_db))
        return executor.execute(tool_name, arguments)

    result = _run_or_exit(run)
    console.print_json(result.model_dump_json())
    if result.status == "error":
        raise typer.Exit(code=1)


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument(help="Question or request for the copilot.")],
    tenant: Annotated[str, typer.Option("--tenant", help="Company id or name.")],
    actor: Annotated[str, typer.Option("--actor", help="Administrator using the copilot.")] = (
        "cli-admin"
    ),
) -> None:
    """Ask one question. Proposed actions are shown for approval before anything runs."""
    settings: Settings = ctx.obj
    agent = _run_or_exit(lambda: FleetCopilotAgent.create(settings, tenant, actor))
    try:
        _handle_turn(agent, _run_or_exit(lambda: agent.ask(question)))
    finally:
        agent.close()


@app.command()
def chat(
    ctx: typer.Context,
    tenant: Annotated[str, typer.Option("--tenant", help="Company id or name.")],
    actor: Annotated[str, typer.Option("--actor", help="Administrator using the copilot.")] = (
        "cli-admin"
    ),
) -> None:
    """Interactive session. Type 'exit' to quit."""
    settings: Settings = ctx.obj
    agent = _run_or_exit(lambda: FleetCopilotAgent.create(settings, tenant, actor))
    console.print(
        f"[bold]Fleet Copilot[/bold] for {agent.tenant.company_name} "
        f"(as {agent.tenant.actor}). Type 'exit' to quit."
    )
    try:
        while True:
            question = console.input("\n[bold cyan]you>[/bold cyan] ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                _handle_turn(agent, _run_or_exit(lambda q=question: agent.ask(q)))
    except (KeyboardInterrupt, EOFError):
        console.print()
    finally:
        agent.close()


@app.command("eval")
def evaluate(
    ctx: typer.Context,
    mode: Annotated[
        str, typer.Option("--mode", help="offline | replay | record | live")
    ] = "replay",
    layer: Annotated[
        list[str] | None, typer.Option("--layer", help="tools | gate | agent (repeatable)")
    ] = None,
    case: Annotated[
        str | None, typer.Option("--case", help="Only case ids containing this.")
    ] = None,
    min_pass_rate: Annotated[
        float, typer.Option("--min-pass-rate", help="Exit 1 below this overall pass rate.")
    ] = 1.0,
) -> None:
    """Run the evaluation suite and write reports/eval_scorecard.md."""
    from fleet_copilot.evaluation.runner import run_evals

    settings: Settings = ctx.obj
    if mode not in ("offline", "replay", "record", "live"):
        console.print(f"[red]Error:[/red] unknown mode '{mode}'.")
        raise typer.Exit(code=2)
    layers = tuple(layer or ("tools", "gate", "agent"))
    run = _run_or_exit(lambda: run_evals(settings, mode, layers, case))  # type: ignore[arg-type]

    table = Table(title=f"Eval results ({mode})")
    for column in ("Layer / suite", "Cases", "Passed", "Failed", "Skipped"):
        table.add_column(column)
    for name, c in run.summary["by_suite"].items():
        table.add_row(name, *(str(c[k]) for k in ("cases", "passed", "failed", "skipped")))
    console.print(table)
    for result in run.results:
        if result.status in ("failed", "error"):
            problems = "; ".join(f"{c.name}: {c.detail}" for c in result.checks if not c.passed)
            label = f"[red]✗ {result.case_id}[/red] ({result.layer})"
            console.print(f"{label} {problems or result.note}")
    overall = run.summary["overall"]
    console.print(
        f"\n{overall['passed']} passed, {overall['failed']} failed, {overall['skipped']} skipped. "
        f"Scorecard: {run.scorecard_path}"
    )
    if overall["pass_rate"] is not None and overall["pass_rate"] < min_pass_rate:
        raise typer.Exit(code=1)


@app.command("eval-compare")
def eval_compare(ctx: typer.Context) -> None:
    """Compare the latest agent eval run of each provider/model: reports/eval_comparison.md."""
    from fleet_copilot.evaluation.comparison import (
        comparison_row,
        latest_per_model,
        load_runs,
        write_comparison,
    )

    settings: Settings = ctx.obj
    reports = settings.paths.reports_dir
    path = write_comparison(reports / "eval_runs", reports / "eval_comparison.md")
    runs = latest_per_model(load_runs(reports / "eval_runs"))
    if not runs:
        console.print(Markdown(path.read_text(encoding="utf-8")))
        return
    # Compact view for the terminal; the Markdown file has every metric.
    columns = ("Provider", "Model", "Mode", "Agent passed", "Pass rate", "Latency s")
    table = Table(title="Provider comparison")
    for column in columns:
        table.add_column(column)
    for run in runs:
        row = comparison_row(run)
        table.add_row(*(row[c] for c in columns))
    console.print(table)
    console.print(f"All metrics: {path}")


def _handle_turn(agent: FleetCopilotAgent, turn: AgentTurn) -> None:
    console.print(Markdown(turn.response or "_(no answer)_"))
    if not turn.awaiting_approval:
        return
    console.print("\n[bold yellow]Approval needed[/bold yellow] (nothing runs without it):")
    decisions = []
    for proposal in turn.pending_proposals:
        args = ", ".join(f"{k}={v}" for k, v in proposal["arguments"].items())
        console.print(f"\n• [bold]{proposal['action']}[/bold]({args})")
        console.print(f"  why: {proposal['justification']}")
        console.print(f"  evidence: {', '.join(proposal['evidence_ids'])}")
        approved = typer.confirm("  Approve?", default=False)
        decisions.append(ApprovalDecision(proposal["proposal_id"], approved))
    final = _run_or_exit(lambda: agent.decide(turn.thread_id, decisions))
    console.print(Markdown(final.response))


def _run_or_exit(action: Callable[[], T]) -> T:
    """Run `action`; turn expected app errors into a clean message and exit code 1."""
    try:
        return action()
    except FleetCopilotError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
