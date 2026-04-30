"""Runner para experimentos com os tres modelos do LRPSD.

Executa cada modelo (Ribeiro, Indice de Drones, Duplicacao de Nos) sobre cada
instancia indicada, salvando logs do Gurobi e metricas agregadas de forma
incremental para que nenhuma execucao seja perdida em caso de falha.

Saida (em ``experiments/<timestamp>[_<tag>]/``):

* ``config.json`` -- parametros usados na execucao.
* ``logs/<instancia>__<modelo>.log`` -- log completo do Gurobi.
* ``runs/<instancia>__<modelo>.json`` -- metricas detalhadas por execucao.
* ``summary.csv`` -- uma linha por (instancia, modelo) com todas as metricas
  necessarias para construir tabelas do artigo.
* ``aggregate.csv`` -- agregados por modelo (medias, taxa de otimo, etc.).
* ``summary_table.md`` -- tabela em Markdown pronta para inspecao rapida.
* ``errors.log`` -- erros e excecoes capturados durante a execucao.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parent))

from problem_reader import read_problem


STATUS_LABELS: dict[int, str] = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.INPROGRESS: "INPROGRESS",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}


def _status_label(code: int | None) -> str:
    if code is None:
        return "NO_MODEL"
    return STATUS_LABELS.get(code, f"UNKNOWN_{code}")


def _parse_gurobi_param_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return 1 if lowered == "true" else 0
    try:
        if any(token in raw for token in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _parse_gurobi_params(raw_params: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in raw_params:
        if "=" not in item:
            raise ValueError(f"Parametro Gurobi invalido: '{item}'. Use formato Nome=Valor.")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Parametro Gurobi invalido: '{item}'. Nome vazio.")
        params[name] = _parse_gurobi_param_value(value.strip())
    return params


def _build_and_run_ribeiro(
    problem,
    log_file: Path,
    time_limit: float | None,
    gurobi_params: dict[str, Any],
    run_stats: dict[str, Any] | None = None,
) -> gp.Model:
    from gb_ribeiro_model import build_model

    model = build_model(problem)[0]
    return _configure_and_optimize(model, log_file, time_limit, gurobi_params, run_stats=run_stats)


def _build_and_run_drone_index(
    problem,
    log_file: Path,
    time_limit: float | None,
    gurobi_params: dict[str, Any],
    run_stats: dict[str, Any] | None = None,
) -> gp.Model:
    from gb_drone_index_model import _subtour_callback, build_model

    model = build_model(problem)[0]
    return _configure_and_optimize(
        model,
        log_file,
        time_limit,
        gurobi_params,
        callback=_subtour_callback,
        run_stats=run_stats,
    )


def _build_and_run_station_copies(
    problem,
    log_file: Path,
    time_limit: float | None,
    gurobi_params: dict[str, Any],
    run_stats: dict[str, Any] | None = None,
) -> gp.Model:
    from gb_station_copies_model import build_model

    model = build_model(problem)[0]
    return _configure_and_optimize(model, log_file, time_limit, gurobi_params, run_stats=run_stats)


def _configure_and_optimize(
    model: gp.Model,
    log_file: Path,
    time_limit: float | None,
    gurobi_params: dict[str, Any],
    callback: Callable | None = None,
    run_stats: dict[str, Any] | None = None,
) -> gp.Model:
    model.Params.LogFile = str(log_file)
    model.Params.LogToConsole = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    for name, value in gurobi_params.items():
        model.setParam(name, value)

    def _tracking_callback(cb_model: gp.Model, where: int) -> None:
        if run_stats is not None and where == GRB.Callback.MIPSOL:
            if run_stats.get("first_incumbent_time_seconds") is None:
                runtime = cb_model.cbGet(GRB.Callback.RUNTIME)
                run_stats["first_incumbent_time_seconds"] = runtime
        if callback is not None:
            callback(cb_model, where)

    if callback is None:
        if run_stats is None:
            model.optimize()
        else:
            model.optimize(_tracking_callback)
    else:
        model.optimize(_tracking_callback)
    return model


SOLVERS: dict[str, Callable] = {
    "ribeiro": _build_and_run_ribeiro,
    "drone_index": _build_and_run_drone_index,
    "station_copies": _build_and_run_station_copies,
}


_ROOT_LP_RE = re.compile(
    r"Root relaxation:\s+objective\s+([-\d.eE+]+),\s+(\d+)\s+iterations,\s+([\d.]+)\s+seconds"
)
_EXPLORED_RE = re.compile(
    r"Explored\s+(\d+)\s+nodes?\s+\((\d+)\s+simplex iterations\)\s+in\s+([\d.]+)\s+seconds"
)
_PRESOLVE_RM_RE = re.compile(
    r"Presolve removed\s+(\d+)\s+rows? and\s+(\d+)\s+columns?"
)
_PRESOLVE_TIME_RE = re.compile(r"Presolve time:\s*([\d.]+)s")
_CUTS_HEADER_RE = re.compile(r"^Cutting planes:\s*$")
_CUTS_LINE_RE = re.compile(r"^\s+([A-Za-z][A-Za-z0-9 \-]*?):\s*(\d+)\s*$")
_LAZY_HINT_RE = re.compile(r"^\s*Lazy constraints?:\s*(\d+)\s*$")


def parse_gurobi_log(log_path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "root_relaxation_objective": None,
        "root_relaxation_iterations": None,
        "root_relaxation_seconds": None,
        "explored_nodes_log": None,
        "presolve_rows_removed": None,
        "presolve_cols_removed": None,
        "presolve_time_seconds": None,
        "cuts_breakdown": {},
        "cuts_total": 0,
    }
    if not log_path.exists():
        return data

    in_cuts = False
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            match = _ROOT_LP_RE.search(line)
            if match:
                data["root_relaxation_objective"] = float(match.group(1))
                data["root_relaxation_iterations"] = int(match.group(2))
                data["root_relaxation_seconds"] = float(match.group(3))
                continue

            match = _EXPLORED_RE.search(line)
            if match:
                data["explored_nodes_log"] = int(match.group(1))
                continue

            match = _PRESOLVE_RM_RE.search(line)
            if match:
                data["presolve_rows_removed"] = int(match.group(1))
                data["presolve_cols_removed"] = int(match.group(2))
                continue

            match = _PRESOLVE_TIME_RE.search(line)
            if match:
                data["presolve_time_seconds"] = float(match.group(1))
                continue

            match = _LAZY_HINT_RE.match(line)
            if match:
                count = int(match.group(1))
                data["cuts_breakdown"]["Lazy"] = count
                data["cuts_total"] += count
                continue

            if _CUTS_HEADER_RE.match(line):
                in_cuts = True
                continue

            if in_cuts:
                match = _CUTS_LINE_RE.match(line)
                if match:
                    name = match.group(1).strip()
                    count = int(match.group(2))
                    data["cuts_breakdown"][name] = (
                        data["cuts_breakdown"].get(name, 0) + count
                    )
                    data["cuts_total"] += count
                else:
                    in_cuts = False

    return data


def collect_model_metrics(model: gp.Model | None) -> dict[str, Any]:
    if model is None:
        return {
            "status": None,
            "status_label": "NO_MODEL",
            "runtime_seconds": None,
            "objective": None,
            "best_bound": None,
            "gap": None,
            "node_count": None,
            "simplex_iterations": None,
            "barrier_iterations": None,
            "num_vars": None,
            "num_bin_vars": None,
            "num_int_vars": None,
            "num_cont_vars": None,
            "num_constrs": None,
            "sol_count": 0,
            "is_mip": None,
            "first_incumbent_time_seconds": None,
        }

    def safe(attr: str, default: Any = None) -> Any:
        try:
            return model.getAttr(attr)
        except Exception:  # noqa: BLE001
            return default

    sol_count = safe("SolCount", 0) or 0
    status = safe("Status")
    objective = safe("ObjVal") if sol_count > 0 else None
    is_mip = bool(safe("IsMIP", 0))
    best_bound = safe("ObjBound") if is_mip else None
    gap = safe("MIPGap") if (is_mip and sol_count > 0) else None
    num_vars = safe("NumVars", 0) or 0
    num_int = safe("NumIntVars", 0) or 0

    return {
        "status": status,
        "status_label": _status_label(status),
        "runtime_seconds": safe("Runtime"),
        "objective": objective,
        "best_bound": best_bound,
        "gap": gap,
        "node_count": safe("NodeCount"),
        "simplex_iterations": safe("IterCount"),
        "barrier_iterations": safe("BarIterCount"),
        "num_vars": num_vars,
        "num_bin_vars": safe("NumBinVars"),
        "num_int_vars": num_int,
        "num_cont_vars": num_vars - num_int,
        "num_constrs": safe("NumConstrs"),
        "sol_count": sol_count,
        "is_mip": is_mip,
        "first_incumbent_time_seconds": None,
    }


CSV_FIELDS: list[str] = [
    "instance",
    "model",
    "status",
    "status_label",
    "objective",
    "best_bound",
    "gap",
    "sol_count",
    "runtime_seconds",
    "first_incumbent_time_seconds",
    "wall_seconds",
    "node_count",
    "explored_nodes_log",
    "simplex_iterations",
    "barrier_iterations",
    "root_relaxation_objective",
    "root_relaxation_iterations",
    "root_relaxation_seconds",
    "presolve_time_seconds",
    "presolve_rows_removed",
    "presolve_cols_removed",
    "cuts_total",
    "cuts_breakdown",
    "num_vars",
    "num_bin_vars",
    "num_int_vars",
    "num_cont_vars",
    "num_constrs",
    "is_mip",
    "error",
]


def _flush_row(
    writer: csv.DictWriter,
    csv_handle,
    json_path: Path,
    row: dict[str, Any],
) -> None:
    sanitized = {field: row.get(field) for field in CSV_FIELDS}
    writer.writerow(sanitized)
    csv_handle.flush()
    json_path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")


def _append_error(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")
        handle.flush()


def _load_existing_keys(summary_csv: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not summary_csv.exists():
        return keys
    with summary_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            instance = record.get("instance")
            model = record.get("model")
            if instance and model:
                keys.add((instance, model))
    return keys


def _write_aggregate(summary_csv: Path, aggregate_csv: Path) -> None:
    if not summary_csv.exists():
        return
    with summary_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    fields = [
        "model",
        "runs",
        "solved_optimal",
        "solved_with_incumbent",
        "avg_objective",
        "avg_best_bound",
        "avg_gap",
        "avg_runtime_seconds",
        "avg_node_count",
        "avg_root_relaxation_seconds",
        "avg_root_relaxation_iterations",
        "avg_cuts_total",
        "avg_num_vars",
        "avg_num_constrs",
    ]

    def _to_float(value: str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _avg(values: list[float | None]) -> float | None:
        clean = [v for v in values if v is not None]
        return statistics.fmean(clean) if clean else None

    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, model_rows in sorted(by_model.items()):
            writer.writerow(
                {
                    "model": model,
                    "runs": len(model_rows),
                    "solved_optimal": sum(1 for r in model_rows if r.get("status_label") == "OPTIMAL"),
                    "solved_with_incumbent": sum(
                        1 for r in model_rows if (_to_float(r.get("sol_count")) or 0) > 0
                    ),
                    "avg_objective": _avg([_to_float(r.get("objective")) for r in model_rows]),
                    "avg_best_bound": _avg([_to_float(r.get("best_bound")) for r in model_rows]),
                    "avg_gap": _avg([_to_float(r.get("gap")) for r in model_rows]),
                    "avg_runtime_seconds": _avg(
                        [_to_float(r.get("runtime_seconds")) for r in model_rows]
                    ),
                    "avg_node_count": _avg([_to_float(r.get("node_count")) for r in model_rows]),
                    "avg_root_relaxation_seconds": _avg(
                        [_to_float(r.get("root_relaxation_seconds")) for r in model_rows]
                    ),
                    "avg_root_relaxation_iterations": _avg(
                        [_to_float(r.get("root_relaxation_iterations")) for r in model_rows]
                    ),
                    "avg_cuts_total": _avg([_to_float(r.get("cuts_total")) for r in model_rows]),
                    "avg_num_vars": _avg([_to_float(r.get("num_vars")) for r in model_rows]),
                    "avg_num_constrs": _avg([_to_float(r.get("num_constrs")) for r in model_rows]),
                }
            )


def _write_markdown_summary(summary_csv: Path, markdown_path: Path) -> None:
    if not summary_csv.exists():
        return
    with summary_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    headers = [
        "instance",
        "model",
        "status_label",
        "objective",
        "best_bound",
        "gap",
        "runtime_seconds",
        "node_count",
        "cuts_total",
        "root_relaxation_seconds",
    ]

    def _fmt(value: str | None) -> str:
        if value is None or value == "":
            return "-"
        try:
            number = float(value)
        except ValueError:
            return str(value)
        if abs(number) >= 1000 or number == int(number):
            return f"{number:.2f}"
        return f"{number:.4f}"

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in sorted(rows, key=lambda r: (r.get("instance", ""), r.get("model", ""))):
        lines.append(
            "| "
            + " | ".join(_fmt(row.get(col)) if col not in {"instance", "model", "status_label"} else (row.get(col) or "-") for col in headers)
            + " |"
        )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--instances-dir",
        default="instances",
        help="Diretorio com arquivos .txt de instancias (default: instances/).",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments",
        help="Diretorio raiz para os experimentos (default: experiments/).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="Limite de tempo por execucao em segundos (default: 300).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SOLVERS.keys()),
        choices=list(SOLVERS.keys()),
        help="Modelos a serem executados (default: todos).",
    )
    parser.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Nomes (basenames) das instancias para rodar. Se omitido, usa todas as .txt do diretorio.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Sufixo opcional para o nome do diretorio de saida.",
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Reaproveita um diretorio existente, pulando execucoes ja registradas no summary.csv.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Mesmo com --resume-dir, refaz todas as execucoes.",
    )
    parser.add_argument(
        "--disable-general-cuts",
        action="store_true",
        help="Desativa cortes gerais do Gurobi (equivale a Cuts=0).",
    )
    parser.add_argument(
        "--gurobi-param",
        action="append",
        default=[],
        help="Parametro extra do Gurobi no formato Nome=Valor (pode repetir).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gurobi_params = _parse_gurobi_params(args.gurobi_param)
    if args.disable_general_cuts:
        gurobi_params["Cuts"] = 0

    instances_dir = Path(args.instances_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.resume_dir is not None:
        run_dir = Path(args.resume_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{args.tag}" if args.tag else timestamp
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = run_dir / "logs"
    runs_dir = run_dir / "runs"
    logs_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(exist_ok=True)

    summary_csv = run_dir / "summary.csv"
    aggregate_csv = run_dir / "aggregate.csv"
    markdown_path = run_dir / "summary_table.md"
    errors_log = run_dir / "errors.log"

    if args.instances:
        instance_paths = [instances_dir / name for name in args.instances]
    else:
        instance_paths = sorted(instances_dir.glob("*.txt"))

    config = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "time_limit": args.time_limit,
        "models": args.models,
        "instances_dir": str(instances_dir.resolve()),
        "instances": [path.name for path in instance_paths],
        "gurobi_params": gurobi_params,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    existing_keys: set[tuple[str, str]] = set()
    if not args.no_resume:
        existing_keys = _load_existing_keys(summary_csv)

    csv_exists = summary_csv.exists()
    csv_handle = summary_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
    if not csv_exists:
        writer.writeheader()
        csv_handle.flush()

    total_runs = len(instance_paths) * len(args.models)
    print(f"[runner] saving experiments to {run_dir}")
    print(
        f"[runner] {len(instance_paths)} instances x {len(args.models)} models = "
        f"{total_runs} runs (time_limit={args.time_limit}s)"
    )
    if existing_keys:
        print(f"[runner] resume mode: {len(existing_keys)} previous runs will be skipped")

    try:
        for instance_path in instance_paths:
            instance_name = instance_path.stem
            try:
                problem = read_problem(instance_path)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                _append_error(errors_log, f"{instance_name} READ_ERROR:\n{tb}\n")
                print(f"[runner] failed to read {instance_path}: {exc}", file=sys.stderr)
                continue

            for model_name in args.models:
                if (instance_name, model_name) in existing_keys:
                    print(f"[runner] skip {instance_name}__{model_name} (already in summary)")
                    continue

                run_id = f"{instance_name}__{model_name}"
                log_file = logs_dir / f"{run_id}.log"
                json_file = runs_dir / f"{run_id}.json"

                row: dict[str, Any] = {field: None for field in CSV_FIELDS}
                row["instance"] = instance_name
                row["model"] = model_name

                print(f"[runner] >>> {run_id}")
                wall_start = time.time()
                model: gp.Model | None = None
                run_stats: dict[str, Any] = {"first_incumbent_time_seconds": None}
                try:
                    solver = SOLVERS[model_name]
                    model = solver(
                        problem,
                        log_file,
                        args.time_limit,
                        gurobi_params,
                        run_stats=run_stats,
                    )
                except KeyboardInterrupt:
                    row["error"] = "KeyboardInterrupt"
                    metrics = collect_model_metrics(model)
                    row.update(metrics)
                    row["first_incumbent_time_seconds"] = run_stats["first_incumbent_time_seconds"]
                    row["wall_seconds"] = round(time.time() - wall_start, 3)
                    log_data = parse_gurobi_log(log_file)
                    row["root_relaxation_objective"] = log_data["root_relaxation_objective"]
                    row["root_relaxation_iterations"] = log_data["root_relaxation_iterations"]
                    row["root_relaxation_seconds"] = log_data["root_relaxation_seconds"]
                    row["explored_nodes_log"] = log_data["explored_nodes_log"]
                    row["presolve_rows_removed"] = log_data["presolve_rows_removed"]
                    row["presolve_cols_removed"] = log_data["presolve_cols_removed"]
                    row["presolve_time_seconds"] = log_data["presolve_time_seconds"]
                    row["cuts_total"] = log_data["cuts_total"]
                    row["cuts_breakdown"] = json.dumps(log_data["cuts_breakdown"])
                    _flush_row(writer, csv_handle, json_file, row)
                    _append_error(errors_log, f"{run_id}: KeyboardInterrupt")
                    raise
                except Exception as exc:  # noqa: BLE001
                    tb = traceback.format_exc()
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    _append_error(errors_log, f"{run_id} EXCEPTION:\n{tb}\n")
                    print(f"[runner] error in {run_id}: {exc}", file=sys.stderr)

                metrics = collect_model_metrics(model)
                row.update(metrics)
                row["first_incumbent_time_seconds"] = run_stats["first_incumbent_time_seconds"]
                row["wall_seconds"] = round(time.time() - wall_start, 3)

                log_data = parse_gurobi_log(log_file)
                row["root_relaxation_objective"] = log_data["root_relaxation_objective"]
                row["root_relaxation_iterations"] = log_data["root_relaxation_iterations"]
                row["root_relaxation_seconds"] = log_data["root_relaxation_seconds"]
                row["explored_nodes_log"] = log_data["explored_nodes_log"]
                row["presolve_rows_removed"] = log_data["presolve_rows_removed"]
                row["presolve_cols_removed"] = log_data["presolve_cols_removed"]
                row["presolve_time_seconds"] = log_data["presolve_time_seconds"]
                row["cuts_total"] = log_data["cuts_total"]
                row["cuts_breakdown"] = json.dumps(log_data["cuts_breakdown"])

                _flush_row(writer, csv_handle, json_file, row)

                if model is not None:
                    try:
                        model.dispose()
                    except Exception:  # noqa: BLE001
                        pass

                _write_aggregate(summary_csv, aggregate_csv)
                _write_markdown_summary(summary_csv, markdown_path)
    finally:
        csv_handle.close()
        _write_aggregate(summary_csv, aggregate_csv)
        _write_markdown_summary(summary_csv, markdown_path)
        print(f"[runner] done. summary: {summary_csv}")


if __name__ == "__main__":
    main()
