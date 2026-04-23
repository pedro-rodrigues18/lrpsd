"""Gerador de instancias para o LRPSD.

Baseado nos parametros do trabalho de Ribeiro et al. (2020) (drones Phantom
3 PRO e Inspire I, custo de estacao = $1000), com a diferenca de que o
problema considerado aqui inclui um limite maximo de rota L. As instancias
geradas seguem subconjuntos da Tabela I de Ribeiro et al.:

    5%  -> 12 criticos / 3  estacoes  (faceis)
    7%  -> 16 criticos / 4  estacoes  (medias)
    10% -> 23 criticos / 6  estacoes  (medias)
    15% -> 35 criticos / 9  estacoes  (dificeis)
    20% -> 46 criticos / 12 estacoes  (dificeis)

Como o trabalho original nao disponibilizou as coordenadas, os pontos sao
amostrados aleatoriamente em uma area quadrada com semente reprodutivel por
instancia. Cada instancia e validada para garantir que todo critico seja
alcancavel (diretamente ou via uma estacao candidata) respeitando bateria
Theta e limite de rota L; se a validacao falhar, novas seeds sao tentadas
automaticamente ate atingir o limite de tentativas.
"""

from __future__ import annotations

import argparse
import os
import random
from math import cos, hypot, pi, sin
from pathlib import Path

# === Parametros globais (Ribeiro et al. 2020) =======================
STATION_COST = 1000
GRID_ROUTE_FACTOR = 3.0  # L = GRID_ROUTE_FACTOR * grid_size

DRONES: dict[str, dict[str, int]] = {
    "Phantom": {"capacity": 17664, "cost": 1200, "beta": 1920},
    "Inspire": {"capacity": 19008, "cost": 2000, "beta": 1540},
}

# === Configuracoes das 25 instancias ================================
# Distribuicao: 5 faceis, 10 medias (5 com 16C/4R + 5 com 23C/6R) e
# 10 dificeis (5 com 35C/9R + 5 com 46C/12R). Cada grupo alterna Phantom
# e Inspire para cobrir os dois drones.
CONFIGURATIONS: list[dict] = [
    # ---------- Faceis (5) : 12 criticos / 3 estacoes -------------
    {"name": "Inst_01_Easy_Phantom",      "num_c": 12, "num_r": 3, "drone": "Phantom", "grid_size": 6000,  "seed": 101},
    {"name": "Inst_02_Easy_Inspire",      "num_c": 12, "num_r": 3, "drone": "Inspire", "grid_size": 6000,  "seed": 102},
    {"name": "Inst_03_Easy_Phantom",      "num_c": 12, "num_r": 3, "drone": "Phantom", "grid_size": 6500,  "seed": 103},
    {"name": "Inst_04_Easy_Inspire",      "num_c": 12, "num_r": 3, "drone": "Inspire", "grid_size": 6500,  "seed": 104},
    {"name": "Inst_05_Easy_Phantom",      "num_c": 12, "num_r": 3, "drone": "Phantom", "grid_size": 7000,  "seed": 105},

    # ---------- Medias (10) ---------------------------------------
    # 16 criticos / 4 estacoes
    {"name": "Inst_06_Medium_Phantom_16", "num_c": 16, "num_r": 4, "drone": "Phantom", "grid_size": 7000,  "seed": 206},
    {"name": "Inst_07_Medium_Inspire_16", "num_c": 16, "num_r": 4, "drone": "Inspire", "grid_size": 7000,  "seed": 207},
    {"name": "Inst_08_Medium_Phantom_16", "num_c": 16, "num_r": 4, "drone": "Phantom", "grid_size": 7500,  "seed": 208},
    {"name": "Inst_09_Medium_Inspire_16", "num_c": 16, "num_r": 4, "drone": "Inspire", "grid_size": 7500,  "seed": 209},
    {"name": "Inst_10_Medium_Phantom_16", "num_c": 16, "num_r": 4, "drone": "Phantom", "grid_size": 8000,  "seed": 210},
    # 23 criticos / 6 estacoes
    {"name": "Inst_11_Medium_Inspire_23", "num_c": 23, "num_r": 6, "drone": "Inspire", "grid_size": 8000,  "seed": 211},
    {"name": "Inst_12_Medium_Phantom_23", "num_c": 23, "num_r": 6, "drone": "Phantom", "grid_size": 8500,  "seed": 212},
    {"name": "Inst_13_Medium_Inspire_23", "num_c": 23, "num_r": 6, "drone": "Inspire", "grid_size": 8500,  "seed": 213},
    {"name": "Inst_14_Medium_Phantom_23", "num_c": 23, "num_r": 6, "drone": "Phantom", "grid_size": 9000,  "seed": 214},
    {"name": "Inst_15_Medium_Inspire_23", "num_c": 23, "num_r": 6, "drone": "Inspire", "grid_size": 9000,  "seed": 215},

    # ---------- Dificeis (10) -------------------------------------
    # 35 criticos / 9 estacoes
    {"name": "Inst_16_Hard_Phantom_35",   "num_c": 35, "num_r": 9, "drone": "Phantom", "grid_size": 9500,  "seed": 316},
    {"name": "Inst_17_Hard_Inspire_35",   "num_c": 35, "num_r": 9, "drone": "Inspire", "grid_size": 9500,  "seed": 317},
    {"name": "Inst_18_Hard_Phantom_35",   "num_c": 35, "num_r": 9, "drone": "Phantom", "grid_size": 10000, "seed": 318},
    {"name": "Inst_19_Hard_Inspire_35",   "num_c": 35, "num_r": 9, "drone": "Inspire", "grid_size": 10000, "seed": 319},
    {"name": "Inst_20_Hard_Phantom_35",   "num_c": 35, "num_r": 9, "drone": "Phantom", "grid_size": 10500, "seed": 320},
    # 46 criticos / 12 estacoes
    {"name": "Inst_21_Hard_Inspire_46",   "num_c": 46, "num_r": 12, "drone": "Inspire", "grid_size": 10500, "seed": 321},
    {"name": "Inst_22_Hard_Phantom_46",   "num_c": 46, "num_r": 12, "drone": "Phantom", "grid_size": 11000, "seed": 322},
    {"name": "Inst_23_Hard_Inspire_46",   "num_c": 46, "num_r": 12, "drone": "Inspire", "grid_size": 11000, "seed": 323},
    {"name": "Inst_24_Hard_Phantom_46",   "num_c": 46, "num_r": 12, "drone": "Phantom", "grid_size": 11500, "seed": 324},
    {"name": "Inst_25_Hard_Inspire_46",   "num_c": 46, "num_r": 12, "drone": "Inspire", "grid_size": 11500, "seed": 325},
]


# === Geracao de coordenadas ==========================================
def _sample_uniform_points(rng: random.Random, count: int, grid_size: float) -> list[tuple[float, float]]:
    return [
        (round(rng.uniform(0, grid_size), 1), round(rng.uniform(0, grid_size), 1))
        for _ in range(count)
    ]


# === Validacao de viabilidade ========================================
def _critical_reachable(
    critical: tuple[float, float],
    stations: list[tuple[float, float]],
    drone: dict[str, int],
    max_route_length: float,
) -> bool:
    """True se existe uma rota (direta ou via uma estacao) que visite o critico.

    Considera os dois padroes elementares: 0 -> c -> 0 (sem estacao) ou
    0 -> r -> c -> r -> 0 (uma unica estacao de recarga).
    """
    cx, cy = critical
    beta = drone["beta"]
    capacity = drone["capacity"]

    d_base_c = hypot(cx, cy)
    if (
        2 * d_base_c + beta <= capacity
        and 2 * d_base_c <= max_route_length
    ):
        return True

    for sx, sy in stations:
        d_base_s = hypot(sx, sy)
        d_s_c = hypot(sx - cx, sy - cy)
        if d_base_s > capacity:
            continue
        if 2 * d_s_c + beta > capacity:
            continue
        if 2 * d_base_s + 2 * d_s_c > max_route_length:
            continue
        return True

    return False


def _validate_instance(
    critical_points: list[tuple[float, float]],
    station_points: list[tuple[float, float]],
    drone: dict[str, int],
    max_route_length: float,
) -> list[tuple[float, float]]:
    return [
        c for c in critical_points
        if not _critical_reachable(c, station_points, drone, max_route_length)
    ]


# === Escrita do arquivo de instancia ================================
def _write_instance_file(
    name: str,
    drone_specs: dict[str, int],
    critical_points: list[tuple[float, float]],
    station_points: list[tuple[float, float]],
    output_dir: str | Path,
    station_cost: float,
    max_route_length: float,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / f"{name}.txt"
    dimension = 1 + len(critical_points) + len(station_points)

    with filename.open("w", encoding="utf-8") as handle:
        handle.write(f"NAME: {name}\n")
        handle.write("TYPE: UAV-COVERAGE-RECHARGE\n")
        handle.write(f"DIMENSION: {dimension}\n")
        handle.write(f"CRITICAL_LOCATIONS: {len(critical_points)}\n")
        handle.write(f"CANDIDATE_STATIONS: {len(station_points)}\n")
        handle.write(f"BATTERY_CAPACITY: {drone_specs['capacity']}\n")
        handle.write(f"MAX_ROUTE_LENGTH: {max_route_length}\n")
        handle.write(f"DRONE_COST: {drone_specs['cost']}\n")
        handle.write(f"STATION_COST: {station_cost}\n")
        handle.write("NODE_DATA_SECTION\n")
        handle.write("# ID\tTYPE\tX\t\tY\t\tBETA\n")
        handle.write("0\t0\t0.0\t\t0.0\t\t0\n")

        next_id = 1
        for x, y in critical_points:
            handle.write(f"{next_id}\t1\t{x}\t\t{y}\t\t{drone_specs['beta']}\n")
            next_id += 1
        for x, y in station_points:
            handle.write(f"{next_id}\t2\t{x}\t\t{y}\t\t0\n")
            next_id += 1

        handle.write("EOF\n")

    return filename


# === Geracao por configuracao =======================================
def _route_limit_for(config: dict) -> float:
    if "max_route_length" in config:
        return float(config["max_route_length"])
    return round(GRID_ROUTE_FACTOR * config["grid_size"], 1)


def generate_instance_file(
    config: dict,
    output_dir: str | Path = "instances",
    max_attempts: int = 25,
) -> dict:
    """Gera uma instancia validada e retorna metadados resumidos."""
    drone_specs = DRONES[config["drone"]]
    station_cost = config.get("station_cost", STATION_COST)
    max_route_length = _route_limit_for(config)
    base_seed = int(config["seed"])

    critical_points: list[tuple[float, float]] = []
    station_points: list[tuple[float, float]] = []
    unreachable: list[tuple[float, float]] = []
    seed_used: int | None = None

    for attempt in range(max_attempts):
        seed_attempt = base_seed + attempt * 1000
        rng = random.Random(seed_attempt)
        critical_points = _sample_uniform_points(rng, config["num_c"], config["grid_size"])
        station_points = _sample_uniform_points(rng, config["num_r"], config["grid_size"])
        unreachable = _validate_instance(
            critical_points, station_points, drone_specs, max_route_length
        )
        if not unreachable:
            seed_used = seed_attempt
            break
    else:
        raise RuntimeError(
            f"{config['name']}: nao foi possivel gerar instancia viavel apos "
            f"{max_attempts} tentativas (ultimo seed={seed_attempt}, "
            f"{len(unreachable)} criticos inalcansaveis)."
        )

    filename = _write_instance_file(
        config["name"],
        drone_specs,
        critical_points,
        station_points,
        output_dir,
        station_cost,
        max_route_length,
    )

    return {
        "name": config["name"],
        "drone": config["drone"],
        "num_c": config["num_c"],
        "num_r": config["num_r"],
        "grid_size": config["grid_size"],
        "max_route_length": max_route_length,
        "battery_capacity": drone_specs["capacity"],
        "seed_attempts": attempt + 1,
        "seed_used": seed_used,
        "filename": str(filename),
    }


# === Limpeza do diretorio ===========================================
def _clean_instances_dir(directory: Path) -> int:
    if not directory.exists():
        return 0
    removed = 0
    for path in sorted(directory.glob("*.txt")):
        path.unlink()
        removed += 1
    return removed


# === CLI =============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera 25 instancias (5 faceis, 10 medias, 10 dificeis) para o LRPSD.",
    )
    parser.add_argument(
        "--output-dir",
        default="instances",
        help="Diretorio de saida (default: instances/).",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Nao apaga arquivos .txt previos do diretorio antes de gerar.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=25,
        help="Tentativas por instancia para encontrar uma seed viavel (default: 25).",
    )
    return parser.parse_args()


def _print_summary(records: list[dict]) -> None:
    if not records:
        return

    header = (
        f"{'Instancia':30s} {'Drone':8s} {'|C|':>4s} {'|R|':>4s} "
        f"{'grid':>6s} {'L':>8s} {'Theta':>6s} {'tent.':>6s}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['name']:30s} {r['drone']:8s} {r['num_c']:>4d} {r['num_r']:>4d} "
            f"{r['grid_size']:>6d} {r['max_route_length']:>8.1f} "
            f"{r['battery_capacity']:>6d} {r['seed_attempts']:>6d}"
        )

    by_class: dict[str, int] = {"Easy": 0, "Medium": 0, "Hard": 0}
    for r in records:
        if "Easy" in r["name"]:
            by_class["Easy"] += 1
        elif "Medium" in r["name"]:
            by_class["Medium"] += 1
        elif "Hard" in r["name"]:
            by_class["Hard"] += 1
    print("\nResumo por categoria: " + ", ".join(f"{k}={v}" for k, v in by_class.items()))


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    if not args.keep_existing:
        removed = _clean_instances_dir(output_dir)
        if removed:
            print(f"Removidos {removed} arquivos .txt anteriores em '{output_dir}'.")

    print(f"Gerando {len(CONFIGURATIONS)} instancias em '{output_dir}/'...\n")

    records: list[dict] = []
    for cfg in CONFIGURATIONS:
        try:
            record = generate_instance_file(
                cfg, output_dir=output_dir, max_attempts=args.max_attempts
            )
        except RuntimeError as exc:
            print(f"  [ERRO] {exc}")
            continue
        records.append(record)
        suffix = (
            f" (seed={record['seed_used']}"
            f", tent={record['seed_attempts']})"
        )
        print(f"  -> {record['name']}.txt OK{suffix}")

    print()
    _print_summary(records)
    print(
        f"\nConcluido: {len(records)}/{len(CONFIGURATIONS)} instancias gravadas em '{output_dir}/'."
    )


if __name__ == "__main__":
    main()
