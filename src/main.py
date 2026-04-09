from __future__ import annotations

from pathlib import Path

from gurobipy import GRB

from gb_csc_model import solve_problem
from problem_reader import read_problem


INSTANCE_PATH = Path("instances/Inst_16_MultiDrone_Inspire.txt")


def main() -> None:
    problem = read_problem(INSTANCE_PATH)
    solution = solve_problem(problem, time_limit=300, debug_subtours=False)

    print(f"Instancia: {problem.name}")
    print(f"Nos: {problem.dimension}")
    print(f"Locais criticos: {len(problem.critical_nodes)}")
    print(f"Estacoes candidatas: {len(problem.station_nodes)}")
    print(f"Status Gurobi: {solution.status}")

    if (
        solution.status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
        or solution.objective_value is None
    ):
        print("Nenhuma solucao viavel foi encontrada.")
        return

    print(f"Objetivo: {solution.objective_value:.2f}")
    if solution.mip_gap is not None:
        print(f"MIP gap: {solution.mip_gap:.6f}")
    print(f"Drones utilizados: {solution.drone_count}")
    print(f"Estacoes ativadas: {list(solution.active_stations)}")
    print("Rotas:")
    for index, route in enumerate(solution.routes, start=1):
        print(f"  Drone {index}: {' -> '.join(map(str, route))}")


if __name__ == "__main__":
    main()
