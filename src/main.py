from __future__ import annotations

from pathlib import Path

from gurobipy import GRB

from gb_station_copies_model import solve_problem as solve_problem_station_copies
from gb_ribeiro_model import solve_problem as solve_problem_ribeiro
from gb_drone_index_model import solve_problem as solve_problem_drone_index
from problem_reader import read_problem


INSTANCE_PATH = Path("instances/Inst_02_Easy_Inspire.txt")


def main() -> None:
    problem = read_problem(INSTANCE_PATH)
    solution = solve_problem_drone_index(problem, time_limit=3600)

    print(f"Instancia: {problem.name}")
    print(f"Nos: {problem.dimension}")
    print(f"Locais criticos: {len(problem.critical_nodes)}")
    print(f"Estacoes candidatas: {len(problem.station_nodes)}")
    print(f"Limite maximo de percurso: {problem.max_route_length:.2f}")
    print(f"Status Gurobi: {solution.status}")

    if solution.objective_value is None:
        print("Nenhuma solucao viavel foi encontrada.")
        return

    if solution.status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
        print(
            "Aviso: a execucao terminou sem prova de otimalidade; abaixo segue a "
            "melhor solucao incumbente encontrada."
        )

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
