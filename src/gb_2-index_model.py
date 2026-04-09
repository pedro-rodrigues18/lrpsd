from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from problem_reader import BASE_NODE, ProblemData, read_problem


INSTANCE_PATH = (
    Path(__file__).resolve().parents[1] / "instances" / "Inst_01_Easy_Phantom.txt"
)


@dataclass(frozen=True)
class SolutionData:
    status: int
    objective_value: float | None
    mip_gap: float | None
    drone_count: int | None
    active_stations: tuple[int, ...]
    routes: tuple[tuple[int, ...], ...]


def _service_nodes(problem: ProblemData) -> tuple[int, ...]:
    return problem.critical_nodes + problem.station_nodes


def _route_arcs(problem: ProblemData) -> tuple[tuple[int, int], ...]:
    service_nodes = _service_nodes(problem)
    return tuple(
        (i, j)
        for i in problem.all_nodes
        for j in service_nodes
        if i != j
    )


def build_model(
    problem: ProblemData,
) -> tuple[
    gp.Model,
    gp.tupledict,
    gp.tupledict,
    gp.tupledict,
    gp.tupledict,
    gp.tupledict,
    gp.tupledict,
]:
    service_nodes = _service_nodes(problem)
    route_arcs = _route_arcs(problem)
    battery_capacity = problem.battery_capacity
    total_nodes = len(problem.all_nodes)

    model = gp.Model(f"lrpsd_ribeiro_{problem.name}")

    x = model.addVars(route_arcs, vtype=GRB.BINARY, name="x")
    y = model.addVars(problem.station_nodes, vtype=GRB.BINARY, name="y")
    z = model.addVars(problem.critical_nodes, vtype=GRB.BINARY, name="z")
    w = model.addVars(
        problem.station_nodes,
        vtype=GRB.INTEGER,
        lb=0,
        ub=len(service_nodes),
        name="w",
    )
    b = model.addVars(
        problem.all_nodes,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        ub=battery_capacity,
        name="b",
    )
    t = model.addVars(
        problem.all_nodes,
        vtype=GRB.INTEGER,
        lb=0,
        ub=total_nodes,
        name="t",
    )

    model.setObjective(
        gp.quicksum(problem.alpha[i, j] * x[i, j] for i, j in route_arcs)
        + gp.quicksum(
            (problem.drone_cost + problem.alpha[c, BASE_NODE]) * z[c]
            for c in problem.critical_nodes
        )
        + gp.quicksum(
            (problem.drone_cost + problem.alpha[r, BASE_NODE]) * w[r]
            for r in problem.station_nodes
        )
        + problem.station_cost * gp.quicksum(y[r] for r in problem.station_nodes),
        GRB.MINIMIZE,
    )

    for c in problem.critical_nodes:
        model.addConstr(
            gp.quicksum(x[i, c] for i in problem.all_nodes if i != c) == 1,
            name=f"critical_in_{c}",
        )
        model.addConstr(
            gp.quicksum(x[c, j] for j in service_nodes if j != c) == 1 - z[c],
            name=f"critical_out_{c}",
        )

    for r in problem.station_nodes:
        model.addConstr(
            gp.quicksum(x[r, j] for j in service_nodes if j != r) + w[r]
            == gp.quicksum(x[i, r] for i in problem.all_nodes if i != r),
            name=f"station_balance_{r}",
        )

    model.addConstr(
        gp.quicksum(x[BASE_NODE, j] for j in service_nodes)
        == gp.quicksum(z[c] for c in problem.critical_nodes)
        + gp.quicksum(w[r] for r in problem.station_nodes),
        name="route_count",
    )

    for r in problem.station_nodes:
        for i in problem.all_nodes:
            if i == r:
                continue
            model.addConstr(
                x[i, r] <= y[r],
                name=f"station_in_activation_{i}_{r}",
            )

        for j in service_nodes:
            if j == r:
                continue
            model.addConstr(
                x[r, j] <= y[r],
                name=f"station_out_activation_{r}_{j}",
            )

    model.addConstr(
        gp.quicksum(x[i, j] for i, j in route_arcs)
        == len(problem.critical_nodes)
        + gp.quicksum(
            x[i, r]
            for r in problem.station_nodes
            for i in problem.all_nodes
            if i != r
        ),
        name="edge_count",
    )

    model.addConstr(b[BASE_NODE] == battery_capacity, name="base_battery")

    for r in problem.station_nodes:
        model.addConstr(
            b[r] == battery_capacity * y[r],
            name=f"station_battery_{r}",
        )

    for i in problem.all_nodes:
        for c in problem.critical_nodes:
            if i == c:
                continue
            model.addConstr(
                b[c]
                <= b[i]
                - (problem.alpha[i, c] + problem.beta[c]) * x[i, c]
                + battery_capacity * (1 - x[i, c]),
                name=f"critical_energy_{i}_{c}",
            )

    for i in problem.all_nodes:
        for r in problem.station_nodes:
            if i == r:
                continue
            model.addConstr(
                b[i]
                - problem.alpha[i, r] * x[i, r]
                + battery_capacity * (1 - x[i, r])
                >= 0,
                name=f"station_energy_{i}_{r}",
            )

    for c in problem.critical_nodes:
        model.addConstr(
            b[c]
            - problem.alpha[c, BASE_NODE] * z[c]
            + battery_capacity * (1 - z[c])
            >= 0,
            name=f"return_energy_{c}",
        )

    model.addConstr(t[BASE_NODE] == total_nodes, name="root_level")

    for node in service_nodes:
        model.addConstr(
            t[node] <= total_nodes - 1,
            name=f"level_upper_{node}",
        )

    for i in problem.all_nodes:
        for j in service_nodes:
            if i == j:
                continue
            model.addConstr(
                t[j] <= t[i] - x[i, j] + total_nodes * (1 - x[i, j]),
                name=f"radial_{i}_{j}",
            )

    return model, x, y, z, w, b, t


def _extract_routes(
    problem: ProblemData,
    x_values: dict[tuple[int, int], float],
    z_values: dict[int, float],
    w_values: dict[int, float],
) -> tuple[tuple[int, ...], ...]:
    service_nodes = _service_nodes(problem)
    remaining_edges = Counter(
        {
            arc: int(round(value))
            for arc, value in x_values.items()
            if int(round(value)) > 0
        }
    )
    remaining_critical_ends = Counter(
        {
            node: int(round(z_values[node]))
            for node in problem.critical_nodes
            if int(round(z_values[node])) > 0
        }
    )
    remaining_station_ends = Counter(
        {
            node: int(round(w_values[node]))
            for node in problem.station_nodes
            if int(round(w_values[node])) > 0
        }
    )

    route_count = int(
        round(
            sum(
                x_values[BASE_NODE, j]
                for j in service_nodes
                if (BASE_NODE, j) in x_values
            )
        )
    )
    routes: list[tuple[int, ...]] = []

    for _ in range(route_count):
        current = BASE_NODE
        route = [BASE_NODE]

        while True:
            if current in problem.critical_nodes and remaining_critical_ends[current] > 0:
                remaining_critical_ends[current] -= 1
                if remaining_critical_ends[current] == 0:
                    del remaining_critical_ends[current]
                break

            if current in problem.station_nodes and remaining_station_ends[current] > 0:
                remaining_station_ends[current] -= 1
                if remaining_station_ends[current] == 0:
                    del remaining_station_ends[current]
                break

            next_candidates = sorted(
                j
                for j in service_nodes
                if current != j and remaining_edges[current, j] > 0
            )

            if not next_candidates:
                raise RuntimeError(
                    "Nao foi possivel reconstruir uma rota radial completa a partir da solucao."
                )

            next_node = next_candidates[0]
            remaining_edges[current, next_node] -= 1
            if remaining_edges[current, next_node] == 0:
                del remaining_edges[current, next_node]

            current = next_node
            route.append(current)

        route.append(BASE_NODE)
        routes.append(tuple(route))

    if remaining_edges or remaining_critical_ends or remaining_station_ends:
        raise RuntimeError(
            "Restaram componentes nao utilizados ao reconstruir as rotas do modelo 2-index."
        )

    return tuple(routes)


def solve_problem(
    problem: ProblemData, time_limit: float | None = None
) -> SolutionData:
    model, x, y, z, w, _, _ = build_model(problem)

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    model.optimize()

    if model.SolCount == 0:
        return SolutionData(
            status=model.Status,
            objective_value=None,
            mip_gap=None,
            drone_count=None,
            active_stations=(),
            routes=(),
        )

    x_values = model.getAttr("X", x)
    y_values = model.getAttr("X", y)
    z_values = model.getAttr("X", z)
    w_values = model.getAttr("X", w)

    service_nodes = _service_nodes(problem)
    drone_count = int(
        round(
            sum(
                x_values[BASE_NODE, j]
                for j in service_nodes
                if (BASE_NODE, j) in x_values
            )
        )
    )
    active_stations = tuple(
        sorted(r for r in problem.station_nodes if y_values[r] > 0.5)
    )
    routes = _extract_routes(problem, x_values, z_values, w_values)
    mip_gap = model.MIPGap if model.IsMIP else 0.0

    return SolutionData(
        status=model.Status,
        objective_value=model.ObjVal,
        mip_gap=mip_gap,
        drone_count=drone_count,
        active_stations=active_stations,
        routes=routes,
    )


def main() -> None:
    problem = read_problem(INSTANCE_PATH)
    solution = solve_problem(problem)

    print(f"Instancia: {problem.name}")
    print("Modelo: Ribeiro et al. 2-index")
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
