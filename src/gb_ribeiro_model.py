from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from problem_reader import BASE_NODE, ProblemData


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
    arcs: list[tuple[int, int]] = []

    for i in problem.all_nodes:
        for j in service_nodes:
            if i == j:
                continue

            travel_cost = problem.alpha[i, j]
            if travel_cost > problem.max_route_length + 1e-9:
                continue

            required_energy = travel_cost + (
                problem.beta[j] if j in problem.critical_nodes else 0.0
            )
            if required_energy > problem.battery_capacity + 1e-9:
                continue

            arcs.append((i, j))

    return tuple(arcs)


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
    gp.tupledict,
    gp.tupledict,
]:
    service_nodes = _service_nodes(problem)
    route_arcs = _route_arcs(problem)
    battery_capacity = problem.battery_capacity
    route_limit = problem.max_route_length
    total_nodes = len(problem.all_nodes)

    incoming: dict[int, tuple[int, ...]] = {node: tuple() for node in service_nodes}
    outgoing: dict[int, tuple[int, ...]] = {node: tuple() for node in problem.all_nodes}
    incoming_lists: dict[int, list[int]] = {node: [] for node in service_nodes}
    outgoing_lists: dict[int, list[int]] = {node: [] for node in problem.all_nodes}

    for i, j in route_arcs:
        outgoing_lists[i].append(j)
        incoming_lists[j].append(i)

    for node in service_nodes:
        incoming[node] = tuple(sorted(incoming_lists[node]))
    for node in problem.all_nodes:
        outgoing[node] = tuple(sorted(outgoing_lists[node]))

    model = gp.Model(f"lrpsd_ribeiro_{problem.name}")

    x = model.addVars(route_arcs, vtype=GRB.BINARY, name="x")
    y = model.addVars(problem.station_nodes, vtype=GRB.BINARY, name="y")
    z = model.addVars(
        problem.critical_nodes, vtype=GRB.BINARY, name="z"
    )  # indica se o nó crítico é o último atendido da rota
    u = model.addVars(problem.station_nodes, vtype=GRB.BINARY, name="u")
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
    l = model.addVars(
        problem.all_nodes,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        ub=route_limit,
        name="l",
    )

    # model.setObjective(
    #     gp.quicksum(problem.alpha[i, j] * x[i, j] for i, j in route_arcs)
    #     + gp.quicksum(
    #         (problem.drone_cost + problem.alpha[c, BASE_NODE]) * z[c]
    #         for c in problem.critical_nodes
    #     )
    #     + gp.quicksum(
    #         (problem.drone_cost + problem.alpha[r, BASE_NODE]) * w[r]
    #         for r in problem.station_nodes
    #     )
    #     + problem.station_cost * gp.quicksum(y[r] for r in problem.station_nodes),
    #     GRB.MINIMIZE,
    # )

    model.setObjective(
        problem.drone_cost
        * gp.quicksum(x[BASE_NODE, j] for j in service_nodes if (BASE_NODE, j) in x)
        + problem.station_cost * gp.quicksum(y[r] for r in problem.station_nodes),
        GRB.MINIMIZE,
    )

    # Garantir que cada nó crítico seja visitado exatamente uma vez
    for critical in problem.critical_nodes:
        model.addConstr(
            gp.quicksum(x[i, critical] for i in incoming[critical]) == 1,
            name=f"critical_in_{critical}",
        )
        # Se o nó crítico for o último da rota, ele não pode ter saída para outro nó que não seja a base {0}; caso contrário, deve ter exatamente uma saída
        model.addConstr(
            gp.quicksum(x[critical, j] for j in outgoing[critical]) == 1 - z[critical],
            name=f"critical_out_{critical}",
        )

    # Conservação de fluxo para as estações
    for station in problem.station_nodes:
        model.addConstr(
            gp.quicksum(x[station, j] for j in outgoing[station]) + w[station]
            == gp.quicksum(x[i, station] for i in incoming[station]),
            name=f"station_balance_{station}",
        )

    model.addConstr(
        gp.quicksum(x[BASE_NODE, j] for j in outgoing[BASE_NODE])
        == gp.quicksum(z[critical] for critical in problem.critical_nodes)
        + gp.quicksum(w[station] for station in problem.station_nodes),
        name="route_count",
    )

    # Garantir que as estações só possam ser ativadas se tiverem pelo menos uma entrada ou saída
    for station in problem.station_nodes:
        for i in incoming[station]:
            model.addConstr(
                x[i, station] <= y[station],
                name=f"station_in_activation_{i}_{station}",
            )

        for j in outgoing[station]:
            model.addConstr(
                x[station, j] <= y[station],
                name=f"station_out_activation_{station}_{j}",
            )

    # model.addConstr(
    #     gp.quicksum(x[i, j] for i, j in route_arcs)
    #     == len(problem.critical_nodes)
    #     + gp.quicksum(
    #         x[i, station]
    #         for station in problem.station_nodes
    #         for i in incoming[station]
    #     ),
    #     name="edge_count",
    # )

    # Restrições de energia
    model.addConstr(b[BASE_NODE] == battery_capacity, name="base_battery")

    for station in problem.station_nodes:
        model.addConstr(
            b[station] == battery_capacity * y[station],
            name=f"station_battery_{station}",
        )

    for i in problem.all_nodes:
        for critical in problem.critical_nodes:
            if (i, critical) not in x:
                continue
            model.addConstr(
                b[critical]
                <= b[i]
                - (problem.alpha[i, critical] + problem.beta[critical]) * x[i, critical]
                + battery_capacity * (1 - x[i, critical]),
                name=f"critical_energy_{i}_{critical}",
            )

    for i in problem.all_nodes:
        for station in problem.station_nodes:
            if (i, station) not in x:
                continue
            model.addConstr(
                b[i]
                - problem.alpha[i, station] * x[i, station]
                + battery_capacity * (1 - x[i, station])
                >= 0,
                name=f"station_energy_{i}_{station}",
            )

    for critical in problem.critical_nodes:
        model.addConstr(
            b[critical]
            - problem.alpha[critical, BASE_NODE] * z[critical]
            + battery_capacity * (1 - z[critical])
            >= 0,
            name=f"return_energy_{critical}",
        )

    # Restrições de ordenação radial
    model.addConstr(t[BASE_NODE] == total_nodes, name="root_level")

    for node in service_nodes:
        model.addConstr(t[node] <= total_nodes - 1, name=f"level_upper_{node}")

    for i, j in route_arcs:
        model.addConstr(
            t[j] <= t[i] - x[i, j] + total_nodes * (1 - x[i, j]),
            name=f"radial_{i}_{j}",
        )

    # Restrições de limite de percurso
    model.addConstr(l[BASE_NODE] == 0.0, name="base_distance")

    for i, j in route_arcs:
        model.addConstr(
            l[j] >= l[i] + problem.alpha[i, j] - route_limit * (1 - x[i, j]),
            name=f"distance_progress_{i}_{j}",
        )

    for critical in problem.critical_nodes:
        model.addConstr(
            l[critical] + problem.alpha[critical, BASE_NODE]
            <= route_limit + route_limit * (1 - z[critical]),
            name=f"critical_route_limit_{critical}",
        )

    for station in problem.station_nodes:
        model.addConstr(
            w[station] <= len(problem.critical_nodes) * u[station],
            name=f"station_route_end_indicator_{station}",
        )
        model.addConstr(
            l[station] + problem.alpha[station, BASE_NODE]
            <= route_limit + route_limit * (1 - u[station]),
            name=f"station_route_limit_{station}",
        )
        model.addConstr(
            problem.alpha[station, BASE_NODE] * u[station]
            <= battery_capacity * u[station],
            name=f"station_return_energy_{station}",
        )
        if problem.alpha[station, BASE_NODE] > battery_capacity + 1e-9:
            u[station].setAttr(GRB.Attr.UB, 0.0)

    return model, x, y, z, w, b, t, l, u


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
            if (
                current in problem.critical_nodes
                and remaining_critical_ends[current] > 0
            ):
                remaining_critical_ends[current] -= 1
                if remaining_critical_ends[current] == 0:
                    del remaining_critical_ends[current]
                break

            next_candidates = sorted(
                j
                for j in service_nodes
                if current != j and remaining_edges[current, j] > 0
            )

            if current in problem.station_nodes and not next_candidates:
                if remaining_station_ends[current] <= 0:
                    raise RuntimeError(
                        "A reconstrucao encontrou uma estacao sem continuidade nem encerramento valido."
                    )
                remaining_station_ends[current] -= 1
                if remaining_station_ends[current] == 0:
                    del remaining_station_ends[current]
                break

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
            "Restaram componentes nao utilizados ao reconstruir as rotas do modelo de Ribeiro."
        )

    return tuple(routes)


def solve_problem(
    problem: ProblemData,
    time_limit: float | None = None,
) -> SolutionData:
    model, x, y, z, w, _, _, _, _ = build_model(problem)

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
        sorted(station for station in problem.station_nodes if y_values[station] > 0.5)
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
