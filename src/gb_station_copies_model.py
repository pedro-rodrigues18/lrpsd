from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import dist

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


@dataclass(frozen=True)
class ExpandedProblem:
    all_nodes: tuple[int, ...]
    service_nodes: tuple[int, ...]
    artificial_station_nodes: tuple[int, ...]
    copy_nodes_by_station: dict[int, tuple[int, ...]]
    station_of_copy: dict[int, int]
    arcs: tuple[tuple[int, int], ...]
    incoming: dict[int, tuple[int, ...]]
    outgoing: dict[int, tuple[int, ...]]
    alpha: dict[tuple[int, int], float]


def _build_expanded_problem(problem: ProblemData) -> ExpandedProblem:
    copy_count = max(1, len(problem.critical_nodes))
    next_node_id = max(problem.all_nodes, default=BASE_NODE) + 1

    copy_nodes_by_station: dict[int, tuple[int, ...]] = {}
    station_of_copy: dict[int, int] = {}
    artificial_station_nodes: list[int] = []

    for station in problem.station_nodes:
        copies: list[int] = []
        for _ in range(copy_count):
            copy_node = next_node_id
            next_node_id += 1
            copies.append(copy_node)
            station_of_copy[copy_node] = station
            artificial_station_nodes.append(copy_node)
        copy_nodes_by_station[station] = tuple(copies)

    all_nodes = (BASE_NODE,) + problem.critical_nodes + tuple(artificial_station_nodes)
    service_nodes = all_nodes[1:]
    critical_set = set(problem.critical_nodes)

    coordinates: dict[int, tuple[float, float]] = {}
    for node in all_nodes:
        source_node = problem.nodes[station_of_copy[node]] if node in station_of_copy else problem.nodes[node]
        coordinates[node] = (source_node.x, source_node.y)

    arcs: list[tuple[int, int]] = []
    alpha: dict[tuple[int, int], float] = {}
    incoming_lists: dict[int, list[int]] = {node: [] for node in all_nodes}
    outgoing_lists: dict[int, list[int]] = {node: [] for node in all_nodes}

    for i in all_nodes:
        coord_i = coordinates[i]
        for j in all_nodes:
            if i == j:
                continue
            if (
                i in station_of_copy
                and j in station_of_copy
                and station_of_copy[i] == station_of_copy[j]
            ):
                continue

            travel_cost = dist(coord_i, coordinates[j])
            if travel_cost > problem.max_route_length + 1e-9:
                continue

            required_energy = travel_cost + (problem.beta[j] if j in critical_set else 0.0)
            if required_energy > problem.battery_capacity + 1e-9:
                continue

            arc = (i, j)
            arcs.append(arc)
            alpha[arc] = travel_cost
            outgoing_lists[i].append(j)
            incoming_lists[j].append(i)

    return ExpandedProblem(
        all_nodes=all_nodes,
        service_nodes=service_nodes,
        artificial_station_nodes=tuple(artificial_station_nodes),
        copy_nodes_by_station=copy_nodes_by_station,
        station_of_copy=station_of_copy,
        arcs=tuple(arcs),
        incoming={node: tuple(sorted(nodes)) for node, nodes in incoming_lists.items()},
        outgoing={node: tuple(sorted(nodes)) for node, nodes in outgoing_lists.items()},
        alpha=alpha,
    )


def _display_node(expanded_problem: ExpandedProblem, node: int) -> int:
    return expanded_problem.station_of_copy.get(node, node)


def build_model(
    problem: ProblemData,
) -> tuple[gp.Model, gp.tupledict, gp.tupledict, ExpandedProblem]:
    expanded_problem = _build_expanded_problem(problem)
    model = gp.Model(f"lrpsd_station_copies_{problem.name}")

    x = model.addVars(expanded_problem.arcs, vtype=GRB.BINARY, name="x")
    y = model.addVars(problem.station_nodes, vtype=GRB.BINARY, name="y")
    f = model.addVars(expanded_problem.arcs, vtype=GRB.CONTINUOUS, lb=0.0, name="f")
    l = model.addVars(
        expanded_problem.arcs,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        ub=problem.max_route_length,
        name="l",
    )

    model.setObjective(
        problem.drone_cost
        * gp.quicksum(x[BASE_NODE, j] for j in expanded_problem.outgoing[BASE_NODE])
        + problem.station_cost * gp.quicksum(y[r] for r in problem.station_nodes),
        GRB.MINIMIZE,
    )

    for critical in problem.critical_nodes:
        model.addConstr(
            gp.quicksum(x[i, critical] for i in expanded_problem.incoming[critical]) == 1,
            name=f"critical_in_{critical}",
        )
        model.addConstr(
            gp.quicksum(x[critical, j] for j in expanded_problem.outgoing[critical]) == 1,
            name=f"critical_out_{critical}",
        )

    for node in expanded_problem.service_nodes:
        model.addConstr(
            gp.quicksum(x[i, node] for i in expanded_problem.incoming[node])
            == gp.quicksum(x[node, j] for j in expanded_problem.outgoing[node]),
            name=f"flow_balance_{node}",
        )

    for station, copies in expanded_problem.copy_nodes_by_station.items():
        for copy_node in copies:
            model.addConstr(
                gp.quicksum(x[copy_node, j] for j in expanded_problem.outgoing[copy_node])
                <= y[station],
                name=f"station_activation_{station}_{copy_node}",
            )

    for j in expanded_problem.outgoing[BASE_NODE]:
        model.addConstr(
            f[BASE_NODE, j] == problem.battery_capacity * x[BASE_NODE, j],
            name=f"base_energy_{j}",
        )

    for copy_node in expanded_problem.artificial_station_nodes:
        for j in expanded_problem.outgoing[copy_node]:
            model.addConstr(
                f[copy_node, j] == problem.battery_capacity * x[copy_node, j],
                name=f"station_energy_{copy_node}_{j}",
            )

    for critical in problem.critical_nodes:
        for j in expanded_problem.outgoing[critical]:
            model.addConstr(
                f[critical, j] <= problem.battery_capacity * x[critical, j],
                name=f"critical_energy_cap_{critical}_{j}",
            )

    for i, j in expanded_problem.arcs:
        required_energy = expanded_problem.alpha[i, j]
        if j in problem.critical_nodes:
            required_energy += problem.beta[j]
        model.addConstr(
            f[i, j] >= required_energy * x[i, j],
            name=f"travel_energy_lb_{i}_{j}",
        )

    for critical in problem.critical_nodes:
        model.addConstr(
            gp.quicksum(f[i, critical] for i in expanded_problem.incoming[critical])
            - gp.quicksum(f[critical, j] for j in expanded_problem.outgoing[critical])
            == gp.quicksum(
                expanded_problem.alpha[i, critical] * x[i, critical]
                for i in expanded_problem.incoming[critical]
            )
            + problem.beta[critical],
            name=f"critical_energy_balance_{critical}",
        )

    for j in expanded_problem.outgoing[BASE_NODE]:
        model.addConstr(
            l[BASE_NODE, j] == expanded_problem.alpha[BASE_NODE, j] * x[BASE_NODE, j],
            name=f"base_distance_{j}",
        )

    for node in expanded_problem.service_nodes:
        model.addConstr(
            gp.quicksum(l[node, j] for j in expanded_problem.outgoing[node])
            - gp.quicksum(l[i, node] for i in expanded_problem.incoming[node])
            == gp.quicksum(
                expanded_problem.alpha[node, j] * x[node, j]
                for j in expanded_problem.outgoing[node]
            ),
            name=f"distance_balance_{node}",
        )

    for i, j in expanded_problem.arcs:
        model.addConstr(
            l[i, j] <= problem.max_route_length * x[i, j],
            name=f"route_length_cap_{i}_{j}",
        )

    return model, x, y, expanded_problem


def _extract_routes(
    expanded_problem: ExpandedProblem,
    x_values: dict[tuple[int, int], float],
) -> tuple[tuple[int, ...], ...]:
    remaining = Counter(
        {
            arc: int(round(value))
            for arc, value in x_values.items()
            if int(round(value)) > 0
        }
    )

    if not remaining:
        return ()

    routes: list[tuple[int, ...]] = []
    base_successors = [
        node for node in expanded_problem.outgoing[BASE_NODE] if remaining[BASE_NODE, node] > 0
    ]

    for start_node in base_successors:
        remaining[BASE_NODE, start_node] -= 1
        if remaining[BASE_NODE, start_node] == 0:
            del remaining[BASE_NODE, start_node]

        route = [BASE_NODE, _display_node(expanded_problem, start_node)]
        current = start_node

        while current != BASE_NODE:
            next_candidates = [
                node
                for node in expanded_problem.outgoing[current]
                if remaining[current, node] > 0
            ]
            if not next_candidates:
                raise RuntimeError(
                    "Nao foi possivel reconstruir uma rota completa do modelo com copias."
                )

            next_node = next_candidates[0]
            remaining[current, next_node] -= 1
            if remaining[current, next_node] == 0:
                del remaining[current, next_node]

            current = next_node
            route.append(_display_node(expanded_problem, current))

        routes.append(tuple(route))

    if remaining:
        raise RuntimeError(
            "Restaram arcos nao utilizados ao reconstruir as rotas do modelo com copias."
        )

    return tuple(routes)


def solve_problem(
    problem: ProblemData,
    time_limit: float | None = None,
    debug_subtours: bool = False,
) -> SolutionData:
    del debug_subtours
    model, x, y, expanded_problem = build_model(problem)

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

    drone_count = int(
        round(sum(x_values[BASE_NODE, j] for j in expanded_problem.outgoing[BASE_NODE]))
    )
    active_stations = tuple(
        sorted(station for station in problem.station_nodes if y_values[station] > 0.5)
    )
    routes = _extract_routes(expanded_problem, x_values)
    mip_gap = model.MIPGap if model.IsMIP else 0.0

    return SolutionData(
        status=model.Status,
        objective_value=model.ObjVal,
        mip_gap=mip_gap,
        drone_count=drone_count,
        active_stations=active_stations,
        routes=routes,
    )
