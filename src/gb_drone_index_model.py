from __future__ import annotations

from collections import Counter, deque
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


def _all_arcs(problem: ProblemData) -> tuple[tuple[int, int], ...]:
    arcs: list[tuple[int, int]] = []

    for i in problem.all_nodes:
        for j in problem.all_nodes:
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


def _drone_ids(problem: ProblemData, drone_upper_bound: int | None) -> tuple[int, ...]:
    if drone_upper_bound is None:
        drone_upper_bound = len(problem.critical_nodes)
    return tuple(range(1, max(0, drone_upper_bound) + 1))


def build_model(
    problem: ProblemData,
    drone_upper_bound: int | None = None,
    debug_subtours: bool = False,
) -> tuple[
    gp.Model,
    gp.tupledict,
    gp.tupledict,
    gp.tupledict,
    gp.tupledict,
    tuple[int, ...],
    tuple[tuple[int, int], ...],
]:
    drones = _drone_ids(problem, drone_upper_bound)
    arcs = _all_arcs(problem)
    arc_keys = tuple((drone, i, j) for drone in drones for i, j in arcs)

    incoming: dict[int, tuple[int, ...]] = {node: tuple() for node in problem.all_nodes}
    outgoing: dict[int, tuple[int, ...]] = {node: tuple() for node in problem.all_nodes}
    incoming_lists: dict[int, list[int]] = {node: [] for node in problem.all_nodes}
    outgoing_lists: dict[int, list[int]] = {node: [] for node in problem.all_nodes}

    for i, j in arcs:
        outgoing_lists[i].append(j)
        incoming_lists[j].append(i)

    for node in problem.all_nodes:
        incoming[node] = tuple(sorted(incoming_lists[node]))
        outgoing[node] = tuple(sorted(outgoing_lists[node]))

    model = gp.Model(f"lrpsd_drone_index_{problem.name}")

    x = model.addVars(
        arc_keys,
        vtype=GRB.BINARY,
        name="x",
    )
    z = model.addVars(drones, vtype=GRB.BINARY, name="z")
    y = model.addVars(problem.station_nodes, vtype=GRB.BINARY, name="y")
    v = model.addVars(
        ((critical, drone) for critical in problem.critical_nodes for drone in drones),
        vtype=GRB.BINARY,
        name="v",
    )
    f = model.addVars(arc_keys, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

    model.setObjective(
        problem.drone_cost * gp.quicksum(z[drone] for drone in drones)
        + problem.station_cost
        * gp.quicksum(y[station] for station in problem.station_nodes),
        GRB.MINIMIZE,
    )

    for drone in drones:
        model.addConstr(
            gp.quicksum(x[drone, BASE_NODE, j] for j in outgoing[BASE_NODE])
            == z[drone],
            name=f"base_departure_{drone}",
        )
        model.addConstr(
            gp.quicksum(x[drone, i, BASE_NODE] for i in incoming[BASE_NODE])
            == gp.quicksum(x[drone, BASE_NODE, j] for j in outgoing[BASE_NODE]),
            name=f"base_balance_{drone}",
        )

    for critical in problem.critical_nodes:
        model.addConstr(
            gp.quicksum(v[critical, drone] for drone in drones) == 1,
            name=f"critical_assignment_{critical}",
        )

        for drone in drones:
            model.addConstr(
                gp.quicksum(x[drone, i, critical] for i in incoming[critical])
                == v[critical, drone],
                name=f"critical_in_{critical}_{drone}",
            )
            model.addConstr(
                gp.quicksum(x[drone, critical, j] for j in outgoing[critical])
                == v[critical, drone],
                name=f"critical_out_{critical}_{drone}",
            )

    for station in problem.station_nodes:
        model.addConstr(
            gp.quicksum(
                x[drone, i, station] for drone in drones for i in incoming[station]
            )
            <= len(problem.critical_nodes) * y[station],
            name=f"station_activation_{station}",
        )

    for drone in drones:
        for station in problem.station_nodes:
            model.addConstr(
                gp.quicksum(x[drone, i, station] for i in incoming[station])
                == gp.quicksum(x[drone, station, j] for j in outgoing[station]),
                name=f"station_balance_{station}_{drone}",
            )

    for drone in drones:
        for j in outgoing[BASE_NODE]:
            model.addConstr(
                f[drone, BASE_NODE, j]
                == problem.battery_capacity * x[drone, BASE_NODE, j],
                name=f"base_energy_{drone}_{j}",
            )

        for station in problem.station_nodes:
            for j in outgoing[station]:
                model.addConstr(
                    f[drone, station, j]
                    == problem.battery_capacity * x[drone, station, j],
                    name=f"station_energy_{drone}_{station}_{j}",
                )

        for critical in problem.critical_nodes:
            for j in outgoing[critical]:
                model.addConstr(
                    f[drone, critical, j]
                    <= problem.battery_capacity * x[drone, critical, j],
                    name=f"critical_energy_cap_{drone}_{critical}_{j}",
                )

    for drone, i, j in arc_keys:
        required_energy = problem.alpha[i, j]
        if j in problem.critical_nodes:
            required_energy += problem.beta[j]
        model.addConstr(
            f[drone, i, j] >= required_energy * x[drone, i, j],
            name=f"travel_energy_lb_{drone}_{i}_{j}",
        )

    for drone in drones:
        for critical in problem.critical_nodes:
            model.addConstr(
                gp.quicksum(f[drone, i, critical] for i in incoming[critical])
                - gp.quicksum(f[drone, critical, j] for j in outgoing[critical])
                == gp.quicksum(
                    problem.alpha[i, critical] * x[drone, i, critical]
                    for i in incoming[critical]
                )
                + problem.beta[critical] * v[critical, drone],
                name=f"critical_energy_balance_{drone}_{critical}",
            )

    for drone in drones:
        model.addConstr(
            gp.quicksum(problem.alpha[i, j] * x[drone, i, j] for i, j in arcs)
            <= problem.max_route_length * z[drone],
            name=f"route_limit_{drone}",
        )

    model.Params.LazyConstraints = 1
    model._problem = problem
    model._drones = drones
    model._all_nodes = problem.all_nodes
    model._arcs = arcs
    model._x = x
    model._v = v
    model._debug_subtours = debug_subtours
    return model, x, y, z, v, drones, arcs


def _disconnected_components(
    all_nodes: tuple[int, ...],
    arcs: tuple[tuple[int, int], ...],
    x_values: dict[tuple[int, int, int], float],
    drone: int,
) -> list[set[int]]:
    adjacency: dict[int, set[int]] = {node: set() for node in all_nodes}
    active_nodes: set[int] = set()

    for i, j in arcs:
        if x_values[drone, i, j] <= 0.5:
            continue
        adjacency[i].add(j)
        adjacency[j].add(i)
        active_nodes.add(i)
        active_nodes.add(j)

    if BASE_NODE not in active_nodes:
        visited_from_base = {BASE_NODE}
    else:
        visited_from_base: set[int] = set()
        queue = deque([BASE_NODE])
        visited_from_base.add(BASE_NODE)

        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in visited_from_base:
                    continue
                visited_from_base.add(neighbor)
                queue.append(neighbor)

    remaining_nodes = active_nodes - visited_from_base
    components: list[set[int]] = []

    while remaining_nodes:
        start = remaining_nodes.pop()
        component = {start}
        queue = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in component or neighbor in visited_from_base:
                    continue
                component.add(neighbor)
                if neighbor in remaining_nodes:
                    remaining_nodes.remove(neighbor)
                queue.append(neighbor)

        components.append(component)

    return components


def _subtour_callback(model: gp.Model, where: int) -> None:
    if where != GRB.Callback.MIPSOL:
        return

    all_nodes: tuple[int, ...] = model._all_nodes
    arcs: tuple[tuple[int, int], ...] = model._arcs
    drones: tuple[int, ...] = model._drones
    x_vars: gp.tupledict = model._x
    v_vars: gp.tupledict = model._v
    problem: ProblemData = model._problem
    x_values = model.cbGetSolution(x_vars)
    v_values = model.cbGetSolution(v_vars)

    critical_set = frozenset(problem.critical_nodes)

    for drone in drones:
        for component in _disconnected_components(all_nodes, arcs, x_values, drone):
            for critical in component:
                if critical not in critical_set:
                    continue
                rhs = v_values[critical, drone]
                if rhs <= 1e-6:
                    continue
                lhs = sum(
                    x_values[drone, i, j]
                    for i in all_nodes
                    if i not in component
                    for j in component
                    if i != j and (drone, i, j) in x_vars
                )
                if lhs + 1e-5 >= rhs:
                    continue
                model.cbLazy(
                    gp.quicksum(
                        x_vars[drone, i, j]
                        for i in all_nodes
                        if i not in component
                        for j in component
                        if i != j and (drone, i, j) in x_vars
                    )
                    >= v_vars[critical, drone]
                )


def _extract_route(
    drone: int,
    all_nodes: tuple[int, ...],
    arcs: tuple[tuple[int, int], ...],
    x_values: dict[tuple[int, int, int], float],
) -> tuple[int, ...]:
    remaining = Counter(
        {
            (i, j): int(round(x_values[drone, i, j]))
            for i, j in arcs
            if int(round(x_values[drone, i, j])) > 0
        }
    )

    if not remaining:
        return ()

    successors: dict[int, list[int]] = {node: [] for node in all_nodes}
    for (i, j), multiplicity in remaining.items():
        successors[i].extend([j] * multiplicity)

    for node in successors:
        successors[node].sort(reverse=True)

    stack = [BASE_NODE]
    circuit: list[int] = []

    while stack:
        current = stack[-1]
        if successors[current]:
            stack.append(successors[current].pop())
            continue
        circuit.append(stack.pop())

    circuit.reverse()

    used_arc_count = len(circuit) - 1
    expected_arc_count = sum(remaining.values())
    if used_arc_count != expected_arc_count:
        raise RuntimeError(
            "Nem todos os arcos do drone foram utilizados na reconstrução da rota."
        )

    if circuit[0] != BASE_NODE or circuit[-1] != BASE_NODE:
        raise RuntimeError("A rota reconstruida do drone nao parte e termina na base.")

    return tuple(circuit)


def solve_problem(
    problem: ProblemData,
    time_limit: float | None = None,
    drone_upper_bound: int | None = None,
    debug_subtours: bool = False,
) -> SolutionData:
    model, x, y, z, _, drones, arcs = build_model(
        problem,
        drone_upper_bound=drone_upper_bound,
        debug_subtours=debug_subtours,
    )

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    model.optimize(_subtour_callback)

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

    active_drones = tuple(sorted(drone for drone in drones if z_values[drone] > 0.5))
    routes = tuple(
        _extract_route(drone, problem.all_nodes, arcs, x_values)
        for drone in active_drones
    )
    active_stations = tuple(
        sorted(station for station in problem.station_nodes if y_values[station] > 0.5)
    )
    mip_gap = model.MIPGap if model.IsMIP else 0.0

    return SolutionData(
        status=model.Status,
        objective_value=model.ObjVal,
        mip_gap=mip_gap,
        drone_count=len(active_drones),
        active_stations=active_stations,
        routes=routes,
    )
