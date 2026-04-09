from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pprint import pformat

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


def build_model(
    problem: ProblemData, debug_subtours: bool = False
) -> tuple[gp.Model, gp.tupledict, gp.tupledict, gp.tupledict]:
    model = gp.Model(f"lrpsd_{problem.name}")

    x = model.addVars(
        problem.arcs,
        vtype=GRB.INTEGER,
        lb=0,
        ub=len(problem.critical_nodes),
        name="x",
    )
    y = model.addVars(problem.station_nodes, vtype=GRB.BINARY, name="y")
    f = model.addVars(problem.arcs, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

    # model.setObjective(
    #     gp.quicksum(problem.alpha[i, j] * x[i, j] for i, j in problem.arcs)
    #     + problem.drone_cost
    #     * gp.quicksum(x[BASE_NODE, j] for j in problem.all_nodes if j != BASE_NODE)
    #     + problem.station_cost * gp.quicksum(y[r] for r in problem.station_nodes),
    #     GRB.MINIMIZE,
    # )

    model.setObjective(
        problem.drone_cost
        * gp.quicksum(x[BASE_NODE, j] for j in problem.all_nodes if j != BASE_NODE)
        + problem.station_cost * gp.quicksum(y[r] for r in problem.station_nodes),
        GRB.MINIMIZE,
    )

    for c in problem.critical_nodes:
        model.addConstr(
            gp.quicksum(x[i, c] for i in problem.all_nodes if i != c) == 1,
            name=f"critical_in_{c}",
        )
        model.addConstr(
            gp.quicksum(x[c, j] for j in problem.all_nodes if j != c) == 1,
            name=f"critical_out_{c}",
        )

    for j in problem.all_nodes:
        if j == BASE_NODE:
            continue
        model.addConstr(
            gp.quicksum(x[i, j] for i in problem.all_nodes if i != j)
            == gp.quicksum(x[j, l] for l in problem.all_nodes if l != j),
            name=f"flow_balance_{j}",
        )

    for r in problem.station_nodes:
        model.addConstr(
            gp.quicksum(x[r, j] for j in problem.all_nodes if j != r)
            <= len(problem.critical_nodes) * y[r],
            name=f"station_activation_{r}",
        )

    for j in problem.all_nodes:
        if j == BASE_NODE:
            continue
        model.addConstr(
            f[BASE_NODE, j] == problem.battery_capacity * x[BASE_NODE, j],
            name=f"base_energy_{j}",
        )

    for r in problem.station_nodes:
        for j in problem.all_nodes:
            if j == r:
                continue
            model.addConstr(
                f[r, j] == problem.battery_capacity * x[r, j],
                name=f"station_energy_{r}_{j}",
            )

    for c in problem.critical_nodes:
        for j in problem.all_nodes:
            if j == c:
                continue
            model.addConstr(
                f[c, j] <= problem.battery_capacity * x[c, j],
                name=f"critical_energy_cap_{c}_{j}",
            )

    for i, j in problem.arcs:
        min_required_energy = problem.alpha[i, j]
        if j in problem.critical_nodes:
            min_required_energy += problem.beta[j]
        model.addConstr(
            f[i, j] >= min_required_energy * x[i, j],
            name=f"travel_energy_lb_{i}_{j}",
        )

    for c in problem.critical_nodes:
        model.addConstr(
            gp.quicksum(f[i, c] for i in problem.all_nodes if i != c)
            - gp.quicksum(f[c, j] for j in problem.all_nodes if j != c)
            == gp.quicksum(
                problem.alpha[i, c] * x[i, c] for i in problem.all_nodes if i != c
            )
            + problem.beta[c],
            name=f"critical_energy_balance_{c}",
        )

    model.Params.LazyConstraints = 1
    model._problem = problem
    model._x = x
    model._debug_subtours = debug_subtours
    return model, x, y, f


def _disconnected_components(
    problem: ProblemData,
    x_values: dict[tuple[int, int], float],
    debug: bool = False,
) -> list[set[int]]:
    adjacency: dict[int, set[int]] = {node: set() for node in problem.all_nodes}
    active_arcs: list[tuple[int, int, float]] = []

    for (i, j), value in x_values.items():
        if value <= 0.5:
            continue
        adjacency[i].add(j)
        adjacency[j].add(i)
        active_arcs.append((i, j, value))

    if debug:
        debug_adjacency = {
            node: neighbors for node, neighbors in adjacency.items() if neighbors
        }
        print("\n[DEBUG] _disconnected_components")
        print(f"[DEBUG] active_arcs = {pformat(active_arcs)}")
        print(f"[DEBUG] adjacency = {pformat(debug_adjacency)}")

    visited: set[int] = set()
    components: list[set[int]] = []

    for start in problem.all_nodes:
        if start in visited or not adjacency[start]:
            visited.add(start)
            continue

        queue = deque([start])
        component: set[int] = set()
        visited.add(start)

        if debug:
            print(f"[DEBUG] start = {start}")
            print(f"[DEBUG] queue_init = {list(queue)}")

        while queue:
            node = queue.popleft()
            component.add(node)
            if debug:
                print(
                    "[DEBUG] pop = "
                    f"{node}, queue = {list(queue)}, component = {component}"
                )
            for neighbor in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
                if debug:
                    print(
                        "[DEBUG] push = "
                        f"{neighbor}, queue = {list(queue)}, "
                        f"visited = {visited}"
                    )

        components.append(component)
        if debug:
            print(f"[DEBUG] completed_component = {component}")

    filtered_components = [
        component
        for component in components
        if BASE_NODE not in component
        and any(node in problem.critical_nodes for node in component)
    ]

    if debug:
        print(f"[DEBUG] visited = {visited}")
        print(
            f"[DEBUG] components = {pformat([component for component in components])}"
        )
        print(
            "[DEBUG] filtered_components = "
            f"{pformat([component for component in filtered_components])}"
        )

    return filtered_components


def _subtour_callback(model: gp.Model, where: int) -> None:
    if where != GRB.Callback.MIPSOL:
        return

    problem: ProblemData = model._problem
    x_vars: gp.tupledict = model._x
    debug_subtours: bool = model._debug_subtours
    x_values = model.cbGetSolution(x_vars)

    for component in _disconnected_components(problem, x_values, debug=debug_subtours):
        if debug_subtours:
            print(f"[DEBUG] lazy_cut_component = {component}")
        model.cbLazy(
            gp.quicksum(
                x_vars[i, j]
                for i in problem.all_nodes
                if i not in component
                for j in component
                if i != j
            )
            >= 1
        )


def _extract_routes(
    problem: ProblemData, x_values: dict[tuple[int, int], float]
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

    successors: dict[int, list[int]] = {node: [] for node in problem.all_nodes}
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
            "Nem todos os arcos da solução foram utilizados na reconstrução das rotas."
        )

    routes: list[tuple[int, ...]] = []
    current_route = [BASE_NODE]

    for node in circuit[1:]:
        current_route.append(node)
        if node == BASE_NODE:
            routes.append(tuple(current_route))
            current_route = [BASE_NODE]

    if len(current_route) != 1:
        raise RuntimeError("A decomposição em rotas terminou com uma rota aberta.")

    return tuple(route for route in routes if len(route) > 1)


def solve_problem(
    problem: ProblemData,
    time_limit: float | None = None,
    debug_subtours: bool = False,
) -> SolutionData:
    model, x, y, _ = build_model(problem, debug_subtours=debug_subtours)

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

    drone_count = int(
        round(sum(x_values[BASE_NODE, j] for j in problem.all_nodes if j != BASE_NODE))
    )
    active_stations = tuple(
        sorted(r for r in problem.station_nodes if y_values[r] > 0.5)
    )
    routes = _extract_routes(problem, x_values)
    mip_gap = model.MIPGap if model.IsMIP else 0.0

    return SolutionData(
        status=model.Status,
        objective_value=model.ObjVal,
        mip_gap=mip_gap,
        drone_count=drone_count,
        active_stations=active_stations,
        routes=routes,
    )
