from __future__ import annotations

from dataclasses import dataclass
from math import dist
from pathlib import Path


BASE_NODE = 0
CRITICAL_NODE = 1
STATION_NODE = 2


@dataclass(frozen=True)
class Node:
    node_id: int
    node_type: int
    x: float
    y: float
    beta: float


@dataclass(frozen=True)
class ProblemData:
    name: str
    dimension: int
    battery_capacity: float
    max_route_length: float
    drone_cost: float
    station_cost: float
    nodes: dict[int, Node]
    critical_nodes: tuple[int, ...]
    station_nodes: tuple[int, ...]
    all_nodes: tuple[int, ...]
    arcs: tuple[tuple[int, int], ...]
    alpha: dict[tuple[int, int], float]
    beta: dict[int, float]


def _parse_header_value(line: str) -> str:
    return line.split(":", maxsplit=1)[1].strip()


def read_problem(instance_path: str | Path) -> ProblemData:
    path = Path(instance_path)
    headers: dict[str, str] = {}
    nodes: dict[int, Node] = {}
    reading_nodes = False

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line == "NODE_DATA_SECTION":
                reading_nodes = True
                continue

            if line == "EOF":
                break

            if not reading_nodes:
                key = line.split(":", maxsplit=1)[0].strip()
                headers[key] = _parse_header_value(line)
                continue

            node_id_str, node_type_str, x_str, y_str, beta_str = line.split()
            node = Node(
                node_id=int(node_id_str),
                node_type=int(node_type_str),
                x=float(x_str),
                y=float(y_str),
                beta=float(beta_str),
            )
            nodes[node.node_id] = node

    all_nodes = tuple(sorted(nodes))
    critical_nodes = tuple(
        node_id for node_id in all_nodes if nodes[node_id].node_type == CRITICAL_NODE
    )
    station_nodes = tuple(
        node_id for node_id in all_nodes if nodes[node_id].node_type == STATION_NODE
    )
    arcs = tuple((i, j) for i in all_nodes for j in all_nodes if i != j)

    alpha = {
        (i, j): dist((nodes[i].x, nodes[i].y), (nodes[j].x, nodes[j].y))
        for i, j in arcs
    }
    beta = {node_id: node.beta for node_id, node in nodes.items()}
    if "MAX_ROUTE_LENGTH" in headers:
        max_route_length = float(headers["MAX_ROUTE_LENGTH"])
    else:
        # Backward-compatible fallback for legacy instances generated before L
        # became part of the instance format.
        max_arc_cost = max(alpha.values(), default=0.0)
        max_route_length = len(all_nodes) * max_arc_cost

    return ProblemData(
        name=headers["NAME"],
        dimension=int(headers["DIMENSION"]),
        battery_capacity=float(headers["BATTERY_CAPACITY"]),
        max_route_length=max_route_length,
        drone_cost=float(headers["DRONE_COST"]),
        station_cost=float(headers["STATION_COST"]),
        nodes=nodes,
        critical_nodes=critical_nodes,
        station_nodes=station_nodes,
        all_nodes=all_nodes,
        arcs=arcs,
        alpha=alpha,
        beta=beta,
    )
