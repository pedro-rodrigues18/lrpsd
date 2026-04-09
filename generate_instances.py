import os
import random
from math import cos, hypot, pi, sin

# Parâmetros globais baseados em Ribeiro et al. (2020)
STATION_COST = 1000

# Especificações dos Drones
DRONES = {
    "Phantom": {"capacity": 17664, "cost": 1200, "beta": 1920},
    "Inspire": {"capacity": 19008, "cost": 2000, "beta": 1540},
}

# Configuração das 15 instâncias (Fácil, Média, Difícil)
# grid_size: define o tamanho da área de operação em metros (X_max, Y_max)
CONFIGURATIONS = [
    # --- Fáceis (Ideais para depuração e testes rápidos) ---
    {
        "name": "Inst_01_Easy_Phantom",
        "num_c": 10,
        "num_r": 3,
        "drone": "Phantom",
        "grid_size": 4000,
    },
    {
        "name": "Inst_02_Easy_Inspire",
        "num_c": 12,
        "num_r": 3,
        "drone": "Inspire",
        "grid_size": 4500,
    },
    {
        "name": "Inst_03_Easy_Phantom",
        "num_c": 12,
        "num_r": 4,
        "drone": "Phantom",
        "grid_size": 4500,
    },
    {
        "name": "Inst_04_Easy_Inspire",
        "num_c": 15,
        "num_r": 4,
        "drone": "Inspire",
        "grid_size": 5000,
    },
    {
        "name": "Inst_05_Easy_Phantom",
        "num_c": 16,
        "num_r": 4,
        "drone": "Phantom",
        "grid_size": 5000,
    },
    # --- Médias (Tempo de execução moderado no solver) ---
    {
        "name": "Inst_06_Medium_Inspire",
        "num_c": 20,
        "num_r": 5,
        "drone": "Inspire",
        "grid_size": 6000,
    },
    {
        "name": "Inst_07_Medium_Phantom",
        "num_c": 23,
        "num_r": 6,
        "drone": "Phantom",
        "grid_size": 6000,
    },
    {
        "name": "Inst_08_Medium_Inspire",
        "num_c": 25,
        "num_r": 6,
        "drone": "Inspire",
        "grid_size": 7000,
    },
    {
        "name": "Inst_09_Medium_Phantom",
        "num_c": 30,
        "num_r": 8,
        "drone": "Phantom",
        "grid_size": 7500,
    },
    {
        "name": "Inst_10_Medium_Inspire",
        "num_c": 35,
        "num_r": 9,
        "drone": "Inspire",
        "grid_size": 8000,
    },
    # --- Difíceis (Podem não provar otimalidade em poucas horas) ---
    {
        "name": "Inst_11_Hard_Phantom",
        "num_c": 40,
        "num_r": 10,
        "drone": "Phantom",
        "grid_size": 9000,
    },
    {
        "name": "Inst_12_Hard_Inspire",
        "num_c": 45,
        "num_r": 11,
        "drone": "Inspire",
        "grid_size": 9000,
    },
    {
        "name": "Inst_13_Hard_Phantom",
        "num_c": 46,
        "num_r": 12,
        "drone": "Phantom",
        "grid_size": 10000,
    },
    {
        "name": "Inst_14_Hard_Inspire",
        "num_c": 50,
        "num_r": 12,
        "drone": "Inspire",
        "grid_size": 10000,
    },
    {
        "name": "Inst_15_Hard_Phantom",
        "num_c": 60,
        "num_r": 15,
        "drone": "Phantom",
        "grid_size": 12000,
    },
]

# Instâncias desenhadas para explorar multi-rotas e a instalação de estações.
# A ideia é criar clusters distantes entre si, com pares de estações candidatas
# próximos de cada cluster. Assim, o modelo tende a:
# 1. precisar de várias rotas por causa da limitação energética; e
# 2. decidir qual subconjunto de estações instalar para acessar cada região.
MULTI_DRONE_CONFIGURATIONS = [
    {
        "name": "Inst_16_MultiDrone_Inspire",
        "generator": "clustered_multi_drone",
        "drone": "Inspire",
        "cluster_centers": [(11200.0, 11200.0), (-11200.0, 11200.0), (11200.0, -11200.0)],
        "criticals_per_cluster": [4, 4, 4],
        "critical_radius": 320.0,
        "gateway_offset": 1700.0,
        "side_station_offset": 850.0,
        "station_cost": 1000,
    },
    {
        "name": "Inst_17_MultiDrone_Phantom",
        "generator": "clustered_multi_drone",
        "drone": "Phantom",
        "cluster_centers": [(10500.0, 10500.0), (-10500.0, 10500.0), (10500.0, -10500.0)],
        "criticals_per_cluster": [4, 4, 4],
        "critical_radius": 300.0,
        "gateway_offset": 1600.0,
        "side_station_offset": 750.0,
        "station_cost": 1000,
    },
    {
        "name": "Inst_18_MultiDrone_Inspire_4Q",
        "generator": "clustered_multi_drone",
        "drone": "Inspire",
        "cluster_centers": [
            (10500.0, 10500.0),
            (-10500.0, 10500.0),
            (-10500.0, -10500.0),
            (10500.0, -10500.0),
        ],
        "criticals_per_cluster": [4, 4, 4, 4],
        "critical_radius": 300.0,
        "gateway_offset": 1600.0,
        "side_station_offset": 800.0,
        "station_cost": 1000,
    },
    {
        "name": "Inst_19_MultiDrone_Asymmetric",
        "generator": "clustered_multi_drone",
        "drone": "Phantom",
        "cluster_centers": [(11800.0, 9800.0), (-10800.0, 10800.0), (9800.0, -11800.0)],
        "criticals_per_cluster": [5, 4, 4],
        "critical_radius": 340.0,
        "gateway_offset": 1800.0,
        "side_station_offset": 900.0,
        "station_cost": 900,
    },
    {
        "name": "Inst_20_MultiDrone_HighStationCost",
        "generator": "clustered_multi_drone",
        "drone": "Inspire",
        "cluster_centers": [(11500.0, 11500.0), (-11500.0, 11500.0), (11500.0, -11500.0)],
        "criticals_per_cluster": [5, 5, 5],
        "critical_radius": 340.0,
        "gateway_offset": 1800.0,
        "side_station_offset": 850.0,
        "station_cost": 1400,
    },
]


def _write_instance_file(name, drone_specs, critical_points, station_points, output_dir, station_cost):
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{name}.txt")
    dimension = 1 + len(critical_points) + len(station_points)

    with open(filename, "w") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: UAV-COVERAGE-RECHARGE\n")
        f.write(f"DIMENSION: {dimension}\n")
        f.write(f"CRITICAL_LOCATIONS: {len(critical_points)}\n")
        f.write(f"CANDIDATE_STATIONS: {len(station_points)}\n")
        f.write(f"BATTERY_CAPACITY: {drone_specs['capacity']}\n")
        f.write(f"DRONE_COST: {drone_specs['cost']}\n")
        f.write(f"STATION_COST: {station_cost}\n")
        f.write("NODE_DATA_SECTION\n")
        f.write("# ID\tTYPE\tX\t\tY\t\tBETA\n")
        f.write("0\t0\t0.0\t\t0.0\t\t0\n")

        current_id = 1
        for x, y in critical_points:
            f.write(f"{current_id}\t1\t{x}\t\t{y}\t\t{drone_specs['beta']}\n")
            current_id += 1

        for x, y in station_points:
            f.write(f"{current_id}\t2\t{x}\t\t{y}\t\t0\n")
            current_id += 1

        f.write("EOF\n")


def _generate_ring_points(center_x, center_y, count, radius):
    points = []
    angle_shift = random.uniform(0, 2 * pi)

    for idx in range(count):
        angle = angle_shift + (2 * pi * idx / count)
        radial_factor = 1.0 + random.uniform(-0.12, 0.12)
        jitter_x = random.uniform(-35.0, 35.0)
        jitter_y = random.uniform(-35.0, 35.0)
        x = center_x + cos(angle) * radius * radial_factor + jitter_x
        y = center_y + sin(angle) * radius * radial_factor + jitter_y
        points.append((round(x, 1), round(y, 1)))

    return points


def _generate_clustered_station_points(config):
    station_points = []
    gateway_offset = config["gateway_offset"]
    side_station_offset = config["side_station_offset"]
    central_decoy_ratio = config.get("central_decoy_ratio")

    for center_x, center_y in config["cluster_centers"]:
        norm = hypot(center_x, center_y)
        unit_x = center_x / norm
        unit_y = center_y / norm
        perp_x = -unit_y
        perp_y = unit_x

        main_station = (
            round(center_x - gateway_offset * unit_x, 1),
            round(center_y - gateway_offset * unit_y, 1),
        )
        alt_station = (
            round(center_x - (gateway_offset - 250.0) * unit_x + side_station_offset * perp_x, 1),
            round(center_y - (gateway_offset - 250.0) * unit_y + side_station_offset * perp_y, 1),
        )

        station_points.extend([main_station, alt_station])

        if central_decoy_ratio is not None:
            station_points.append(
                (
                    round(center_x * central_decoy_ratio, 1),
                    round(center_y * central_decoy_ratio, 1),
                )
            )

    return station_points


def _generate_clustered_multi_drone_instance_file(config, output_dir="instances"):
    drone_specs = DRONES[config["drone"]]
    critical_points = []

    for (center_x, center_y), count in zip(
        config["cluster_centers"], config["criticals_per_cluster"], strict=True
    ):
        critical_points.extend(
            _generate_ring_points(center_x, center_y, count, config["critical_radius"])
        )

    station_points = _generate_clustered_station_points(config)
    station_cost = config.get("station_cost", STATION_COST)

    _write_instance_file(
        config["name"],
        drone_specs,
        critical_points,
        station_points,
        output_dir,
        station_cost,
    )


def generate_instance_file(config, output_dir="instances"):
    if config.get("generator") == "clustered_multi_drone":
        _generate_clustered_multi_drone_instance_file(config, output_dir)
        return

    name = config["name"]
    num_c = config["num_c"]
    num_r = config["num_r"]
    drone_type = config["drone"]
    grid_size = config["grid_size"]
    drone_specs = DRONES[drone_type]
    critical_points = []
    station_points = []

    for _ in range(num_c):
        critical_points.append(
            (
                round(random.uniform(0, grid_size), 1),
                round(random.uniform(0, grid_size), 1),
            )
        )

    for _ in range(num_r):
        station_points.append(
            (
                round(random.uniform(0, grid_size), 1),
                round(random.uniform(0, grid_size), 1),
            )
        )

    _write_instance_file(
        name,
        drone_specs,
        critical_points,
        station_points,
        output_dir,
        config.get("station_cost", STATION_COST),
    )


if __name__ == "__main__":

    random.seed(42)

    print("Gerando instâncias...")
    for cfg in CONFIGURATIONS:
        generate_instance_file(cfg)
        print(f" -> {cfg['name']}.txt gerada com sucesso.")
    print("\nGerando instâncias multi-drone...")
    for cfg in MULTI_DRONE_CONFIGURATIONS:
        generate_instance_file(cfg)
        print(f" -> {cfg['name']}.txt gerada com sucesso.")
    total_instances = len(CONFIGURATIONS) + len(MULTI_DRONE_CONFIGURATIONS)
    print(f"\nConcluído! {total_instances} instâncias foram salvas na pasta 'instances'.")
