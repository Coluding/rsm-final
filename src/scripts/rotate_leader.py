from typing import List
import os

from src.environment import JavaSimulator
import matplotlib.pyplot as plt


class BaseRotationStrategy:
    def __init__(self, config_dir: str, conn: JavaSimulator, num_steps: int = 1000):
        self.conn = conn
        self.initial_config = read_out_initial_config(config_dir)
        self.num_steps = num_steps

        self.coordinator: int = 0

    def rotate(self):
        pass

class StaticCoordinatorRotator(BaseRotationStrategy):
    def __init__(self, config_dir: str, conn: JavaSimulator, num_steps: int = 1000,
                 coordinator: int = 0):
        super().__init__( config_dir, conn, num_steps)
        self.config_dir = config_dir
        self.num_steps = num_steps
        self.initial_config = read_out_initial_config(config_dir)
        self.initial_config_identifier = [self.conn.get_node_id(x) for x in self.initial_config]
        self.coordinator = coordinator
        self.coordinator_identifier = self.conn.get_node_id(self.coordinator)
        assert coordinator in self.initial_config, "Coordinator not in initial config"

    def rotate(self):
        return self.coordinator

    def __str__(self):
        return (f"Static-initial config {self.initial_config_identifier} "
                f"and coordinator {self.coordinator_identifier}")


def read_out_initial_config(config_dir: str):
    with open(config_dir + "server0/xmr/config/system.config", "r") as f: #TODO: Change this to a more general path
        lines = f.readlines()
        for line in lines:
            if "system.initial.view = " in line:
                initial_config =  line.split(" = ")[1].split(",")
                break

        return [int(x) for x in initial_config]


def setup_conn(jar_path: str = "/home/lukas/Projects/emusphere/simulator-xmr/target/simulator-xmr-0.0.1-SNAPSHOT-jar-with-dependencies.jar",
               base_configuration_directory_simulator: str = "../../ressources/run_configs/400_steps",
               config_dir: str = "/simurun/",
               jvm_options: list[str] = ['-Djava.security.properties=/home/lukas/flusim/simurun/server0/xmr/config/java.security'],
               ):
    conn = JavaSimulator(jvm_options=jvm_options,
        jar_path=jar_path,
        base_configuration_directory_simulator=base_configuration_directory_simulator,
        deterministic_config_dir=config_dir

    )

    return conn

def leader_run(conn: JavaSimulator, coordinator_rotator: BaseRotationStrategy, ):
    latencies = []

    for i in range(coordinator_rotator.num_steps):
        coordinator = coordinator_rotator.rotate()
        res = conn.coordinator_swap_step(coordinator)
        latencies.append(res.mean_delay)

    return latencies, coordinator_rotator

def plot_leader_run(latencies: List[float], coordinator_rotator: BaseRotationStrategy,
                    save_dir: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(latencies)
    ax.set_xlabel("Step")
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_ylabel("Latency")
    ax.set_title(f"Latency development of {coordinator_rotator}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_dir + f"{coordinator_rotator}-{coordinator_rotator.num_steps}.png")

    plt.show()


def leader_run_different_starts(conns: List[JavaSimulator], coordinator_rotators: List[BaseRotationStrategy]):
    latencies = {str(coordinator_rotator): [] for coordinator_rotator in coordinator_rotators}
    for conn, coordinator_rotator in zip(conns, coordinator_rotators):
        latencies[str(coordinator_rotator)], _ = leader_run(conn, coordinator_rotator)

    return latencies

def leader_run_different_starts_static(base_configuration_directory_simulator: str, config_dir: str,
                                       num_steps: int = 1000, start_dirs: List[str] = None):
    if start_dirs is None:
        start_dirs = os.listdir(base_configuration_directory_simulator)
    conns = [setup_conn(base_configuration_directory_simulator=base_configuration_directory_simulator,
                        config_dir=start_dir) for start_dir in start_dirs]
    coordinator_rotators = [StaticCoordinatorRotator(conn=conn, config_dir=base_configuration_directory_simulator + start_dir,
                                                   coordinator=0, num_steps=num_steps) for conn, start_dir in zip(conns, start_dirs)]

    return leader_run_different_starts(conns, coordinator_rotators)

def plot_leader_run_different_starts(latencies: dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, value in latencies.items():
        ax.plot(value, label=key)
    ax.set_xlabel("Step")
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_ylabel("Latency")
    ax.set_title("Latency development of different starting configurations")
    ax.legend()
    plt.show()


def plot_different_starting_points(coord_lats: dict, title: str , save_dir: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, value in coord_lats.items():
        ax.plot(value, label=key)
    ax.set_xlabel("Step")
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_ylabel("Latency")
    ax.set_title(title)
    # Place legend outside the plot on the right
    ax.legend(bbox_to_anchor=(0.75, 1), loc='upper left')

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_dir + f"{title}.png")

    plt.show()



if __name__ == "__main__":
    base_configuration_directory_simulator = "../../ressources/run_configs/400_steps"
    config_dir = "/simurun_E/"

    lats = {}
    conn = setup_conn(base_configuration_directory_simulator=base_configuration_directory_simulator,
                      config_dir=config_dir)

    active_nodes = read_out_initial_config(base_configuration_directory_simulator + config_dir)

    print(active_nodes)
    for coordinator in active_nodes:
        conn = setup_conn(base_configuration_directory_simulator=base_configuration_directory_simulator,
                          config_dir=config_dir)
        coordinator_rotator = StaticCoordinatorRotator(conn=conn, config_dir=base_configuration_directory_simulator + config_dir,
                                                       coordinator=coordinator, num_steps=400)
        latencies, _ = leader_run(conn, coordinator_rotator)
        lats[f"Static Leader: {conn.get_node_id(coordinator)}"] = latencies

    plot_different_starting_points(lats, f"Initial config: {coordinator_rotator.initial_config_identifier}",
                                   save_dir="plots/")

