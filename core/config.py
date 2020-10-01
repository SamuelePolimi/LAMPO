from rlbench.tasks import WaterPlants, CloseDrawer
from core.augmented_tasks.tasks import ReachTarget
from core.rl_bench_box import RLBenchBox
from core.lab_connection import TCPTask

config = {
    "tcp": {
        "task_box": lambda headless: TCPTask(5056, 20),
        "n_features": 20,
        "n_cluster": 6,
        "latent_dim": 3,
        "state_dim": 7,
        "n_samples": 200
    },
    "close_drawer": {
        "task_class": CloseDrawer,  # TODO: remove
        "task_box": lambda headless: RLBenchBox(CloseDrawer, 94, 20, headless),
        "n_cluster": 10,
        "latent_dim": 5,
        "n_features": 20,   # TODO: remove
        "state_dim": 94,    # TODO: remove
        "n_samples": 200
    },
    "water_plants": {
        "task_class": WaterPlants,  # TODO: remove
        "task_box": lambda headless: RLBenchBox(WaterPlants, 84, 20, headless),
        "n_cluster": 2,
        "latent_dim": 10,
        "n_features": 20,  # TODO: remove
        "state_dim": 84,  # TODO: remove
        "n_samples": 200
    },
    "reach_target": {
        "task_class": ReachTarget,
        "task_box": lambda headless: RLBenchBox(ReachTarget, 3, 20, headless),
        "n_cluster": 6,
        "latent_dim": 3,
        "n_features": 20,
        "state_dim": 3,
        "n_samples": 1000
    }
}
