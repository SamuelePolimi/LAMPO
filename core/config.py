from core.augmented_tasks.tasks import ReachTarget, CloseDrawer, WaterPlants
from core.rl_bench_box import RLBenchBox, Reacher2D, ObstacleReacher2d
from core.lab_connection import TCPTask

config = {
    "tcp": {
        "task_box": lambda headless: TCPTask(5056, 20),
        "n_features": 20,
        "n_cluster": 4,
        "latent_dim": 2,
        "state_dim": 3,
        "n_samples": 50
    },
    "tcp_pouring": {
        "task_box": lambda headless: TCPTask(5056, 20),
        "n_features": 20,
        "n_cluster": 4,
        "latent_dim": 4,
        "state_dim": 1,
        "n_samples": 100
    },
    "reacher2d_1": {
        "task_box": lambda headless: Reacher2D(20, 1, headless),
        "n_features": 20,
        "n_cluster": 1,
        "latent_dim": 2,
        "state_dim": 2,
        "n_samples": 100
    },
    "reacher2d_2": {
        "task_box": lambda headless: Reacher2D(20, 2, headless),
        "n_features": 20,
        "n_cluster": 2,
        "latent_dim": 2,
        "state_dim": 2,
        "n_samples": 100
    },
    "reacher2d_3": {
        "task_box": lambda headless: Reacher2D(20, 3, headless),
        "n_features": 20,
        "n_cluster": 3,
        "latent_dim": 2,
        "state_dim": 2,
        "n_samples": 100
    },
    "reacher2d_4": {
        "task_box": lambda headless: Reacher2D(20, 4, headless),
        "n_features": 20,
        "n_cluster": 4,
        "latent_dim": 2,
        "state_dim": 2,
        "n_samples": 100
    },
    # "reacher2d_obstacle": {
    #     "task_box": lambda headless: ObstacleReacher2d(20, headless),
    #     "n_features": 20,
    #     "n_cluster": 25,
    #     "latent_dim": 5,
    #     "state_dim": 2,
    #     "n_samples": 9000
    # },
    "reacher2d_obstacle": {
        "task_box": lambda headless: ObstacleReacher2d(20, headless),
        "n_features": 20,
        "n_cluster": 40,
        "latent_dim": 5,
        "state_dim": 2,
        "n_samples": 16500
    },
    "close_drawer": {
        "task_class": CloseDrawer,  # TODO: remove
        "task_box": lambda headless: RLBenchBox(CloseDrawer, 94, 20, headless),
        "n_cluster": 10,
        "latent_dim": 4,
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
        "n_samples": 1000
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
