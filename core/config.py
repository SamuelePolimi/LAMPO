from rlbench.tasks import ReachTarget, WaterPlants, CloseDrawer


config = {
    "close_drawer": {
        "task_class": CloseDrawer,
        "n_cluster": 10,
        "latent_dim": 5,
        "n_features": 20,
        "state_dim": 94,
        "n_samples": 200
    },
    "water_plants": {
        "task_class": WaterPlants,
        "n_cluster": 2,
        "latent_dim": 10,
        "n_features": 20,
        "state_dim": 84,
        "n_samples": 200
    },
    "reach_target": {
        "task_class": ReachTarget,
        "n_cluster": 6,
        "latent_dim": 3,
        "n_features": 20,
        "state_dim": 3,
        "n_samples": 200
    }
}
