import PPO_trainer as agent_trainer
import PPO_testing as agent_evaluator

experiment_config = {
    "name": "trained_agent",
    "description": """The baseline agent trained on default parameters.""",
    "evaluation_only": True, # for testing only, False for training
    "results_file": "testing_results",
    
    "behavior": {
        "deterministic_policy": False,
        "enable_action_masking": True,
        "manual_override": False,
    },
    
    "sensors": {
        "type": "sonar",
        "ray_count": 4,
        "beam_width": 0.1,
        "precision": -1.0,
        "noise_factor": 0.025,
        "detection_range": 100
    },
    
    "progression_stages": {
        "stage_1": {"environment": "random", "obstacle_count": 10,
                   "target_distance_min": 10, "target_distance_max": 10, "learning_steps": 262144},
        "stage_2": {"environment": "corridor", "obstacle_count": 2,
                   "target_distance_min": 2, "target_distance_max": 2, "learning_steps": 524288},
        "stage_3": {"environment": "corridor", "obstacle_count": 4,
                   "target_distance_min": 3, "target_distance_max": 3, "learning_steps": 524288},
        "stage_4": {"environment": "corridor", "obstacle_count": 6,
                   "target_distance_min": 4, "target_distance_max": 4, "learning_steps": 524288},
        "stage_5": {"environment": "corridor", "obstacle_count": 10,
                   "target_distance_min": 5, "target_distance_max": 6, "learning_steps": 524288},
        "stage_6": {"environment": "random", "obstacle_count": 25,
                   "target_distance_min": 10, "target_distance_max": 12, "learning_steps": 1048576}
    },
    
    "episode": {
        "max_steps": 16_384,
        "history_timesteps": 1,
        "history_seconds": 1,
        "include_actions": True,
        "collision_recovery": 4096,
        "target_proximity": 0.1,
    },
    
    "evaluation": {
        "episodes_per_stage": 100,
        "disabled_sensors": []
    },
    
    "rewards": {
        "distance_factor": 1.0,
        "heading_factor": 1.0,
        "proximity_penalty": 10.0,
        "success_bonus": 1000.0,
        "collision_penalty": 100.0,
        "motion_efficiency": 0.0,
        "velocity_incentive": 0.0,
    },
    
    "learning": {
        "batch_interval": 2048,
        "minibatch_size": 64,
        "discount": 0.999,
        "lambda_gae": 0.95,
        "kl_target": None,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.001,
        "learning_rate": lambda f: f * 0.0003,
        "network_architecture": {
            "policy": [1024, 512, 256],
            "value": [2048, 1024, 512]
        }
    },
    
    "environment": {
        "width": 7,
        "height": 7,
        "cell_dimensions": None
    },
    
    "random_seed": 1
}

ds_params = {
    "ds_type": experiment_config["sensors"]["type"],
    "ds_n_rays": experiment_config["sensors"]["ray_count"],
    "ds_aperture": experiment_config["sensors"]["beam_width"],
    "ds_resolution": experiment_config["sensors"]["precision"],
    "ds_noise": experiment_config["sensors"]["noise_factor"],
    "max_ds_range": experiment_config["sensors"]["detection_range"]
}

difficulty_dict = {}
for i, (key, stage) in enumerate(experiment_config["progression_stages"].items()):
    difficulty_dict[f"diff_{i}"] = {
        "type": stage["environment"],
        "number_of_obstacles": stage["obstacle_count"],
        "min_target_dist": stage["target_distance_min"],
        "max_target_dist": stage["target_distance_max"],
        "total_timesteps": stage["learning_steps"]
    }

environment = agent_trainer.run(
    experiment_name=experiment_config["name"],
    experiment_description=experiment_config["description"],
    manual_control=experiment_config["behavior"]["manual_override"],
    only_test=experiment_config["evaluation_only"],
    maximum_episode_steps=experiment_config["episode"]["max_steps"],
    step_window=experiment_config["episode"]["history_timesteps"],
    seconds_window=experiment_config["episode"]["history_seconds"],
    add_action_to_obs=experiment_config["episode"]["include_actions"],
    ds_params=ds_params,
    reset_on_collisions=experiment_config["episode"]["collision_recovery"],
    on_tar_threshold=experiment_config["episode"]["target_proximity"],
    target_dist_weight=experiment_config["rewards"]["distance_factor"],
    target_angle_weight=experiment_config["rewards"]["heading_factor"],
    dist_sensors_weight=experiment_config["rewards"]["proximity_penalty"],
    target_reach_weight=experiment_config["rewards"]["success_bonus"],
    collision_weight=experiment_config["rewards"]["collision_penalty"],
    smoothness_weight=experiment_config["rewards"]["motion_efficiency"],
    speed_weight=experiment_config["rewards"]["velocity_incentive"],
    net_arch=experiment_config["learning"]["network_architecture"],
    n_steps=experiment_config["learning"]["batch_interval"],
    batch_size=experiment_config["learning"]["minibatch_size"],
    gamma=experiment_config["learning"]["discount"],
    gae_lambda=experiment_config["learning"]["lambda_gae"],
    target_kl=experiment_config["learning"]["kl_target"],
    vf_coef=experiment_config["learning"]["value_coefficient"],
    ent_coef=experiment_config["learning"]["entropy_coefficient"],
    lr_rate=experiment_config["learning"]["learning_rate"],
    difficulty_dict=difficulty_dict,
    seed=experiment_config["random_seed"],
    map_w=experiment_config["environment"]["width"],
    map_h=experiment_config["environment"]["height"],
    cell_size=experiment_config["environment"]["cell_dimensions"]
)

environment.ds_denial_list = experiment_config["evaluation"]["disabled_sensors"]

test_seed = experiment_config["random_seed"] + 1
agent_evaluator.run(
    experiment_config["name"],
    environment,
    experiment_config["behavior"]["deterministic_policy"],
    experiment_config["behavior"]["enable_action_masking"],
    testing_results_filename=experiment_config["results_file"],
    tests_per_difficulty=experiment_config["evaluation"]["episodes_per_stage"],
    seed=test_seed
)