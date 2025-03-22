import numpy as np
import torch
import random
import csv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

def get_valid_actions(env):
    return env.get_action_mask()

def run(experiment_name, env, deterministic, use_masking, testing_results_filename=None, tests_per_difficulty=100,
        seed=None):
    test_scenarios = {
        "scenario_1": {"environment": "corridor", "obstacle_count": 0,
                       "target_distance_min": 1, "target_distance_max": 3},
        "scenario_2": {"environment": "corridor", "obstacle_count": 2,
                       "target_distance_min": 2, "target_distance_max": 2},
        "scenario_3": {"environment": "corridor", "obstacle_count": 4,
                       "target_distance_min": 3, "target_distance_max": 3},
        "scenario_4": {"environment": "corridor", "obstacle_count": 6,
                       "target_distance_min": 4, "target_distance_max": 4},
        "scenario_5": {"environment": "corridor", "obstacle_count": 8,
                       "target_distance_min": 5, "target_distance_max": 5},
        "scenario_6": {"environment": "random", "obstacle_count": 25,
                       "target_distance_min": 5, "target_distance_max": 12}
    }
    
    scenario_to_difficulty = {
        "scenario_1": "diff_0", 
        "scenario_2": "diff_1", 
        "scenario_3": "diff_2",
        "scenario_4": "diff_3", 
        "scenario_5": "diff_4", 
        "scenario_6": "diff_5"
    }
    
    difficulty_dict = {}
    for scenario_key, scenario in test_scenarios.items():
        diff_key = scenario_to_difficulty[scenario_key]
        difficulty_dict[diff_key] = {
            "type": scenario["environment"],
            "number_of_obstacles": scenario["obstacle_count"],
            "min_target_dist": scenario["target_distance_min"],
            "max_target_dist": scenario["target_distance_max"]
        }
    
    difficulty_keys = list(difficulty_dict.keys())

    env.reset_on_collisions = -1
    env.set_maximum_episode_steps(env.maximum_episode_steps * 2)
    
    env.set_reward_weight_dict(
        target_distance_weight=0.0, 
        target_angle_weight=0.0, 
        dist_sensors_weight=0.0,
        target_reach_weight=1000.0, 
        collision_weight=1.0,
        smoothness_weight=0.0, 
        speed_weight=0.0
    )
    
    experiment_dir = f"./experiments/{experiment_name}"
    model_path = experiment_dir + f"/{experiment_name}_diff_5_agent"

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    try:
        model = MaskablePPO.load(model_path)
    except FileNotFoundError:
        model_path += ".zip"
        model = MaskablePPO.load(model_path)

    env.set_difficulty(difficulty_dict["diff_5"], "diff_5")
    print("################### STANDARD EVALUATION STARTED ###################")
    print(f"Running standard evaluation for {tests_per_difficulty} episodes in complex scenario")
    print(f"Model: {experiment_name}, Deterministic: {deterministic}")
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=tests_per_difficulty,
        deterministic=deterministic, 
        use_masking=use_masking
    )
    
    print("################### STANDARD EVALUATION COMPLETED ###################")
    print(f"Model: {experiment_name}, Deterministic: {deterministic}")
    print(f"Mean reward: {mean_reward:.2f}, Standard deviation: {std_reward:.2f}")

    print("################### DETAILED EVALUATION STARTED ###################")
    print(f"Model: {experiment_name}, Deterministic: {deterministic}")
    
    scenario_index = 0
    env.set_difficulty(
        difficulty_dict[difficulty_keys[scenario_index]], 
        key=difficulty_keys[scenario_index]
    )

    csv_headers = [experiment_name]
    for i in range(len(difficulty_keys)):
        for j in range(tests_per_difficulty):
            csv_headers.append(f"{difficulty_keys[i]}")

    episode_rewards = ["reward"]
    done_reasons = ["done_reason"]
    steps_row = ["steps"]
    
    if testing_results_filename is None:
        results_filename = "/testing_results.csv" if not deterministic else "/testing_results_det.csv"
    else:
        results_filename = f"/{testing_results_filename}.csv" if not deterministic else f"/{testing_results_filename}_det.csv"
    
    with open(experiment_dir + results_filename, 'w', encoding='UTF8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_headers)
        
        step_counter = 0
        reward_sum = 0.0
        scenario_tests_completed = 0
        observation = env.reset()
        success_count = 0
        
        while True:
            action_mask = get_valid_actions(env) if use_masking else None
            
            action, _ = model.predict(
                observation, 
                deterministic=deterministic, 
                action_masks=action_mask
            )
            
            observation, reward, done, info = env.step(action)
            step_counter += 1
            reward_sum += reward
            
            if done:
                episode_rewards.append(reward_sum)
                steps_row.append(step_counter)
                done_reasons.append(info['done_reason'])
                
                if info["done_reason"] == "reached target":
                    success_count += 1
                
                print(f"{experiment_name} - Episode reward: {reward_sum:.2f}, steps: {step_counter}")
                
                total_episodes_completed = (tests_per_difficulty * scenario_index) + (scenario_tests_completed + 1)
                total_episodes_planned = tests_per_difficulty * len(difficulty_keys)
                
                try:
                    success_rate = 100 * success_count / total_episodes_completed
                except ZeroDivisionError:
                    success_rate = 100 * success_count
                print(f"Success rate: {success_rate:.2f}%")
                
                progress_percent = (total_episodes_completed / total_episodes_planned) * 100.0
                print(f"Progress: {progress_percent:.2f}%, {total_episodes_completed}/{total_episodes_planned}")
                
                reward_sum = 0.0
                step_counter = 0
                
                scenario_tests_completed += 1
                if scenario_tests_completed == tests_per_difficulty:
                    scenario_index += 1
                    try:
                        env.set_difficulty(
                            difficulty_dict[difficulty_keys[scenario_index]], 
                            key=difficulty_keys[scenario_index]
                        )
                    except IndexError:
                        print("Evaluation complete.")
                        break
                    scenario_tests_completed = 0
                
                observation = env.reset()

        csv_writer.writerow(episode_rewards)
        csv_writer.writerow(done_reasons)
        csv_writer.writerow(steps_row)
        csv_writer.writerow(["standard_mean:", mean_reward, "standard_std:", std_reward])
    
    print("################### DETAILED EVALUATION COMPLETED ###################")
    
    print("Starting continuous demonstration - close Webots window to stop")
    step_counter = 0
    reward_sum = 0.0
    observation = env.reset()
    
    while True:
        action_mask = get_valid_actions(env)
        
        action, _ = model.predict(
            observation, 
            deterministic=deterministic, 
            action_masks=action_mask
        )
        
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        step_counter += 1
        
        if done:
            print(f"{experiment_name} - Demo episode reward: {reward_sum:.2f}, steps: {step_counter}")
            reward_sum = 0.0
            step_counter = 0
            observation = env.reset()