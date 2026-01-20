import torch
import random
import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time


# go back .. directory
import sys
sys.path.append("../")

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss


from pipeline.run_pipeline import load_and_sample_datasets
from pipeline.run_pipeline import filter_data
from pipeline.submodules.generate_directions import get_mean_activations, get_activations_array
from pipeline.submodules.evaluate_loss import compute_loss_for_target_strings

import argparse
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def main():
    # model, direction_num
    # for each model save each direction seed and use it if exists
    # for each model and direction seed save its results in a separate directory

    parser = argparse.ArgumentParser(description="Bias Direction Pipeline")

    parser.add_argument("--model_path", type=str, default="meta-llama/llama-2-7b-chat-hf")
    parser.add_argument("--direction_num", type=int, default=10)
    parser.add_argument("--bias_data_path", type=str, default="main_data/quiz_bias")
    parser.add_argument("--results_dir", type=str, default="results_multi")

    args = parser.parse_args()

    model_path = args.model_path
    direction_num = args.direction_num
    bias_data_path = args.bias_data_path
    results_dir = args.results_dir


    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    model_base = construct_model_base(cfg.model_path)

    # load data
    bias_data_dict = {}
    # loop through all subject directories in bias_data_path
    for subject_category in os.listdir(bias_data_path):
        subject_category_path = os.path.join(bias_data_path, subject_category)
        # loop through all types of subjects
        for subject_type in os.listdir(subject_category_path):
            subject_type_path = os.path.join(subject_category_path, subject_type)
            # loop through all files in the subject type directory
            for subject in os.listdir(subject_type_path):
                file_path = os.path.join(subject_type_path, subject)
                real_subject = subject.replace('.csv', '')  # remove .csv extension for key
                key = f"{subject_category}${subject_type}${real_subject}"
                # read the file and store it in the bias_data_dict
                df = pd.read_csv(file_path)
                string_list = df['goal'].tolist()
                bias_data_dict[key] = string_list

    # load harmful and harmless data
    # # Load the WildJailbreak training set (DEPRECATED)
    # dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False, token=hf_token)
    # # Filter harmful examples
    # vanilla_harmful = dataset["train"].filter(lambda example: example["data_type"] == "vanilla_harmful")["vanilla"]
    # # Filter benign examples
    # vanilla_benign = dataset["train"].filter(lambda example: example["data_type"] == "vanilla_benign")["vanilla"]
    
    # del dataset  # Free up memory

    # create results directory for completions
    os.makedirs(results_dir, exist_ok=True)

    # create model subdirectory
    model_dir = os.path.join(results_dir, model_alias)
    os.makedirs(model_dir, exist_ok=True)

    # create quiz_bias subdirectory
    quiz_bias_dir = os.path.join(model_dir, "quiz_bias")
    os.makedirs(quiz_bias_dir, exist_ok=True)

    # create direction_gen_time.csv as df
    direction_gen_time_file = os.path.join(model_dir, "direction_gen_time.csv")
    if not os.path.exists(direction_gen_time_file):
        # create file with header
        with open(direction_gen_time_file, 'w') as f:
            f.write("direction_seed,time_taken\n")


    for direction_seed in range(direction_num):
        print(f"Processing direction seed {direction_seed}...")
        # set random seed for reproducibility of direction

        # check if direction seed already exists in direction_gen_time_file
        with open(direction_gen_time_file, 'r') as f:
            lines = f.readlines()
            # check if any row starts with direction_seed
            if any(line.startswith(f"{direction_seed},") for line in lines):
                print(f"Direction seed {direction_seed} already exists. Skipping generation.")
                continue

        # (1) calculate refusal direction
        random.seed(direction_seed)
        harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
        harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
        
        # (DEPRECATED) 
        # harmless_train = random.sample(vanilla_benign, cfg.n_train)
        # harmful_train = random.sample(vanilla_harmful, cfg.n_train)
        
        start_time = time.time()
        candidate_refusal_direction = generate_directions(
                model_base,
                harmful_train,
                harmless_train,
                artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))
        _, layer, direction = select_direction(
                model_base,
                harmful_train,
                harmless_train,
                candidate_refusal_direction,
                artifact_dir=os.path.join(cfg.artifact_path(), "select_direction"),
                induce_refusal_threshold=-30.0,# reduced threshold to avoid crashes
            )
        baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
        ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
        actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []
        time_taken = time.time() - start_time
        print(f"Generated direction hooks in {time_taken:.2f} seconds")

        # save generation time of direction
        with open(direction_gen_time_file, 'a') as f:
            f.write(f"{direction_seed},{time_taken}\n")

        # (2) generate completions for bias data 
        # collapse bias data into a single list of strings
        for key, strings in bias_data_dict.items():
            subject_category, subject_type, subject = key.split('$')
            print("=========================")
            print(f"Processing {subject_category} - {subject_type} - {subject}")


            # save completions to file under quiz_bias_dir/direction_seed/subject_category/subject_type/subject_{baseline/ablation/actadd}.json
            os.makedirs(os.path.join(quiz_bias_dir, str(direction_seed)), exist_ok=True)
            os.makedirs(os.path.join(quiz_bias_dir, str(direction_seed), subject_category), exist_ok=True)
            os.makedirs(os.path.join(quiz_bias_dir, str(direction_seed), subject_category, subject_type), exist_ok=True)

            baseline_path = os.path.join(quiz_bias_dir, str(direction_seed), subject_category, subject_type, f"{subject}_baseline.json")
            ablation_path = os.path.join(quiz_bias_dir, str(direction_seed), subject_category, subject_type, f"{subject}_ablation.json")
            actadd_path = os.path.join(quiz_bias_dir, str(direction_seed), subject_category, subject_type, f"{subject}_actadd.json")


            # if files already exist, skip generation
            if os.path.exists(baseline_path) and os.path.exists(ablation_path) and os.path.exists(actadd_path):
                print(f"Completions already exist for {subject_category} - {subject_type} - {subject}. Skipping generation.")
                continue
            else:
                print(f"Generating completions for {subject_category} - {subject_type} - {subject}...")
                # make this list into a list of dicts with 2 keys: 'instruction' and 'category'
                bias_harmful_test = [{'instruction': string, 'category': key} for string, key in zip(strings, [key] * len(strings))]

                # baseline (x)
                start_time = time.time()
                completions_baseline = model_base.generate_completions(bias_harmful_test, fwd_pre_hooks=baseline_fwd_pre_hooks, fwd_hooks=baseline_fwd_hooks, max_new_tokens=cfg.max_new_tokens)
                time_taken = time.time() - start_time
                print(f"Generated baseline completions in {time_taken:.2f} seconds")
                for completion in completions_baseline:
                    completion['time_taken'] = time_taken / len(completions_baseline)
                with open(baseline_path, 'w') as f:
                    json.dump(completions_baseline, f, indent=4)

                # ablation (x-rr^Tx)
                start_time = time.time()
                completions_ablation = model_base.generate_completions(bias_harmful_test, fwd_pre_hooks=ablation_fwd_pre_hooks, fwd_hooks=ablation_fwd_hooks, max_new_tokens=cfg.max_new_tokens)
                time_taken = time.time() - start_time
                print(f"Generated ablation completions in {time_taken:.2f} seconds")
                for completion in completions_ablation:
                    completion['time_taken'] = time_taken / len(completions_ablation)
                with open(ablation_path, 'w') as f:
                    json.dump(completions_ablation, f, indent=4)

                # actadd (x-r)
                start_time = time.time()
                completions_actadd = model_base.generate_completions(bias_harmful_test, fwd_pre_hooks=actadd_fwd_pre_hooks, fwd_hooks=actadd_fwd_hooks, max_new_tokens=cfg.max_new_tokens)
                time_taken = time.time() - start_time
                print(f"Generated actadd completions in {time_taken:.2f} seconds")
                for completion in completions_actadd:
                    completion['time_taken'] = time_taken / len(completions_actadd)
                with open(actadd_path, 'w') as f:
                    json.dump(completions_actadd, f, indent=4)

        # free up memory
        del completions_baseline, completions_ablation, completions_actadd



if __name__ == "__main__":
    main()