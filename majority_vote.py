import json
import os
import csv
from copy import deepcopy
from tqdm import tqdm


VAL_PAIRS = "data/data_parsed/pairs_val.csv"

VAL_SUBMISSIONS_PATHS = [
    "BEST/e5-large-v2/submission_val_checkpoint_e5-large-v2_epoch=5_acc=91.61_epoch=0_acc=91.61.json",
    "BEST/multilingual-e5-large-instruct/submission_val_checkpoint_multilingual-e5-large-instruct_epoch=5_acc=92.09_epoch=0_acc=92.09.json",
    "BEST/e5-large-v2_hard/submission_val_e5-large-v2_epoch=3_acc=95.66.json",
    "BEST/multilingual-e5-large-instruct_hard/submission_val_multilingual-e5-large-instruct_epoch=3_acc=95.3.json",
]

DEV_SUBMISSIONS_PATHS = [
    "BEST/e5-large-v2/submission_dev_checkpoint_e5-large-v2_epoch=5_acc=91.61_epoch=0_acc=91.61.json",
    "BEST/multilingual-e5-large-instruct/submission_dev_checkpoint_multilingual-e5-large-instruct_epoch=5_acc=92.09_epoch=0_acc=92.09.json",
    "BEST/e5-large-v2_hard/submission_dev_e5-large-v2_epoch=3_acc=95.66.json",
    "BEST/multilingual-e5-large-instruct_hard/submission_dev_multilingual-e5-large-instruct_epoch=3_acc=95.3.json",
]

DEV_SAVE_PATH = "submissions_final/submission_dev_majority_vote"
os.makedirs(DEV_SAVE_PATH, exist_ok=True)

TEST_SUBMISSIONS_PATHS = [
    "BEST/e5-large-v2/submission_test_checkpoint_e5-large-v2_epoch=5_acc=91.61_epoch=5.json",
    "BEST/multilingual-e5-large-instruct/submission_test_checkpoint_multilingual-e5-large-instruct_epoch=5_acc=92.09_epoch=5.json",
    "BEST/e5-large-v2_hard/submission_test_e5-large-v2_epoch=3.json",
    "BEST/multilingual-e5-large-instruct_hard/submission_test_multilingual-e5-large-instruct_epoch=3.json",
]

TEST_SAVE_PATH = "submissions_final/submission_test_majority_vote"
os.makedirs(TEST_SAVE_PATH, exist_ok=True)


def success10(predictions, pairs):
    success = {k : 0 for k in predictions.keys()}
    
    for pair in pairs:
        post_id = int(pair["post_id"])
        fact_check_id = int(pair["fact_check_id"])
        if post_id in predictions and fact_check_id in predictions[post_id]:
            success[post_id] = 1

    return round(sum(list(success.values())) / len(success) * 100, 2)


all_configs = []
def compute_all_configs(n, curr_config):
    if n == 1:
        all_configs.append(deepcopy(curr_config + [10 - sum(curr_config)]))
        return

    for i in range(10 - sum(curr_config) + 1):
        compute_all_configs(n-1, curr_config + [i])


def compute_pred(config, task, all_submissions):
    sum_cos_sim = {}
    for i in range(len(all_submissions)):
        for post in all_submissions[i][task]:
            if post not in sum_cos_sim:
                sum_cos_sim[post] = {}
            for x in all_submissions[i][task][post]:
                if x[0] not in sum_cos_sim[post]:
                    sum_cos_sim[post][x[0]] = 0
                sum_cos_sim[post][x[0]] += x[1] * config[i]
    
    votes_per_post = {}
    for k, v in sum_cos_sim.items():
        votes_per_post[k] = sorted(list(v.items()), key=lambda x : x[1], reverse=True)

    pred = {}
    for id, ls in votes_per_post.items():
        pred[int(id)] = [x[0] for x in ls[:10]]
    
    return pred


def find_best_config():
    with open(VAL_PAIRS, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file) 
        val_pairs = list(reader)

    all_submissions = []
    for submission in VAL_SUBMISSIONS_PATHS:
        with open(submission, "r") as file:
            data = json.load(file)
            all_submissions.append(data)
    
    compute_all_configs(len(all_submissions), [])
    print(len(all_configs))

    best_configs = {}
    for task in all_submissions[0].keys():
        best_acc = 0
        best_config = None
        for config in tqdm(all_configs):
            val_pred = compute_pred(config, task, all_submissions)
            acc = success10(val_pred, val_pairs)
            if acc > best_acc:
                best_acc = acc
                best_config = config

        best_configs[task] = (best_config, best_acc)

    return best_configs


def get_mean_config(best_configs):
    mean_config = [0 for _ in range(len(TEST_SUBMISSIONS_PATHS))]
    cnt = 0
    for task in best_configs:
        if "_" not in task:
            continue
        cnt += 1
        for i, x in enumerate(best_configs[task][0]):
            mean_config[i] += x
    
    mean_config = [x / cnt for x in mean_config]
    return mean_config


def make_majority_vote_submission(best_configs, submissions_path, save_path):
    all_submissions = []
    for submission in submissions_path:
        with open(submission, "r") as file:
            data = json.load(file)
            all_submissions.append(data)
    
    monolingual_submission = {}
    crosslingual_submission = {}

    for task in all_submissions[0].keys():
        if task in best_configs:
            curr_config = best_configs[task][0]
        else:
            curr_config = get_mean_config(best_configs)
            print(f"{save_path} -> {task} : {curr_config}")
        pred = compute_pred(curr_config, task, all_submissions)
        if "_" in task:
            for id, ls in pred.items():
                monolingual_submission[id] = ls
        else:
            for id, ls in pred.items():
                crosslingual_submission[id] = ls

    with open(os.path.join(save_path, "monolingual_predictions.json"), "w") as file:
        json.dump(monolingual_submission, file)

    with open(os.path.join(save_path, "crosslingual_predictions.json"), "w") as file:
        json.dump(crosslingual_submission, file)
    
    with open(os.path.join(save_path, "best_configs.json"), "w") as file:
        json.dump(best_configs, file)

    with open(os.path.join(save_path, "submissions_for_majority_vote.txt"), "w") as file:
        for x in submissions_path:
            file.write(f"{x}\n")




if __name__ == "__main__":
    best_configs = find_best_config()
    print(best_configs)
    make_majority_vote_submission(best_configs, DEV_SUBMISSIONS_PATHS, DEV_SAVE_PATH)
    make_majority_vote_submission(best_configs, TEST_SUBMISSIONS_PATHS, TEST_SAVE_PATH)

