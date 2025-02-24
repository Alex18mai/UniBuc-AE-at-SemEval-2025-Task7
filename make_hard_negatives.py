import json
import csv
import os

# SUBMISSION_PATH = "submissions/submission_train_checkpoint_e5-large-v2_epoch=5_acc=91.61_epoch=0_acc=91.61.json"
SUBMISSION_PATH = "submissions/submission_train_checkpoint_multilingual-e5-large-instruct_epoch=5_acc=92.09_epoch=0_acc=92.09.json"
TRAIN_PAIRS_PATH = "data/data_parsed/pairs_train.csv"

# HARD_NEGATIVES_PATH = "hard_negatives/hard_negatives_e5-large-v2.json"
HARD_NEGATIVES_PATH = "hard_negatives/hard_negatives_multilingual-e5-large-instruct.json"
os.makedirs("hard_negatives", exist_ok=True)



if __name__ == "__main__":
    with open(SUBMISSION_PATH, "r") as file:
        data = json.load(file)

    with open(TRAIN_PAIRS_PATH, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file) 
        train_pairs = list(reader)


    id_to_positive = {}

    for x in train_pairs:
        post_id = int(x["post_id"])
        fact_check_id = int(x["fact_check_id"])
        if post_id not in id_to_positive:
            id_to_positive[post_id] = []
        id_to_positive[post_id].append(fact_check_id)

    hard_negatives = {}
    for k, v in data.items():
        dump_json = {}
        for str_id, ls in v.items():
            id = int(str_id)
            assert id not in hard_negatives
            hard_negatives[id] = []
            for x in ls:
                if int(x[0]) in id_to_positive[id]:
                    continue
                hard_negatives[id].append(int(x[0]))
                if len(hard_negatives[id]) == 10:
                    break

    
    with open(HARD_NEGATIVES_PATH, "w") as file:
        json.dump(hard_negatives, file, indent=4)
    