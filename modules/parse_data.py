import os
import csv
import json
from sklearn.model_selection import train_test_split


def _read_csv(path_read):
    with open(path_read, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file) 
        return list(reader)

def _save_csv(data, path_save):
    fieldnames = data[0].keys()
    with open(path_save, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def _read_json(path_read):
    with open(path_read, "r") as file:
        return json.load(file)

def _save_json(data, path_save):
    with open(path_save, "w") as file:
        json.dump(data, file)


def _save_all_posts_and_factchecks(path_train_data, path_test_data, path_parsed_data):
    posts_train = _read_csv(os.path.join(path_train_data, "posts.csv"))
    posts_test = _read_csv(os.path.join(path_test_data, "posts.csv"))

    posts_all = []
    used_ids = set()
    for x in posts_train:
        if x["post_id"] not in used_ids:
            posts_all.append(x)
            used_ids.add(x["post_id"])
    for x in posts_test:
        if x["post_id"] not in used_ids:
            posts_all.append(x)
            used_ids.add(x["post_id"])
    _save_csv(posts_all, os.path.join(path_parsed_data, "posts.csv"))

    factchecks_train = _read_csv(os.path.join(path_train_data, "fact_checks.csv"))
    factchecks_test = _read_csv(os.path.join(path_test_data, "fact_checks.csv"))

    factchecks_all = []
    used_ids = set()
    for x in factchecks_train:
        if x["fact_check_id"] not in used_ids:
            factchecks_all.append(x)
            used_ids.add(x["fact_check_id"])
    for x in factchecks_test:
        if x["fact_check_id"] not in used_ids:
            factchecks_all.append(x)
            used_ids.add(x["fact_check_id"])


    _save_csv(factchecks_all, os.path.join(path_parsed_data, "fact_checks.csv"))


def _split_tasks(path_train_data, path_test_data, path_parsed_data):
    train_val_dev_tasks = _read_json(os.path.join(path_train_data, "tasks.json"))
    test_tasks = _read_json(os.path.join(path_test_data, "tasks.json"))

    save_train_tasks = {}
    save_val_tasks = {}
    all_val_posts = set()
    save_dev_tasks = {}
    save_test_tasks = {}
    
    # train, val, dev
    for lang in train_val_dev_tasks['monolingual'].keys():
        data = train_val_dev_tasks['monolingual'][lang]
        fact_checks = data["fact_checks"]
        train_posts, val_posts = train_test_split(data['posts_train'], test_size=0.2, random_state=42)
        for x in val_posts:
            all_val_posts.add(x)
        dev_posts = data['posts_dev']
        save_train_tasks[f"monolingual_{lang}"] = {"fact_checks" : fact_checks, "posts" : train_posts}
        save_val_tasks[f"monolingual_{lang}"] = {"fact_checks" : fact_checks, "posts" : val_posts}
        save_dev_tasks[f"monolingual_{lang}"] = {"fact_checks" : fact_checks, "posts" : dev_posts}

    data = train_val_dev_tasks['crosslingual']
    fact_checks = data["fact_checks"]
    train_posts, val_posts = train_test_split(data['posts_train'], test_size=0.2, random_state=42)
    for x in val_posts:
        all_val_posts.add(x)
    dev_posts = data['posts_dev']
    save_train_tasks['crosslingual'] = {"fact_checks" : fact_checks, "posts" : train_posts}
    save_val_tasks['crosslingual'] = {"fact_checks" : fact_checks, "posts" : val_posts}
    save_dev_tasks['crosslingual'] = {"fact_checks" : fact_checks, "posts" : dev_posts}
    
    # test
    for lang in test_tasks['monolingual'].keys():
        data = test_tasks['monolingual'][lang]
        save_test_tasks[f"monolingual_{lang}"] = {"fact_checks" : data["fact_checks"], "posts" : data["posts_test"]}

    data = test_tasks['crosslingual']
    save_test_tasks['crosslingual'] = {"fact_checks" : data["fact_checks"], "posts" : data["posts_test"]}

    # save jsons
    _save_json(save_train_tasks, os.path.join(path_parsed_data, "train.json"))
    _save_json(save_val_tasks, os.path.join(path_parsed_data, "val.json"))
    _save_json(save_dev_tasks, os.path.join(path_parsed_data, "dev.json"))
    _save_json(save_test_tasks, os.path.join(path_parsed_data, "test.json"))

    # pairs
    pairs = _read_csv(os.path.join(path_train_data, "pairs.csv"))

    pairs_train = []
    pairs_val = []

    for pair in pairs:
        if int(pair["post_id"]) in all_val_posts:
            pairs_val.append(pair)
        else:
            pairs_train.append(pair)

    # save csv
    _save_csv(pairs_train, os.path.join(path_parsed_data, "pairs_train.csv"))
    _save_csv(pairs_val, os.path.join(path_parsed_data, "pairs_val.csv"))



def parse_data(path_train_data, path_test_data, path_parsed_data):
    if os.path.exists(path_parsed_data):
        return
    os.makedirs(path_parsed_data, exist_ok=True)

    _save_all_posts_and_factchecks(path_train_data, path_test_data, path_parsed_data)

    _split_tasks(path_train_data, path_test_data, path_parsed_data)