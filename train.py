import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from copy import deepcopy

from modules.parse_data import parse_data, _read_csv, _read_json, _save_json
from modules.posts_dataset import PostsDataset
from modules.factchecks_dataset import FactchecksDataset
from modules.pairs_dataset import PairsDataset
from modules.model import Embedder




PATH_TRAIN_DATA = "data/data_train"
PATH_TEST_DATA = "data/data_test"
PATH_PARSED_DATA = "data/data_parsed"

MODEL_PRETRAINED = 'intfloat/multilingual-e5-large-instruct'
# MODEL_PRETRAINED = 'intfloat/e5-large-v2'

MODEL_CHECKPOINT = None

MODEL_NAME = MODEL_PRETRAINED.split('/')[-1]

TEXTS = ["text_multi"] # text_multi, text_eng

PATH_CHECKPOINTS = "checkpoints"
os.makedirs(PATH_CHECKPOINTS, exist_ok=True)

PATH_SUBMISSIONS = "submissions"
os.makedirs(PATH_SUBMISSIONS, exist_ok=True)


# train hyperparams
EPOCHS = 5
LR = 1e-5
WD = 1e-5
BS = 8
TEMP = 0.01

PATH_HARD_NEGATIVES = None
HARD_NAME = ""
if PATH_HARD_NEGATIVES is not None:
    HARD_NAME = "_hard"

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logging
logger = logging.getLogger(__name__)
time_formatter = logging.Formatter('%(asctime)s - %(message)s')
logger_file = logging.FileHandler(os.path.join(PATH_CHECKPOINTS, f'train_{MODEL_NAME}.log'))
logger_file.setFormatter(time_formatter)
logger.addHandler(logger_file)
logger_console = logging.StreamHandler()
logger_console.setFormatter(time_formatter)
logger.addHandler(logger_console)

# prompts
def prompt_post(post):
    if "instruct" in MODEL_NAME:
        return f"Instruct: Given a social media post, retrieve relevant fact-checked claims for the post\nQuery: {post}"
    return f"query: {post}"

def prompt_factcheck(factcheck):
    if "instruct" in MODEL_NAME:
        return factcheck
    return f"passage: {factcheck}"


def save_model(model, epoch, val_acc):
    name = f'checkpoint_{MODEL_NAME}{HARD_NAME}_epoch={epoch+1}_acc={val_acc}.pth'
    torch.save(model.state_dict(), os.path.join(PATH_CHECKPOINTS, name))


def success10(predictions, pairs):
    success = {k : 0 for k in predictions.keys()}
    
    for pair in pairs:
        post_id = int(pair["post_id"])
        fact_check_id = int(pair["fact_check_id"])
        if post_id in predictions and fact_check_id in predictions[post_id]:
            success[post_id] = 1

    return round(sum(list(success.values())) / len(success) * 100, 2)



def get_embeddings(model, tokenizer, dataloader):
    ids_all = []
    embeddings_all = []
    for batch in tqdm(dataloader):
        for x in batch["id"]:
            ids_all.append(int(x))
        batch_tokens = tokenizer(batch["text"], max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
        embeddings = model(**batch_tokens).detach().cpu()
        for x in embeddings:
            embeddings_all.append(x)
    embeddings_all = torch.stack(embeddings_all)
    return embeddings_all, ids_all,


def get_top1000(embeddings_posts, embeddings_factchecks, ids_posts, ids_factchecks):
    scores = embeddings_posts @ embeddings_factchecks.T
    top1000 = torch.topk(scores, k=min(1000, scores.shape[1]), dim=1) 
    top1000_indices = top1000.indices
    top1000_values = top1000.values
    predictions = {}
    for i, post in enumerate(ids_posts):
        top_fact_checks = [(ids_factchecks[idx], value) for idx, value in zip(top1000_indices[i].tolist(), top1000_values[i].tolist())]
        predictions[post] = top_fact_checks
    return predictions


def validation(model, tokenizer, epoch):
    logger.warning(f"Running validation for epoch={epoch+1}")

    model.eval()
    with torch.no_grad():
        val_tasks = _read_json(os.path.join(PATH_PARSED_DATA, "val.json"))
        dev_tasks = _read_json(os.path.join(PATH_PARSED_DATA, "dev.json"))
        train_tasks = _read_json(os.path.join(PATH_PARSED_DATA, "train.json"))
        val_pairs = _read_csv(os.path.join(PATH_PARSED_DATA, "pairs_val.csv"))
        
        total_monolingual = 0
        cnt_monolingual = 0

        submission_val = {}
        submission_dev = {}
        submission_train = {}

        for (task, values), (dev_task, dev_values), (train_task, train_values) in zip(val_tasks.items(), dev_tasks.items(), train_tasks.items()):
            assert task == dev_task
            assert task == train_task
            assert len(values['fact_checks']) == len(dev_values['fact_checks'])
            assert len(values['fact_checks']) == len(train_values['fact_checks'])

            dataset_posts_val = PostsDataset(os.path.join(PATH_PARSED_DATA, "posts.csv"), set(values['posts']), TEXTS, prompt_post)
            dataset_posts_dev = PostsDataset(os.path.join(PATH_PARSED_DATA, "posts.csv"), set(dev_values['posts']), TEXTS, prompt_post)
            dataset_posts_train = PostsDataset(os.path.join(PATH_PARSED_DATA, "posts.csv"), set(train_values['posts']), TEXTS, prompt_post)
            dataset_factchecks_val = FactchecksDataset(os.path.join(PATH_PARSED_DATA, "fact_checks.csv"), set(values['fact_checks']), TEXTS, prompt_factcheck)

            dataloader_posts_val = DataLoader(dataset = dataset_posts_val, batch_size = BS*4, shuffle = False)
            dataloader_posts_dev = DataLoader(dataset = dataset_posts_dev, batch_size = BS*4, shuffle = False)
            dataloader_posts_train = DataLoader(dataset = dataset_posts_train, batch_size = BS*4, shuffle = False)
            dataloader_factchecks_val = DataLoader(dataset = dataset_factchecks_val, batch_size = BS*4, shuffle = False)

            # factchecks
            embeddings_factchecks, ids_factchecks = get_embeddings(model, tokenizer, dataloader_factchecks_val)

            # posts
            embeddings_posts, ids_posts = get_embeddings(model, tokenizer, dataloader_posts_val)

            # posts dev
            embeddings_posts_dev, ids_posts_dev = get_embeddings(model, tokenizer, dataloader_posts_dev)

            # posts train
            embeddings_posts_train, ids_posts_train = get_embeddings(model, tokenizer, dataloader_posts_train)

            # predict
            scores = embeddings_posts @ embeddings_factchecks.T
            top10_indices = torch.topk(scores, k=10, dim=1).indices 
            predictions = {}
            for i, post in enumerate(ids_posts):
                top_fact_checks = [ids_factchecks[idx] for idx in top10_indices[i].tolist()]
                predictions[post] = top_fact_checks

            acc = success10(predictions, val_pairs)
            logger.warning(f"{task} : {acc}")

            if "_" in task:
                total_monolingual += acc
                cnt_monolingual += 1

            # submission
            submission_val[task] = get_top1000(embeddings_posts, embeddings_factchecks, ids_posts, ids_factchecks)
            submission_dev[task] = get_top1000(embeddings_posts_dev, embeddings_factchecks, ids_posts_dev, ids_factchecks)
            submission_train[task] = get_top1000(embeddings_posts_train, embeddings_factchecks, ids_posts_train, ids_factchecks)

            if epoch+1 != EPOCHS:
                break

        avg_monolingual = round(total_monolingual / cnt_monolingual, 2)
        logger.warning(f"avg monolingual : {avg_monolingual}")

        # save submissions
        name = MODEL_CHECKPOINT.split('/')[-1][:-4]
        _save_json(submission_val, os.path.join(PATH_SUBMISSIONS, f"submission_val_{name}_epoch={epoch+1}_acc={avg_monolingual}.json"))
        _save_json(submission_dev, os.path.join(PATH_SUBMISSIONS, f"submission_dev_{name}_epoch={epoch+1}_acc={avg_monolingual}.json"))
        _save_json(submission_train, os.path.join(PATH_SUBMISSIONS, f"submission_train_{name}_epoch={epoch+1}_acc={avg_monolingual}.json"))


    logger.warning("----------------------------------")
    return avg_monolingual


def contrastive_loss(pred_posts, pred_factchecks):
    cos_sim = (pred_posts @ pred_factchecks.T) / TEMP 
    criterion = nn.CrossEntropyLoss()
    labels = torch.arange(cos_sim.shape[0]).to(device) 
    return criterion(cos_sim, labels) 


def train(model, tokenizer):
    train_dataset = PairsDataset(pairs_csv = os.path.join(PATH_PARSED_DATA, "pairs_train.csv"), 
                                 posts_csv = os.path.join(PATH_PARSED_DATA, "posts.csv"), 
                                 factchecks_csv = os.path.join(PATH_PARSED_DATA, "fact_checks.csv"), 
                                 texts = TEXTS, 
                                 prompt_post = prompt_post, 
                                 prompt_factcheck = prompt_factcheck,
                                 hard_negatives_csv = PATH_HARD_NEGATIVES                          
                                 )
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = BS, shuffle = True)

    optimizer = torch.optim.AdamW(model.parameters(), lr = LR, weight_decay = WD)
    warmup_steps = 100
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler()

    # training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()

            posts = tokenizer(batch["post"], max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            factcheck_text = batch["factcheck"]
            if "hard_negatives" in batch:
                for x in batch["hard_negatives"]:
                    factcheck_text += x
            factcheck = tokenizer(factcheck_text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)

            with autocast():
                pred_posts = model(**posts)
                pred_factchecks = model(**factcheck)
                loss = contrastive_loss(pred_posts, pred_factchecks)

            if torch.isnan(loss):
                print("Loss is NaN. Exiting...")
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        logger.warning(f"Train loss epoch={epoch+1} : {train_loss / (len(train_dataloader))}")
        
        val_acc = validation(model, tokenizer, epoch)
        save_model(model, epoch, val_acc)


def test(model, tokenizer):
    logger.warning(f"Running test")

    model.eval()
    with torch.no_grad():
        test_tasks = _read_json(os.path.join(PATH_PARSED_DATA, "test.json"))
        submission_test = {}

        for task, values in test_tasks.items():

            dataset_posts_test = PostsDataset(os.path.join(PATH_PARSED_DATA, "posts.csv"), set(values['posts']), TEXTS, prompt_post)
            dataset_factchecks_test = FactchecksDataset(os.path.join(PATH_PARSED_DATA, "fact_checks.csv"), set(values['fact_checks']), TEXTS, prompt_factcheck)

            dataloader_posts_test = DataLoader(dataset = dataset_posts_test, batch_size = BS*4, shuffle = False)
            dataloader_factchecks_test = DataLoader(dataset = dataset_factchecks_test, batch_size = BS*4, shuffle = False)

            # factchecks
            embeddings_factchecks, ids_factchecks = get_embeddings(model, tokenizer, dataloader_factchecks_test)

            # posts
            embeddings_posts, ids_posts = get_embeddings(model, tokenizer, dataloader_posts_test)

            # submission
            submission_test[task] = get_top1000(embeddings_posts, embeddings_factchecks, ids_posts, ids_factchecks)

        # save submissions
        name = MODEL_CHECKPOINT.split('/')[-1][:-4]
        _save_json(submission_test, os.path.join(PATH_SUBMISSIONS, f"submission_test_{name}_epoch={EPOCHS}.json"))


    logger.warning("----------------------------------")



if __name__ == "__main__":

    parse_data(path_train_data=PATH_TRAIN_DATA, path_test_data=PATH_TEST_DATA, path_parsed_data=PATH_PARSED_DATA)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PRETRAINED)
    model = Embedder(MODEL_PRETRAINED)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT))
    model.to(device)

    validation(model, tokenizer, -1)
    train(model, tokenizer)
    test(model, tokenizer)

