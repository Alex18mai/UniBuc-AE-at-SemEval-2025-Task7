from torch.utils.data import Dataset
from modules.utils_dataset import read_posts, read_factchecks
from modules.parse_data import _read_csv, _read_json


class PairsDataset(Dataset):
    def __init__(self, pairs_csv, posts_csv, factchecks_csv, texts, prompt_post, prompt_factcheck, hard_negatives_csv):
        super().__init__()

        all_posts = read_posts(posts_csv)
        post_from_id = {}
        for x in all_posts:
            text_str = " ".join([x[text] for text in texts])
            text_str = prompt_post(text_str)
            post_from_id[x["id"]] = text_str

        all_factchecks = read_factchecks(factchecks_csv)
        factchecks_from_id = {}
        for x in all_factchecks:
            text_str = " ".join([x[text] for text in texts])
            text_str = prompt_factcheck(text_str)
            factchecks_from_id[x["id"]] = text_str

        pairs_ids = _read_csv(pairs_csv)

        if hard_negatives_csv is not None:
            hard_negatives = _read_json(hard_negatives_csv)
            hard_negatives = {int(k) : v for k,v in hard_negatives.items()}
        
        self.pairs = []
        for pair in pairs_ids:
            post_id = int(pair["post_id"])
            fact_check_id = int(pair["fact_check_id"])
            sample = {
                "post" : post_from_id[post_id],
                "factcheck" : factchecks_from_id[fact_check_id]
            }
            if hard_negatives_csv is not None:
                assert len(hard_negatives[post_id]) >= 3
                assert len(hard_negatives[post_id]) <= 10
                top_negative = hard_negatives[post_id][-3:]
                sample["hard_negatives"] = [factchecks_from_id[x] for x in top_negative]
            self.pairs.append(sample)

    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]