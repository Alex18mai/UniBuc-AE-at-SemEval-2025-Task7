from torch.utils.data import Dataset
from modules.utils_dataset import read_factchecks


class FactchecksDataset(Dataset):
    def __init__(self, path_csv, ids, texts, prompt_factcheck):
        super().__init__()

        all_factchecks = read_factchecks(path_csv)

        self.factchecks = []
        for x in all_factchecks:
            if x["id"] not in ids:
                continue
            text_str = " ".join([x[text] for text in texts])
            text_str = prompt_factcheck(text_str)
            factcheck = {
                "id" : x["id"],
                "text" : text_str,
            }
            self.factchecks.append(factcheck)
    
    def __len__(self):
        return len(self.factchecks)
    
    def __getitem__(self, idx):
        return self.factchecks[idx]