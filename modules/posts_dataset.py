from torch.utils.data import Dataset
from modules.utils_dataset import read_posts


class PostsDataset(Dataset):
    def __init__(self, path_csv, ids, texts, prompt_post):
        super().__init__()

        all_posts = read_posts(path_csv)

        self.posts = []
        for x in all_posts:
            if x["id"] not in ids:
                continue
            text_str = " ".join([x[text] for text in texts])
            text_str = prompt_post(text_str)
            post = {
                "id" : x["id"],
                "text" : text_str,
            }
            self.posts.append(post)
    
    def __len__(self):
        return len(self.posts)
    
    def __getitem__(self, idx):
        return self.posts[idx]