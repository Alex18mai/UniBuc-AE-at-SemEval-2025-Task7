# UniBuc-AE at SemEval-2025 Task 7

This Github repository contains all algorithms and models used by the UniBuc-AE team at SemEval-2025 Task 7.

---

### Abstract


This paper describes our approach to the SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval on both the monolingual and crosslingual tracks. Our training methodology for text embedding models combines contrastive pre-training and hard negatives mining in order to fine-tune models from the E5 family. 

Additionally, we introduce a novel approach for merging the results from multiple models by finding the best majority vote weighted configuration for each subtask using the validation dataset. 

Our team ranked $6^{th}$ in the monolingual track scoring a 0.933 S@10 averaged over all languages and achieved a 0.79 S@10 on the crosslingual task, ranking $8^{th}$ in this track.

---

### System Overview

Models used:
- **multilingual-e5-large-instruct** with original text
- **e5-large-v2** with English translations

Training steps:
- Contrastive Pre-training
- Hard Negatives Mining

The final submission was generated combining multiple models using a Weighted Majority Vote where we find the best weighted configuration on each subtask.

---

### Citation

```
@inproceedings{semeval2025task7,
	title={UniBuc-AE at SemEval-2025 Task 7 : Training Text Embedding Models for Multilingual and Crosslingual Fact-Checked Claim Retrieval},
	author={Enache, Alexandru},
	booktitle = {Proceedings of the 19th International Workshop on Semantic Evaluation},
    series = {SemEval 2025},
    year = {2025},
    address = {Vienna, Austria},
    month = {July},
    pages = {},
    doi= {}
}
```