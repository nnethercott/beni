# from sklearn import CountVectorizer
from transformers import AutoTokenizer
from typing import Iterable, List, Any
import uuid
from datasketch import MinHashLSH, MinHash
from tqdm import tqdm


class MinHashLSHDeduplicator:
    """
    from .minhash import MinHashLSH
    from .d import load_recap

    datasets = [[item['response'] for item in load_recap().data] for _ in range(10)]
    lsh = MinHashLSH(*datasets, num_permutations=10)
    ids = lsh.deduplicate(jaccard_sim = 0.5)

    # deduplicate manually ...
    """

    def __init__(self, tokenizer: AutoTokenizer, *datasets: Iterable[Any]):
        self.datasets = list(datasets)
        self.tokenizer = tokenizer

    def build_minhash(self, sample, num_perm: int = 10):
        minhash = MinHash(num_perm=num_perm)
        tokens = set(self.tokenizer.tokenize(sample))
        for token in tokens:
            minhash.update(token.encode("utf8"))

        return minhash

    def deduplicate(
        self, jaccard_sim: float = 0.8, num_perm: int = 10
    ) -> List[List[int]]:
        """
        returns bad ids to be dropped from respective datasets
        """
        ids = {i: str(uuid.uuid1()) for i in range(len(self.datasets))}

        # maybe add redis store
        lsh = MinHashLSH(threshold=jaccard_sim, num_perm=num_perm)

        duplicate_ids = []

        # iterate over new datasets and merge lsh indexes
        for i, dataset in enumerate(self.datasets):
            uuid_ = ids[i]

            local_duplicate_ids = []
            for j, item in tqdm(enumerate(dataset)):
                minhash = self.build_minhash(item, num_perm=num_perm)

                # edge case
                if lsh.is_empty():
                    lsh.insert(f"{uuid_}_{j}", minhash)
                    continue

                res = lsh.query(minhash)
                if len(res) > 0:
                    # remove item from dataset
                    local_duplicate_ids.append(j)

                    # print similar sample
                    _uuid, index = res[0].split("_")
                    dataset_id = {v: k for k, v in ids.items()}[_uuid]
                    print(f"sample: {item}\ndataset: {i}, sample: {j}")
                    print(
                        f"ref: {self.datasets[dataset_id][int(index)]}\ndataset: {dataset_id}, sample: {int(index)}"
                    )
                    print("\n")

                else:
                    lsh.insert(f"{uuid_}_{j}", minhash)

            duplicate_ids.append(local_duplicate_ids)

        return duplicate_ids


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import time
    from .d import load_recap, load_allava_laion

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M-Instruct")

    recap = load_recap(tok, n=50000)
    allava = load_allava_laion(tok, n=50000)

    datasets = [[item["response"] for item in d.data] for d in [recap, allava]]

    minhash = MinHashLSHDeduplicator(tok, *datasets)

    before = time.time()
    ids = minhash.deduplicate(jaccard_sim=0.85, num_perm=128)
    print(time.time() - before)
