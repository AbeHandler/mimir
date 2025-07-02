import datasets
ds = datasets.load_dataset("abehandlerorg/minhashblocksample)
ds = datasets.load_dataset("abehandlerorg/minhashblocksample")
ds = datasets.load_dataset(self.name)["train"].shuffle(seed=42)
ds = datasets.load_dataset("abehandlerorg/minhashblocksample")["train"].shuffle(seed=42)

urls = set(o.strip("\n") for o in open("targets.txt"))
ds.filter(lambda ex: ex.get("url") in urls)
%history -f  tmp.py
