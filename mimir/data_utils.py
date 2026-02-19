"""
    Datasets and data-processing utilities
"""
import datasets
import numpy as np
import os
import mimir.custom_datasets as custom_datasets
from mimir.config import ExperimentConfig
from nltk.tokenize import WhitespaceTokenizer
from urllib.parse import urlparse
from collections import defaultdict
from huggingface_hub import list_datasets as hf_list_datasets


def pythia_kluge(name_key_mapping: dict) -> dict:
    """Add all abehandlerorg/pythia-* HF datasets to name_key_mapping with value "text".

    Example:
        >>> pythia_kluge(name_key_mapping)
    """
    for ds in hf_list_datasets(author="abehandlerorg", search="pythia"):
        if ds.id.startswith("abehandlerorg/pythia-"):
            name_key_mapping[ds.id] = "text"
    return name_key_mapping


def normalize_domain(url):
    if not url:
        print("warning no url")
        return ""
    url = url.strip('"')
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _load_ccnews_jan2022() -> "datasets.Dataset":
    """Load abehandlerorg/ccnews-jan2022 filtered to URLs in the neighbor log.

    Mirrors the logic in scripts/R2/inthewild/hf_ds_maker.py but returns a
    HuggingFace Dataset directly instead of writing a parquet file.
    """
    import json
    from pathlib import Path

    jsonl_path = os.path.expanduser(
        "~/dolma/logs/scripts/R2/extract/inthewild/analyze_neighbor_log.jsonl.gz"
    )
    jsonl_url_field = "query_url"
    hf_url_field = "url"

    log_path = Path(jsonl_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Neighbor log not found: {log_path}")

    query_urls = set()
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            url = record.get(jsonl_url_field)
            if url:
                query_urls.add(url)
    print(f"[ccnews-jan2022] Loaded {len(query_urls)} query URLs from {log_path}")

    print("[ccnews-jan2022] Loading HuggingFace dataset abehandlerorg/ccnews-jan2022 ...")
    ds = datasets.load_dataset("abehandlerorg/ccnews-jan2022", split="train")
    ds = ds.filter(lambda ex: ex.get(hf_url_field) in query_urls)
    print(f"[ccnews-jan2022] Filtered to {ds.num_rows} matching rows")
    ds = ds.map(lambda ex: {"id": ex["url"]})
    return ds


class Data:
    """
    Data class to load and cache datasets.
    """
    def __init__(self, name,
                 config: ExperimentConfig,
                 presampled: str = None,
                 name_key_mapping: dict = {"abehandlerorg/bloxbypublisher": "text", 
                                           "abehandlerorg/twfe": "text",
                                           "abehandlerorg/suffixesnoblocksbin": "text",
                                           "abehandlerorg/twfecontrols": "text",
                                           "abehandlerorg/minhashblocksample": "text",
                                           "abehandlerorg/localblockeddocs": "text",
                                           "abehandlerorg/blockeddocs": "text",
                                           "abehandlerorg/copyrighttrapszeros": "text",
                                           "abehandlerorg/confounddataset": "text",
                                           "abehandlerorg/confounddatasetxpress": "text",
                                           "abehandlerorg/nobloxbypublisher": "text",
                                           "abehandlerorg/excluded-docs": "text",
                                           "abehandlerorg/excluded-docs-mini": "text",
                                           "abehandlerorg/bothbins": "text",
                                           "abehandlerorg/hawaiinewsnow_scm": "text",
                                           "abehandlerorg/nelsoncountygazette_scm": "text",
                                           "abehandlerorg/fox40_scm": "text", 
                                           "abehandlerorg/richmond_scm": "text",
                                           "abehandlerorg/phl17_scm": "text",
                                           "abehandlerorg/atlanticcityweekly_scm": "text",
                                           "abehandlerorg/maysville-online_scm": "text",
                                           "abehandlerorg/nj1015_scm": "text",
                                           "abehandlerorg/kobi5_scm": "text",
                                           "abehandlerorg/theeagle_scm": "text",
                                           "abehandlerorg/fremonttribune_scm": "text",
                                           "abehandlerorg/nelsoncountygazette_scm": "text",
                                           "abehandlerorg/matching_neighbors": "text",
                                           "abehandlerorg/olmobypublisherdev": "text",
                                           'abehandlerorg/sutva_click2houston_com_2022-03-01_pair2_control_run4_filtered': "text",
                                           "abehandlerorg/copywritetraps": "text",
                                           "abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_neighbors_top100": "text",
                                           "abehandlerorg/sutva_click2houston_com_2022-05-01_pair1_treated_run1": "text",
                                           "abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_treated_run3": "text",
                                           "abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_filtered": "text",
                                           "abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4": "text",
                                           "abehandlerorg/sutva_click2houston_com_2022-05-01_pair1_control_run2": "text",
                                           "abehandlerorg/ccnews-jan2022": "text"}):

        with open('configs/single_publishers.txt', 'r') as inf:
            for _ in inf:
                _ = _.strip('\n')
                k = "abehandlerorg/" + _ + "_scm"
                name_key_mapping[k] = "text"

        name_key_mapping["abehandlerorg/cptgptoss_bothbins_20240730_20240730"] = "text"
        name_key_mapping["abehandlerorg/cptgptoss_excluded_20240730_20240730"] = "text"

        name_key_mapping["abehandlerorg/cptllama_excluded_20240130_20240130"] = "text"
        name_key_mapping["abehandlerorg/cptllama_bothbins_20240130_20240130"] = "text"

        # needed for pythia experiments on blackwell machine
        name_key_mapping = pythia_kluge(name_key_mapping)

        self.name_key_mapping = name_key_mapping
        self.config = config
        self.name = name
        self.presampled = presampled
        self.key = (
            config.dataset_key
            if config.dataset_key
            else self.name_key_mapping.get(name, None)
        )
        if self.key is None:
            raise ValueError(
                f"Key for dataset {name} not provided or found inname_key_mapping"
            )
        self.cache_dir = self.config.env_config.cache_dir

    def load_neighbors(
        self,
        train: bool,
        num_neighbors: int,
        model: str = "bert",
        in_place_swap: bool = False,
    ):
        """
        Load neighbors from cache (local or from HF)
        """
        data_split = "train" if train else "test"
        data_split += "_neighbors"
        filename = self._get_name_to_save() + "_neighbors_{}_{}".format(
            num_neighbors, model
        )
        if in_place_swap:
            filename += "_in_place_swap"
        data = custom_datasets.load_cached(
            self.cache_dir,
            data_split,
            filename,
            min_length=self.config.min_words,
            max_length=self.config.max_words,
            n_samples=self.config.n_samples,
            max_tokens=self.config.max_tokens,
            load_from_hf=self.config.load_from_hf
        )
        return data

    def dump_neighbors(
        self,
        data,
        train: bool,
        num_neighbors: int,
        model: str = "bert",
        in_place_swap: bool = False,
    ):
        """
        Dump neighbors to cache local cache.
        """
        data_split = "train" if train else "test"
        data_split += "_neighbors"
        filename = self._get_name_to_save() + "_neighbors_{}_{}".format(
            num_neighbors, model
        )
        if in_place_swap:
            filename += "_in_place_swap"
        custom_datasets.dump_to_cache(
            data,
            self.cache_dir,
            data_split,
            filename,
            min_length=self.config.min_words,
            max_length=self.config.max_words,
            n_samples=self.config.n_samples,
            max_tokens=self.config.max_tokens,
        )

    def load(self, train: bool, mask_tokenizer=None, specific_source: str = None):
        data_split = "train" if train else "test"
        n_samples = self.config.n_samples

        # Load from numpy file storing pretokenized sample in a 2d array of shape (num_samples, num_tokens_per_sample)
        if self.config.pretokenized:

            assert "we will not use this" == "setting"  # AH 3/29 we wont use this branch

            '''
            assert self.presampled
            # TODO: Pretokenized full documents (split into substrs) is not currently supported
            assert not self.config.full_doc
            data = np.load(self.presampled)
            return data
            '''
        elif (self.config.load_from_cache or self.config.load_from_hf):

            #  👀 simplify here for our setting

            # special filtering here
            if self.name == "abehandlerorg/ccnews-jan2022":
                ds = _load_ccnews_jan2022()
                assert "SHARD_ID" in os.environ
                return select_shard(ds, shard_size=20)

            ds = datasets.load_dataset(self.name)["train"].shuffle(seed=42)

            # name_key_mapping["abehandlerorg/cptgptoss_bothbins_20240730_20240730"] = "text"
            # name_key_mapping["abehandlerorg/cptgptoss_excluded_20240730_20240730"] = "text"
            if self.name == "abehandlerorg/cptgptoss_bothbins_20240730_20240730":
                ds = ds.map(lambda x: {"id": x["url"]})
                SHARD_SIZE = 10
                shard_id = int(os.environ["SHARD_ID"])
                start = shard_id * SHARD_SIZE
                end = (shard_id + 1) * SHARD_SIZE
                end = min(end, len(ds)) # if end if past len ds then pick end
                ds = ds.map(lambda x: {"id": x["url"]})
                return ds.select(range(start, end))

            if self.name == "abehandlerorg/cptgptoss_excluded_20240730_20240730":
                ds = ds.map(lambda x: {"id": x["url"]})
                shard_id = int(os.environ["SHARD_ID"])
                SHARD_SIZE = 10
                start = shard_id * SHARD_SIZE
                end = (shard_id + 1) * SHARD_SIZE
                end = min(end, len(ds)) # if end if past len ds then pick end
                ds = ds.map(lambda x: {"id": x["url"]})
                return ds.select(range(start, end))

            if self.name == "abehandlerorg/cptllama_excluded_20240130_20240130":
                ds = ds.map(lambda x: {"id": x["url"]})

                if "SHARD_ID" in os.environ:
                    return select_shard(ds, shard_size=100)
                else:
                    return ds

            if self.name == "abehandlerorg/cptllama_bothbins_20240130_20240130":
                # Clip text to 25K characters to prevent OOM errors during inference.
                # Long texts (>25K chars) cause memory spikes when computing logits,
                # even with batch_size=1. Sample #977 had 152K chars and crashed at 22.34GB.
                # 99% of articles are <25K chars, so this only affects rare edge cases.
                ds = ds.map(lambda x: {
                    "id": x["url"],
                    "text": x["text"][:25000] if len(x["text"]) > 25000 else x["text"]
                })

                if "SHARD_ID" in os.environ:
                    return select_shard(ds, shard_size=100)
                else:
                    return ds

            if self.name == "abehandlerorg/sutva_click2houston_com_2022-05-01_pair2_control_run4_neighbors_top100":
                return ds.select(range(n_samples))

            if "sutva" in self.name and "pair" in self.name:
                ds = ds.map(lambda x: {"id": x["url"]})
                return ds.select(range(n_samples))

            if self.name.startswith("abehandlerorg/pythia-"):
                ds = ds.filter(lambda example: len(example["text"]) > 100)
                return ds.select(range(n_samples))

            if self.name == "abehandlerorg/matching_neighbors":
                ds = ds.filter(lambda example: len(example["text"]) > 100)
                shard_id = int(os.environ["SHARD_ID"])
                SHARD_SIZE = 5000
                start = shard_id * SHARD_SIZE
                end = (shard_id + 1) * SHARD_SIZE
                end = min(end, len(ds)) # if end if past len ds then pick end
                ds = ds.map(lambda x: {"id": x["url"]})
                return ds.select(range(start, end))

            if self.name == "abehandlerorg/bothbins":
                ds = ds.filter(lambda example: len(example["text"]) > 100)
                shard_id = int(os.environ["SHARD_ID"])
                SHARD_SIZE = 5000
                start = shard_id * SHARD_SIZE
                end = (shard_id + 1) * SHARD_SIZE
                end = min(end, len(ds)) # if end if past len ds then pick end
                return ds.select(range(start, end))

            if self.name == "abehandlerorg/confounddatasetxpress":
                return ds

            if self.name == "abehandlerorg/nobloxbypublisher":
                return ds.select(range(n_samples))

            if self.name == "abehandlerorg/excluded-docs-mini":
                return ds.select(range(n_samples))

            if self.name == "abehandlerorg/confounddataset":
                if "SHARD_ID" not in os.environ:
                    raise ValueError("SHARD_ID not set in environment")
                shard_id = int(os.environ["SHARD_ID"])
                SHARD_SIZE = 5000
                start = shard_id * SHARD_SIZE
                end = (shard_id + 1) * SHARD_SIZE
                # This does not have end = min(end, len(ds)) but I think its from earlier code. #TODO check later but I think it is
                return ds.select(range(start, end))

            if self.name == "abehandlerorg/excluded-docs":
                if "SHARD_ID" not in os.environ:
                    raise ValueError("SHARD_ID not set in environment")
                shard_id = int(os.environ["SHARD_ID"])
                SHARD_SIZE = 5000
                start = shard_id * SHARD_SIZE
                end = (shard_id + 1) * SHARD_SIZE
                end = min(end, len(ds)) # if end if past len ds then pick end
                return ds.select(range(start, end))

            # e.g. abehandlerorg/hawaiinewsnow_scm
            if self.name.endswith("_scm"):
                # strip the " at start and end
                ds = ds.map(lambda example: {"id": example["url"]})
                return ds.filter(lambda example: len(example["text"]) > 100)

            if self.name == "abehandlerorg/twfe":
                # strip the " at start and end
                ds = ds.map(lambda example: {"id": example["url"]})
                return ds.select(range(n_samples))

            if self.name == "abehandlerorg/twfecontrols":
                ds = datasets.load_dataset(self.name, download_mode="force_redownload")["train"].shuffle(seed=42)
                ds = ds.map(lambda example: {"id": example["url"]})
                # some of these short texts cause mimir errors
                return ds.filter(lambda example: len(example["text"]) > 100)

            if self.name == "abehandlerorg/suffixesnoblocksbin":
                datasets.load_dataset(self.name)["train"].shuffle(seed=42)

                ds = ds.map(lambda example: {"id": example["sequence"]})
                # because we are filtering to shard _0_ we need to ensure that noblocks > blocks
                # this is often true but is not when (1) the sequence only appears outside shard 0
                # (2) the sequence may appear more in shard_0 in blocks bin which is rare
                # see debugging emails Jun 30, 2025 w/ team
                ds = ds.map(lambda x: {"text": x["sequence"].strip('"')})
                ds = ds.filter(lambda example: example["noblocksbin"] > example["blocksbin"])
                ds = ds.filter(lambda example: example["blocksbin"] == 0)

                # this line ony ran for ne which has to run on a 50% samlpe.
                return ds.select(range(n_samples)) #  iadded this line for the ne method on Jul 13 after everything else ran

                return ds

            if self.name == "abehandlerorg/copywritetraps":
                # strip the " at start and end
                ds = ds.map(lambda x: {"text": x["text"].strip('"')})
                return ds


            if self.name == "abehandlerorg/copyrighttrapszeros":
                ds = ds.map(lambda x: {"text": x["text"].strip('"')})
                return ds

            if self.name == "abehandlerorg/minhashblocksample":
                urls = set(o.strip("\n") for o in open("targets.txt"))
                ds = ds.filter(lambda ex: ex.get("url") in urls)
                # the ne method does not use the full targets
                return ds.select(range(n_samples))

            #some of these examples do not have urls. why?
            orig_len = ds.num_rows
            ds = ds.filter(lambda ex: ex.get("url") is not None, batched=False)
            dropped = orig_len - ds.num_rows
            if dropped > 0:
                print(f"Warning: {dropped} examples had no URL and were dropped")

            return ds.select(range(n_samples))
            
            #data = custom_datasets.load_cached(
            #    self.cache_dir,
            #    data_split,
            #    filename,
            #    min_length=self.config.min_words,
            #    max_length=self.config.max_words,
            #    n_samples=self.config.n_samples,
            #    max_tokens=self.config.max_tokens,
            #    load_from_hf=self.config.load_from_hf
            #)
            return data
        else:
            assert "we will not use this" == "setting"  # AH 3/29 we wont use this branch
            '''
            if self.presampled or self.config.full_doc:
                print("using presampled data")
                data = datasets.load_dataset(
                    "json",
                    data_files=self.presampled,
                    split=f"train",
                    cache_dir=self.cache_dir,
                )[self.key]
            elif self.name in custom_datasets.DATASETS:
                data = custom_datasets.load(self.name)
            elif self.name == "the_pile":
                min_load = max(10000, self.config.max_data)
                data = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(
                        self.config.env_config.data_source,
                        "pile/00.jsonl.zst" if train else "pile/test.jsonl.zst",
                    ),
                    cache_dir=self.cache_dir,
                    split=f"train[:{min_load}]",
                )
                specific_source_use = (
                    self.config.specific_source
                    if specific_source is None
                    else specific_source
                )
                data = pile_selection_utility(
                    data, self.key, wanted_source=specific_source_use
                )
            elif "human" in self.name:
                data = datasets.load_dataset(
                    self.name, split=f"train[:100]", cache_dir=self.cache_dir
                )[self.key]
            elif "nthngdy" in self.name:
                data = datasets.load_dataset(
                    self.name, split="test", cache_dir=self.cache_dir
                )[self.key]
            else:
                data = datasets.load_dataset(
                    self.name, split=f"train", cache_dir=self.cache_dir
                )[self.key]
            '''

        if not self.config.full_doc:
            assert "not" == "used"  # AH 3/29/25
            '''
            # get unique examples
            # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
            # then take just the examples that are <= 512 tokens (for the mask model)
            # then generate n_samples samples
            wsp_tokenizer = WhitespaceTokenizer()

            # remove duplicates from the data
            data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

            whitespace_tokenized_spans = [
                (x, list(wsp_tokenizer.span_tokenize(x))) for x in data
            ]

            # Pick samples with at least self.config.min_words words
            whitespace_tokenized_spans = [
                x
                for x in whitespace_tokenized_spans
                if len(x[1]) >= self.config.min_words
            ]
            if len(whitespace_tokenized_spans) == 0:
                raise ValueError("No examples with length >= min_words")

            if self.config.max_words_cutoff:
                last_spans = [
                    x[1][min(self.config.max_words, len(x[1])) - 1][1]
                    for x in whitespace_tokenized_spans
                ]
                data = [
                    x[0][:y] for x, y in zip(whitespace_tokenized_spans, last_spans)
                ]
            else:
                data = [
                    x[0]
                    for x in whitespace_tokenized_spans
                    if len(x[1]) < self.config.max_words
                ]
                if len(data) == 0:
                    raise ValueError("No examples with length < max_words")

            # TODO: why shuffle
            # random.seed(0)
            # random.shuffle(data)

            data = data[: self.config.max_data]

            # If there is mask tokenizer, keep only examples with <= 512 tokens according to mask_tokenizer
            # this step has the extra effect of removing examples with low-quality/garbage content
            if mask_tokenizer:
                tokenized_data = mask_tokenizer(data)
                new_data = []
                for i, (x, y) in enumerate(zip(data, tokenized_data["input_ids"])):
                    if len(y) <= self.config.max_tokens:
                        new_data.append(x)
                    else:
                        print(
                            "Trimming text to nearest word that fits within mask tokenizer window"
                        )
                        max_token_char_span = tokenized_data.token_to_chars(
                            i, self.config.max_tokens - 1
                        )
                        x = x[: max_token_char_span.end]
                        token_truncated_word_spans = list(
                            wsp_tokenizer.span_tokenize(x)
                        )

                        # Pop off the last "word" since it may be a word piece
                        second_last_span = token_truncated_word_spans[-2]
                        x = x[: second_last_span[1]]

                        new_len = len(mask_tokenizer(x)["input_ids"])
                        assert new_len <= self.config.max_tokens
                        new_data.append(x)
                data = new_data

            # print stats about remainining data
            print(f"Total number of samples: {len(data)}")
            print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

            if n_samples > len(data):
                print(f"WARNING: n_samples ({n_samples}) > len(data) ({len(data)})")
            '''

        # Sample 'n_samples' examples
        data = data[:n_samples]

        # Save to cache (if requested)
        if self.config.dump_cache:
            self.dump_to_cache(data, data_split)

        return data

    def dump_to_cache(self, data, data_split):
        filename = self._get_name_to_save()
        custom_datasets.dump_to_cache(
            data,
            self.cache_dir,
            data_split,
            filename,
            min_length=self.config.min_words,
            max_length=self.config.max_words,
            n_samples=self.config.n_samples,
            max_tokens=self.config.max_tokens,
        )

    def _get_name_to_save(self):
        if self.config.specific_source and self.name == "the_pile":
            processed_source = sourcename_process(self.config.specific_source)
            filename = f"{self.name}_{processed_source}"
        else:
            filename = self.name
        return filename


def select_shard(ds, shard_size: int = 5000, shard_env_var="SHARD_ID"):
    """
    Slice dataset into a shard based on SHARD_ID env var.
    Adds an 'id' column from the 'url' field.
    """
    shard_id = int(os.environ[shard_env_var])
    start = shard_id * shard_size
    end = min((shard_id + 1) * shard_size, len(ds))
    return ds.select(range(start, end))


def strip_newlines(text):
    """
    Strip newlines from each example; replace one or more newlines with a single space
    """
    return " ".join(text.split())


def trim_to_shorter_length(text_a: str, text_b: str, max_length: int = None):
    """
    Truncate to shorter of o and s
    """
    shorter_length = min(len(text_a.split(" ")), len(text_b.split(" ")))
    if max_length is not None:
        shorter_length = min(shorter_length, max_length)
    text_a = " ".join(text_a.split(" ")[:shorter_length])
    text_b = " ".join(text_b.split(" ")[:shorter_length])
    return text_a, text_b


def truncate_to_substring(text: str, substring: str, idx_occurrence: int):
    """
    Truncate everything after the idx_occurrence occurrence of substring
    """
    assert idx_occurrence > 0, "idx_occurrence must be > 0"
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def pile_selection_utility(data, key: str, wanted_source: str = None):
    """
    Filter and select data corresponding to source, if requested.
    """
    if wanted_source is None:
        return data[key]
    wanted_data = []
    # Pick sources that match requested source
    for datum in data:
        if datum["meta"]["pile_set_name"] == wanted_source:
            wanted_data.append(datum[key])
    return wanted_data


def sourcename_process(x: str):
    """
        Helper function to process source name.
    """
    return x.replace(" ", "_").replace("-", "_").lower()


def drop_last_word(text):
    """
        Drop the last word from a given text.
    """
    return " ".join(text.split(" ")[:-1])
