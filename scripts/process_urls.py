import json
from urllib.parse import urlparse

def normalize_domain(url: str) -> str:
    url = url.strip('"')
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc

# adjust this path to wherever your file actually lives
INPUT_PATH = "tmp_results/olmobypublisher/allenai_OLMo-7B/abehandlerorg/olmobypublisherdev/min_k_results.json"

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

# if your id_to_score is nested under "member"/"nonmember", you can handle both:
raw = data["id_to_score"]

# If it’s a two‐level dict:
if all(isinstance(v, dict) for v in raw.values()):
    normalized = {
        group: { normalize_domain(url): score for url, score in mapping.items() }
        for group, mapping in raw.items()
    }
# Otherwise it’s a flat mapping:
else:
    normalized = { normalize_domain(url): score for url, score in raw.items() }

print(json.dumps(normalized, indent=2))
