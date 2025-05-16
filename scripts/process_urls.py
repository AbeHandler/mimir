import json
import pandas as pd
from urllib.parse import urlparse

def normalize_domain(url: str) -> str:
    url = url.strip('"')
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc

# Path to your JSON file
INPUT_PATH = "tmp_results/olmobypublisher/allenai_OLMo-7B/abehandlerorg/olmobypublisherdev/min_k_results.json"

# Load the JSON data
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

# Extract id_to_score (handles nested structure if present)
id_to_score = data.get("id_to_score", {})
# If nested under member/nonmember, flatten by choosing 'member'
if all(isinstance(v, dict) for v in id_to_score.values()):
    id_to_score = id_to_score.get("member", {})

# Build a list of rows with domain, score, method
rows = []
for url, score in id_to_score.items():
    domain = normalize_domain(url)
    rows.append({"domain": domain, "score": score, "method": "min_k"})

# Create DataFrame and display
df = pd.DataFrame(rows)

df.to_csv('a.csv')