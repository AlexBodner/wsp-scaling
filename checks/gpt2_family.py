"""Weight counts and popularity of the GPT-2 family, from public configs and the Hugging Face API."""
import json
import urllib.parse
import urllib.request


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "x"})
    return json.load(urllib.request.urlopen(request, timeout=25))


for name in ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
    query = urllib.parse.urlencode({"expand[]": ["downloads", "likes"]}, doseq=True)
    info = get(f"https://huggingface.co/api/models/{name}?{query}")
    config = get(f"https://huggingface.co/{name}/raw/main/config.json")
    layers, width = config["n_layer"], config["n_embd"]
    # Token and position tables, tied head, then each block's attention, MLP and two LayerNorms.
    total = config["vocab_size"] * width + config["n_positions"] * width
    total += layers * (12 * width**2 + 9 * width + 4 * width) + 2 * width
    print(name, layers, width, f"{total:,}", info.get("downloads"), info.get("likes"))
