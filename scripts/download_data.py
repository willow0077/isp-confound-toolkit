"""Download the two scPerturb datasets used by the toolkit.

Source: scPerturb (Peidli et al., Nat Methods 2024) — https://scperturb.org
Hosted on Zenodo as the scPerturb "RNA and protein h5ad files" collection
(concept DOI 10.5281/zenodo.7041848; latest record at time of writing: 13350497).

This script does NOT hard-code per-file URLs. It queries the Zenodo record's
file listing at runtime and matches by filename, so it downloads the right file
or fails loudly — it never guesses a URL. If the record has moved, update
ZENODO_RECORD below or grab the current links from https://scperturb.org.

Files (placed under ./data, or $ISP_DATA_ROOT):
  FrangiehIzar2021_RNA.h5ad
  ReplogleWeissman2022_K562_essential.h5ad

Usage:
  python scripts/download_data.py
"""
import sys
import json
import urllib.request
from isp_confound.config import DATA_ROOT

ZENODO_RECORD = "13350497"   # scPerturb RNA + protein h5ad collection (concept 7041848)
FILES = [
    "FrangiehIzar2021_RNA.h5ad",
    "ReplogleWeissman2022_K562_essential.h5ad",
]


def resolve_file_urls(record: str) -> dict:
    """Return {filename: download_url} from the Zenodo record's file listing."""
    api = f"https://zenodo.org/api/records/{record}"
    with urllib.request.urlopen(api, timeout=60) as resp:
        meta = json.load(resp)
    out = {}
    for f in meta.get("files", []):
        # Zenodo schema: file key + links.self (download URL)
        key = f.get("key") or f.get("filename")
        url = (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download")
        if key and url:
            out[key] = url
    return out


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        urls = resolve_file_urls(ZENODO_RECORD)
    except Exception as e:  # network / record moved
        sys.exit(
            f"Could not query Zenodo record {ZENODO_RECORD}: {e}\n"
            f"Get the current download links from https://scperturb.org"
        )

    for name in FILES:
        dest = DATA_ROOT / name
        if dest.exists():
            print(f"[skip] {name} already present at {dest}")
            continue
        url = urls.get(name)
        if not url:
            print(
                f"[!] {name} not found in Zenodo record {ZENODO_RECORD}. "
                f"Check https://scperturb.org for its current location."
            )
            continue
        print(f"[get] {name}\n      <- {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"      saved -> {dest}")

    print("\nDone. (Set ISP_DATA_ROOT to download elsewhere.)")


if __name__ == "__main__":
    main()
