"""
expand_dataset.py

Expands the art dataset for the IDC project to 100 images per category:
- classical: 100 images (Renaissance & Baroque masters: Raphael, Rembrandt, Caravaggio, Botticelli, Vermeer)
- modern: 100 images (Modernist masters: Kandinsky, Cézanne, Gauguin, Monet)
- picasso: 100 images (Paintings by Pablo Picasso across his periods)
- van_gogh: 100 images (Paintings by Vincent van Gogh)

Sources: Wikipedia & Wikimedia Commons API with academic User-Agent attribution.
"""

import io
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import requests

sys.stdout.reconfigure(encoding="utf-8")

USER_AGENT = "IDCResearchBot/1.0 (academic computational aesthetics research; student@univ.edu)"
HEADERS = {"User-Agent": USER_AGENT}
WIKI_API = "https://en.wikipedia.org/w/api.php"
MAX_IMAGE_DIM = 1600
MIN_IMAGE_DIM = 300
TARGET_COUNT = 100


def clean_filename(title: str, max_len: int = 50) -> str:
    """Sanitize title for safe filename usage."""
    # Remove 'File:' prefix if present
    if title.startswith("File:"):
        title = title[5:]
    # Remove extension if in title
    title = re.sub(r"\.(jpg|jpeg|png|tif|tiff|webp)$", "", title, flags=re.I)
    # Remove unwanted punctuation
    title = re.sub(r"[^\w\s-]", "", title).strip()
    title = re.sub(r"[\s_]+", "_", title)
    return title[:max_len].strip("_")


def download_and_save_image(url: str, dest_path: Path) -> bool:
    """
    Downloads image from URL, verifies with PIL, converts to RGB,
    resizes if larger than MAX_IMAGE_DIM, and saves as JPEG.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200 or len(resp.content) < 5000:
            return False

        with Image.open(io.BytesIO(resp.content)) as im:
            w, h = im.size
            if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
                return False

            im_rgb = im.convert("RGB")

            # Resize if excessive
            if max(w, h) > MAX_IMAGE_DIM:
                scale = MAX_IMAGE_DIM / max(w, h)
                new_size = (int(w * scale), int(h * scale))
                im_rgb = im_rgb.resize(new_size, Image.Resampling.LANCZOS)

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            im_rgb.save(dest_path, "JPEG", quality=95)
            return True

    except Exception as e:
        return False


def get_image_urls_from_files(file_titles: List[str]) -> List[Tuple[str, str]]:
    """Batch fetch thumbnail URLs for a list of File:... titles."""
    results = []
    chunk_size = 50
    for i in range(0, len(file_titles), chunk_size):
        chunk = file_titles[i:i + chunk_size]
        params = {
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "iiurlwidth": "1280",
            "format": "json"
        }
        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20).json()
            pages = r.get("query", {}).get("pages", {})
            for p in pages.values():
                title = p.get("title", "")
                infos = p.get("imageinfo", [])
                if infos:
                    info = infos[0]
                    thumb = info.get("thumburl") or info.get("url")
                    w = info.get("width", 0)
                    h = info.get("height", 0)
                    if thumb and w >= MIN_IMAGE_DIM and h >= MIN_IMAGE_DIM:
                        results.append((title, thumb))
        except Exception as e:
            print(f"Error querying batch: {e}")
        time.sleep(0.1)
    return results


def get_images_from_wiki_pages(page_titles: List[str]) -> List[Tuple[str, str]]:
    """Extract list of painting files from given Wikipedia list pages."""
    all_files = []
    valid_exts = (".jpg", ".jpeg", ".png")
    skip_keywords = [
        "icon", "logo", "arrow", "flag", "portal", "symbol", "stub",
        "wikimedia", "commons", "wikidata", "filewarning", "signature",
        "diagram", "map", "tomb", "statue", "monument", "plaque"
    ]

    for page_title in page_titles:
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "images",
            "format": "json"
        }
        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20).json()
            images = r.get("parse", {}).get("images", [])
            for img in images:
                if any(img.lower().endswith(ext) for ext in valid_exts):
                    if not any(k in img.lower() for k in skip_keywords):
                        all_files.append(f"File:{img}")
        except Exception as e:
            print(f"Error parsing page {page_title}: {e}")

    # Remove duplicates preserving order
    seen = set()
    unique_files = [f for f in all_files if not (f in seen or seen.add(f))]
    return get_image_urls_from_files(unique_files)


def get_picasso_candidates() -> List[Tuple[str, str]]:
    """Fetch Picasso painting images using Wikipedia articles in Category:Paintings by Pablo Picasso."""
    articles = []
    cmcontinue = None
    while len(articles) < 250:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": "Category:Paintings by Pablo Picasso",
            "gcmlimit": "100",
            "format": "json"
        }
        if cmcontinue:
            params["gcmcontinue"] = cmcontinue
        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20).json()
            pages = r.get("query", {}).get("pages", {})
            for p in pages.values():
                title = p.get("title")
                if not any(skip in title.lower() for skip in ["list", "template", "referendum", "film"]):
                    articles.append(title)
            if "continue" in r:
                cmcontinue = r["continue"]["gcmcontinue"]
            else:
                break
        except Exception:
            break
        time.sleep(0.1)

    candidates = []
    chunk_size = 50
    for i in range(0, len(articles), chunk_size):
        batch = articles[i:i + chunk_size]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "pageimages",
            "pithumbsize": "1280",
            "format": "json"
        }
        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20).json()
            pages = r.get("query", {}).get("pages", {})
            for p in pages.values():
                title = p.get("title", "")
                thumb = p.get("thumbnail", {}).get("source")
                if thumb:
                    candidates.append((title, thumb))
        except Exception:
            pass
        time.sleep(0.1)

    return candidates


def expand_category(category_name: str, candidates: List[Tuple[str, str]]):
    """Fills the category folder until it reaches exactly TARGET_COUNT images."""
    cat_dir = Path(f"data/images/{category_name}")
    existing_files = sorted(list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpeg")))
    current_count = len(existing_files)
    print(f"\n[{category_name.upper()}] Current count: {current_count}. Target: {TARGET_COUNT}.")

    if current_count >= TARGET_COUNT:
        print(f"[{category_name.upper()}] Already has {current_count} images. Skipping.")
        return

    needed = TARGET_COUNT - current_count
    print(f"[{category_name.upper()}] Downloading {needed} additional images...")

    next_idx = current_count + 1
    added = 0

    for title, url in candidates:
        if next_idx > TARGET_COUNT:
            break

        safe_title = clean_filename(title)
        filename = f"{next_idx:02d}_{safe_title}.jpg"
        dest_path = cat_dir / filename

        print(f"  [{next_idx}/{TARGET_COUNT}] Fetching: {safe_title} ... ", end="", flush=True)
        success = download_and_save_image(url, dest_path)
        if success:
            print("OK")
            next_idx += 1
            added += 1
        else:
            print("Failed (skipped)")

        time.sleep(0.1)

    final_count = len(list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpeg")))
    print(f"[{category_name.upper()}] Finished! Total images now: {final_count} (added {added}).")


def main():
    print("=" * 60)
    print("IDC ART DATASET EXPANSION (TARGET: 100 IMAGES PER CATEGORY)")
    print("=" * 60)

    # 1. Van Gogh
    print("\n--- Collecting Van Gogh Candidates ---")
    van_gogh_cands = get_images_from_wiki_pages([
        "List of works by Vincent van Gogh"
    ])
    print(f"Collected {len(van_gogh_cands)} Van Gogh candidates.")
    expand_category("van_gogh", van_gogh_cands)

    # 2. Picasso
    print("\n--- Collecting Picasso Candidates ---")
    picasso_cands = get_picasso_candidates()
    print(f"Collected {len(picasso_cands)} Picasso candidates.")
    expand_category("picasso", picasso_cands)

    # 3. Classical
    print("\n--- Collecting Classical Art Candidates ---")
    classical_cands = get_images_from_wiki_pages([
        "List of paintings by Raphael",
        "List of paintings by Rembrandt",
        "List of paintings by Caravaggio",
        "List of paintings by Sandro Botticelli",
        "List of paintings by Johannes Vermeer"
    ])
    print(f"Collected {len(classical_cands)} Classical candidates.")
    expand_category("classical", classical_cands)

    # 4. Modern
    print("\n--- Collecting Modern Art Candidates ---")
    modern_cands = get_images_from_wiki_pages([
        "List of paintings by Wassily Kandinsky",
        "List of paintings by Paul Cézanne",
        "List of paintings by Paul Gauguin",
        "List of paintings by Claude Monet"
    ])
    print(f"Collected {len(modern_cands)} Modern candidates.")
    expand_category("modern", modern_cands)

    print("\n" + "=" * 60)
    print("DATASET EXPANSION COMPLETE")
    print("=" * 60)
    for cat in ["classical", "modern", "picasso", "van_gogh"]:
        files = list(Path(f"data/images/{cat}").glob("*"))
        print(f"  {cat:12s}: {len(files)} images")


if __name__ == "__main__":
    main()
