"""Phase 2: Food-101 class -> USDA FoodData Central nutrition per 100 g.
Source: FDC_API_KEY set -> FDC search API (1 call/class). Otherwise the official USDA bulk CSV
downloads (SR Legacy + FNDDS survey; DEMO_KEY allows only ~10 calls/hour so it cannot cover 101 classes).
Missing values stay null (empty). Writes data/nutrition.csv (never printed) + reports/mapping_review.md; prints aggregate counts only.

Inputs:  data/labels.json (written by `eval`); USDA bulk zips (downloaded once into data/usda/) or FDC_API_KEY.
Outputs: data/nutrition.csv, reports/mapping_review.md.
Example: python run.py data
"""
import argparse
import csv
import io
import json
import os
import time
import urllib.request
import zipfile

import requests

from food_lens import console
from food_lens.config import ROOT, load_config
from food_lens.usda import NUTRIENTS, best_match

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--api", action="store_true", help="require FDC_API_KEY and use the FDC search API instead of bulk CSVs")
args = ap.parse_args()
console.setup_logging()
console.banner("Build nutrition table", "Maps each of the 101 Food-101 classes to a USDA FoodData Central row (per 100 g).",
               "aggregate counts only (matched / with kcal / needing review); missing values stay empty, never invented")
cfg = load_config()
data = ROOT / cfg["paths"]["data"]
labels = json.loads((data / "labels.json").read_text())
# Hand-written search phrases where the class name is not a USDA food name (reviewed in mapping_review.csv).
QUERY = {
    "bibimbap": "bibimbap", "caprese_salad": "tomato mozzarella salad",
    "chicken_curry": "chicken curry", "club_sandwich": "club sandwich", "croque_madame": "ham cheese sandwich",
    "deviled_eggs": "deviled egg", "eggs_benedict": "eggs benedict", "french_fries": "potato french fried",
    "gnocchi": "gnocchi", "hot_dog": "frankfurter", "huevos_rancheros": "huevos rancheros",
    "macaroni_and_cheese": "macaroni and cheese", "miso_soup": "miso soup", "nachos": "nachos",
    "onion_rings": "onion rings", "pad_thai": "pad thai", "pancakes": "pancakes", "peking_duck": "duck roasted",
    "pho": "beef noodle soup", "pizza": "pizza cheese", "ravioli": "ravioli", "risotto": "risotto",
    "spaghetti_bolognese": "spaghetti with meat sauce", "spaghetti_carbonara": "spaghetti carbonara",
    "waffles": "waffle", "tacos": "taco", "sushi": "sushi", "sashimi": "fish tuna raw",
    "baby_back_ribs": "pork ribs", "filet_mignon": "beef tenderloin", "prime_rib": "beef rib", "donuts": "doughnut",
    "cup_cakes": "cupcake", "omelette": "omelet", "spring_rolls": "egg roll", "macarons": "macaroon", "gyoza": "pork dumpling",
    "seaweed_salad": "seaweed", "foie_gras": "liver pate goose", "pulled_pork_sandwich": "pork sandwich",
    "lobster_roll_sandwich": "lobster salad", "fish_and_chips": "fish chips", "beef_tartare": "beef raw",
}
# FNDDS food_nutrient.csv carries legacy nutrient numbers (208=energy kcal) instead of FDC nutrient ids.
NBR = {208: 1008, 203: 1003, 204: 1004, 205: 1005, 291: 1079}
FIELDS = ["class", "fdc_id", "matched_description", "data_type", "match_score", *NUTRIENTS.values(), "source_url"]


def q(label: str) -> str:
    return QUERY.get(label, label.replace("_", " "))


def load_bulk() -> tuple[list, dict]:
    urls = [
        "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip",
        "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2024-10-31.zip",
    ]
    foods, nuts = [], {}
    cache = data / "usda"
    cache.mkdir(parents=True, exist_ok=True)
    for u in urls:
        zp = cache / u.rsplit("/", 1)[1]
        if not zp.exists():
            urllib.request.urlretrieve(u, zp)
        with zipfile.ZipFile(zp) as z:
            def rd(name):
                n = next(x for x in z.namelist() if x.endswith("/" + name) or x == name)
                return csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8"))
            for r in rd("food.csv"):
                foods.append((int(r["fdc_id"]), r["description"], r["data_type"]))
            seen = {}
            for r in rd("food_nutrient.csv"):
                nid = NBR.get(int(r["nutrient_id"]), int(r["nutrient_id"]))  # FNDDS uses nutrient numbers
                seen[nid] = seen.get(nid, 0) + 1
                if nid in NUTRIENTS and r["amount"] != "":
                    nuts.setdefault(int(r["fdc_id"]), {})[NUTRIENTS[nid]] = float(r["amount"])
            console.hint(f"{zp.name}: nutrient rows={sum(seen.values())} distinct ids={len(seen)} top={sorted(seen, key=seen.get, reverse=True)[:6]} energy_ids={[i for i in (1008, 2047, 2048) if i in seen]}")
    return foods, nuts


def api_lookup(label: str, key: str):
    r = requests.get(cfg["nutrition"]["fdc_url"], params={
        "api_key": key, "query": q(label), "pageSize": 15, "dataType": cfg["nutrition"]["data_types"]}, timeout=30)
    r.raise_for_status()
    foods = r.json().get("foods", [])
    m = best_match(q(label), [(f["fdcId"], f["description"], f["dataType"]) for f in foods])
    if not m:
        return None, {}
    f = next(x for x in foods if x["fdcId"] == m[0])
    vals = {NUTRIENTS[n["nutrientId"]]: n["value"] for n in f["foodNutrients"] if n.get("nutrientId") in NUTRIENTS}
    return m, vals


key = os.environ.get("FDC_API_KEY")
if args.api and not key:
    console.fail("--api needs FDC_API_KEY in the environment")
    raise SystemExit(2)
rows, bulk = [], None
if not key:
    bulk = load_bulk()
    console.ok(f"bulk foods={len(bulk[0])} with_nutrients={len(bulk[1])} with_kcal={sum('kcal' in v for v in bulk[1].values())}")
for lab in labels:
    if key:
        m, vals = api_lookup(lab, key)
        time.sleep(0.4)
    else:
        m = best_match(q(lab), bulk[0])
        vals = bulk[1].get(m[0], {}) if m else {}
    row = {"class": lab, **{k: "" for k in FIELDS[1:]}}
    if m:
        row.update(fdc_id=m[0], matched_description=m[1], data_type=m[2], match_score=round(m[3], 2),
                   source_url=f"https://fdc.nal.usda.gov/food-details/{m[0]}/nutrients",
                   **{k: vals.get(k, "") for k in NUTRIENTS.values()})
    rows.append(row)
with open(data / "nutrition.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, FIELDS)
    w.writeheader()
    w.writerows(rows)
reports = ROOT / cfg["paths"]["reports"]
reports.mkdir(exist_ok=True)
n_match = sum(1 for r in rows if r["fdc_id"])
n_kcal = sum(1 for r in rows if r["kcal"] != "")
review = [r for r in rows if (not r["fdc_id"]) or float(r["match_score"] or 0) < 0.95 or not r["kcal"]]
src = "api" if key else "bulk"
lines = [f"# Nutrition mapping review (source={src})", "",
         f"- classes: {len(rows)}; matched: {n_match}; with kcal: {n_kcal}; needing review: {len(review)}", "",
         "| class | query | matched description | score | kcal | note |", "|---|---|---|---|---|---|"]
for r in review:
    note = "no match (nulls kept)" if not r["fdc_id"] else ("no kcal" if not r["kcal"] else "partial token match")
    lines.append(f"| {r['class']} | {q(r['class'])} | {r['matched_description']} | {r['match_score']} | {r['kcal']} | {note} |")
(reports / "mapping_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
console.kv_table({"classes": len(rows), "matched": n_match, "with_kcal": n_kcal, "needs_review": len(review), "source": src}, "Nutrition table")
console.ok("wrote data/nutrition.csv and reports/mapping_review.md")
console.note("Nutrition numbers come straight from USDA, so the VLM cannot invent them.",
             "Matches are generic USDA foods, not your plate; unmatched classes stay empty; see reports/mapping_review.md.",
             "Replace the CSV with a small nutrition service and curated per-dish recipes.")
