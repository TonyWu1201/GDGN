import pickle
import re
import urllib.request
from pathlib import Path

import pandas as pd
import pubchempy as pcp
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GDSC_RAW = PROJECT_ROOT / "data" / "raw" / "drug_sensitivity" / "GDSC2_fitted_dose_response_27Oct23.csv"
DGIDB_DRUGS = PROJECT_ROOT / "data" / "raw" / "drug_gene_interaction" / "drugs.tsv"
DGIDB_INTERACTIONS = PROJECT_ROOT / "data" / "raw" / "drug_gene_interaction" / "interactions.tsv"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "drug_structures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GDSC_CID_CACHE = OUT_DIR / "gdsc_to_cid.csv"
CID_SYN_CACHE = OUT_DIR / "cid_to_synonyms.pkl"
DGIDB_CID_PATH = OUT_DIR / "dgidb_to_cid.csv"
COMMON_DRUGS_PATH = PROJECT_ROOT / "data" / "common_drugs_pubchem.csv"

XREF_FALLBACK = False

PAREN_RE = re.compile(r"\s*\([^)]*\)\s*")
PUNCT_RE = re.compile(r"[^a-z0-9]")


def normalize(name):
    if not isinstance(name, str) or not name.strip():
        return []
    base = PAREN_RE.sub(" ", name).strip().lower()
    keys = []
    if base:
        keys.append(base)
    compact = PUNCT_RE.sub("", base)
    if compact and compact not in keys:
        keys.append(compact)
    return keys


def query_cid(name):
    candidates = [name, PAREN_RE.sub(" ", name).strip()]
    seen = set()
    for cand in candidates:
        if not cand or cand.lower() in seen:
            continue
        seen.add(cand.lower())
        try:
            res = pcp.get_compounds(cand, "name")
            if res and res[0].cid:
                return res[0].cid
        except Exception:
            continue
    return None


def map_gdsc_to_cid(gdsc_names):
    existing = {}
    if GDSC_CID_CACHE.exists():
        cached = pd.read_csv(GDSC_CID_CACHE)
        existing = dict(zip(cached["gdsc_name"], cached["cid"]))
    rows = [(n, existing[n]) for n in gdsc_names if n in existing]
    todo = [n for n in gdsc_names if n not in existing]
    for name in tqdm(todo, desc="GDSC name -> CID"):
        rows.append((name, query_cid(name)))
    df = pd.DataFrame(rows, columns=["gdsc_name", "cid"])
    df.to_csv(GDSC_CID_CACHE, index=False)
    return df


def load_cid_synonyms(cids):
    cache = {}
    if CID_SYN_CACHE.exists():
        with open(CID_SYN_CACHE, "rb") as f:
            cache = pickle.load(f)
    todo = [int(c) for c in cids if pd.notna(c) and int(c) not in cache]
    for cid in tqdm(todo, desc="CID -> synonyms"):
        syns = []
        try:
            results = pcp.get_synonyms(cid, "cid")
            for r in results:
                syns.extend(r.get("Synonym", []))
        except Exception:
            syns = []
        cache[cid] = syns
    with open(CID_SYN_CACHE, "wb") as f:
        pickle.dump(cache, f)
    return cache


def build_name_to_cid(gdsc_df, cid_synonyms):
    mapping = {}

    def add(name, cid):
        for k in normalize(name):
            mapping.setdefault(k, int(cid))

    for row in gdsc_df.itertuples():
        if pd.isna(row.cid):
            continue
        add(row.gdsc_name, row.cid)
        add(f"cid:{int(row.cid)}", row.cid)
        for syn in cid_synonyms.get(int(row.cid), []):
            add(syn, row.cid)
    return mapping


def xref_to_cid(concept_id):
    if not isinstance(concept_id, str) or ":" not in concept_id:
        return None
    prefix, value = concept_id.split(":", 1)
    if prefix == "rxcui":
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/xref/source/NLM-rxnorm/sourceid/{value}/cids/TXT"
    elif prefix == "chembl":
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/xref/source/ChEMBL/sourceid/{value}/cids/TXT"
    else:
        return None
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            txt = resp.read().decode().strip()
        if txt and txt.isdigit():
            return int(txt)
    except Exception:
        return None
    return None


def apply_xref_fallback(dgidb_df, name_to_cid, interactions_df):
    matched_names = set(dgidb_df.loc[dgidb_df["cid"].notna(), "dgidb_name"])
    relevant = interactions_df.loc[~interactions_df["drug_name"].isin(matched_names), ["drug_name", "drug_concept_id"]].drop_duplicates()
    relevant = relevant[relevant["drug_concept_id"].str.startswith(("rxcui:", "chembl:"), na=False)]
    xref_map = {}
    for row in tqdm(relevant.itertuples(index=False), total=len(relevant), desc="xref fallback"):
        if row.drug_name in xref_map:
            continue
        xref_map[row.drug_name] = xref_to_cid(row.drug_concept_id)
    for row in dgidb_df.itertuples():
        if pd.isna(row.cid) and row.dgidb_name in xref_map:
            dgidb_df.at[row.Index, "cid"] = xref_map[row.dgidb_name]
    return dgidb_df


def map_dgidb_to_cid(dgidb_df, name_to_cid, interactions_df):
    cids = []
    for row in tqdm(dgidb_df.itertuples(index=False), total=len(dgidb_df), desc="DGIdb name -> CID"):
        matched = None
        for k in normalize(row.drug_name):
            if k in name_to_cid:
                matched = name_to_cid[k]
                break
        cids.append(matched)
    out = pd.DataFrame({
        "dgidb_name": dgidb_df["drug_name"].values,
        "concept_id": dgidb_df["concept_id"].values,
        "cid": cids,
    })
    if XREF_FALLBACK:
        out = apply_xref_fallback(out, name_to_cid, interactions_df)
    out.to_csv(DGIDB_CID_PATH, index=False)
    return out


def build_common_drugs(gdsc_meta, gdsc_df, dgidb_df):
    gdsc_ok = gdsc_df.dropna(subset=["cid"]).copy()
    gdsc_ok["cid"] = gdsc_ok["cid"].astype(int)
    dgidb_ok = dgidb_df.dropna(subset=["cid"]).copy()
    dgidb_ok["cid"] = dgidb_ok["cid"].astype(int)

    common_cids = set(gdsc_ok["cid"]) & set(dgidb_ok["cid"])
    gdsc_ok = gdsc_ok[gdsc_ok["cid"].isin(common_cids)]
    dgidb_ok = dgidb_ok[dgidb_ok["cid"].isin(common_cids)]

    merged = gdsc_ok.merge(dgidb_ok[["dgidb_name", "cid"]], on="cid", how="inner")
    merged = merged.merge(gdsc_meta, left_on="gdsc_name", right_on="DRUG_NAME", how="left")

    out = merged[["cid", "DRUG_ID", "gdsc_name", "dgidb_name", "PUTATIVE_TARGET", "PATHWAY_NAME"]].copy()
    out.columns = ["CID", "GDSC_DRUG_ID", "GDSC_DRUG_NAME", "DGIDB_DRUG_NAME", "PUTATIVE_TARGET", "PATHWAY_NAME"]
    out = out.drop_duplicates().sort_values(["GDSC_DRUG_NAME", "DGIDB_DRUG_NAME"]).reset_index(drop=True)
    return out


gdsc_raw = pd.read_csv(GDSC_RAW, usecols=["DRUG_ID", "DRUG_NAME", "PUTATIVE_TARGET", "PATHWAY_NAME"])
gdsc_meta = gdsc_raw.drop_duplicates("DRUG_NAME").reset_index(drop=True)
gdsc_names = gdsc_meta["DRUG_NAME"].tolist()

print(f"GDSC unique drugs: {len(gdsc_names)}")

gdsc_df = map_gdsc_to_cid(gdsc_names)
hit = gdsc_df["cid"].notna().sum()
print(f"GDSC mapped to CID: {hit}/{len(gdsc_df)}")

cid_synonyms = load_cid_synonyms(gdsc_df["cid"].tolist())
print(f"Synonym cache size: {len(cid_synonyms)}")

name_to_cid = build_name_to_cid(gdsc_df, cid_synonyms)
print(f"Name->CID dict size: {len(name_to_cid)}")

dgidb_df = pd.read_csv(DGIDB_DRUGS, sep="\t", usecols=["drug_name", "concept_id"])
interactions_df = pd.read_csv(DGIDB_INTERACTIONS, sep="\t", usecols=["drug_name", "drug_concept_id"])

dgidb_cid_df = map_dgidb_to_cid(dgidb_df, name_to_cid, interactions_df)
dgidb_hit = dgidb_cid_df["cid"].notna().sum()
print(f"DGIdb drugs matched to a GDSC CID: {dgidb_hit}/{len(dgidb_cid_df)}")

common = build_common_drugs(gdsc_meta, gdsc_df, dgidb_cid_df)
common.to_csv(COMMON_DRUGS_PATH, index=False)
print(f"Common drugs rows: {len(common)}")
print(f"Unique GDSC drugs in intersection: {common['GDSC_DRUG_NAME'].nunique()}")
print(f"Unique CIDs in intersection: {common['CID'].nunique()}")
print(f"Saved to: {COMMON_DRUGS_PATH}")
