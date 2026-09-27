"""
Inspect predicted matches side-by-side with feature values and hard-veto status.
"""

import os
import sys
import pandas as pd
from typing import Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CFG
from src.features import compute_pair_features, df_to_record_dict
from src.preprocessing import preprocess_dataframe
from src.utils import parse_id_list

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    test_dir = os.path.abspath("../../dataset/test")
    matching_tsv = os.path.abspath("../../output/matching_results.tsv")
    
    print(f"Loading test source files from {test_dir}...")
    s1_df = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    s2_df = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3_df = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", dtype=str).fillna("")
    
    matches_df = pd.read_csv(matching_tsv, sep="\t", dtype=str).fillna("")
    matched_rows = matches_df[matches_df["matched_entity_ids"] != ""]
    print(f"Total S1 entities with matches: {len(matched_rows):,}")
    
    sample_s1 = matched_rows.sample(n=min(30, len(matched_rows)), random_state=CFG.seed)
    sample_s1_ids = set(sample_s1["source1_entity_id"])
    
    matched_c_ids = set()
    for _, row in sample_s1.iterrows():
        matched_c_ids.update(parse_id_list(row["matched_entity_ids"]))
        
    s1_sub = s1_df[s1_df["entity_id"].isin(sample_s1_ids)].copy()
    s2_sub = s2_df[s2_df["entity_id"].isin(matched_c_ids)].copy()
    s3_sub = s3_df[s3_df["entity_id"].isin(matched_c_ids)].copy()
    
    s1_proc = preprocess_dataframe(s1_sub, suffixes=CFG.legal_suffixes, abbrevs=CFG.address_abbrevs)
    s2_proc = preprocess_dataframe(s2_sub, suffixes=CFG.legal_suffixes, abbrevs=CFG.address_abbrevs)
    s3_proc = preprocess_dataframe(s3_sub, suffixes=CFG.legal_suffixes, abbrevs=CFG.address_abbrevs)
    
    s1_records = df_to_record_dict(s1_proc)
    pool_records = {**df_to_record_dict(s2_proc), **df_to_record_dict(s3_proc)}
    
    # Store raw names/addresses for display
    raw_info = {}
    for df in [s1_sub, s2_sub, s3_sub]:
        for _, r in df.iterrows():
            raw_info[r["entity_id"]] = {
                "name": r["business_name"],
                "addr": r["business_address"],
                "country": r["country"],
            }
            
    print("\n" + "="*90)
    print("SIDE-BY-SIDE INSPECTION OF REAL PREDICTED MATCHES")
    print("="*90)
    
    count = 0
    vetoed_count = 0
    allowed_count = 0
    
    for _, row in sample_s1.iterrows():
        s1_id = row["source1_entity_id"]
        matched_ids = parse_id_list(row["matched_entity_ids"])
        s1_rec = s1_records.get(s1_id)
        s1_raw = raw_info.get(s1_id, {})
        if not s1_rec:
            continue
            
        for cid in matched_ids:
            cand_rec = pool_records.get(cid)
            cand_raw = raw_info.get(cid, {})
            if not cand_rec:
                continue
                
            count += 1
            feats = compute_pair_features(
                s1_name=s1_rec["name_clean"],
                s1_addr=s1_rec["addr_clean"],
                s1_name_tokens=s1_rec["name_tokens"],
                s1_addr_tokens=s1_rec["addr_tokens"],
                s1_addr_numbers=s1_rec["addr_numbers"],
                s1_country=s1_rec["country"],
                cand_name=cand_rec["name_clean"],
                cand_addr=cand_rec["addr_clean"],
                cand_name_tokens=cand_rec["name_tokens"],
                cand_addr_tokens=cand_rec["addr_tokens"],
                cand_addr_numbers=cand_rec["addr_numbers"],
                cand_country=cand_rec["country"],
                s1_name_dropped=s1_rec["name_script_dropped"],
                cand_name_dropped=cand_rec["name_script_dropped"],
            )
            
            token_sort = feats[1]
            token_set = feats[2]
            addr_sort = feats[11]
            mismatch = feats[25]
            overlap = feats[26]
            prefix_sim = feats[27]
            conflict = feats[28]
            
            # Hard veto rule
            veto = (mismatch == 1.0 and token_sort < CFG.veto_distinctive_mismatch_max_sort) or \
                   (conflict == 1.0 and token_sort < CFG.veto_prefix_suffix_conflict_max_sort)
            
            if veto:
                vetoed_count += 1
                status = "[VETO TRIGGERED: FALSE MATCH BLOCKED]"
            else:
                allowed_count += 1
                status = "[GENUINE MATCH CONFIRMED]"
                
            print(f"\n--- Pair #{count}: {status} ---")
            print(f"S1  [{s1_raw.get('country')}]: {s1_raw.get('name')}")
            print(f"    Clean: '{s1_rec['name_clean']}' | Addr: {s1_raw.get('addr')}")
            print(f"Cand[{cand_raw.get('country')}]: {cand_raw.get('name')}")
            print(f"    Clean: '{cand_rec['name_clean']}' | Addr: {cand_raw.get('addr')}")
            print(f"Metrics: NameTokenSort={token_sort:.2f} | NameTokenSet={token_set:.2f} | AddrSort={addr_sort:.2f}")
            print(f"         DistinctiveMismatch={mismatch:.0f} | OverlapRatio={overlap:.2f} | PrefixSim={prefix_sim:.2f} | Conflict={conflict:.0f}")

    print("\n" + "="*90)
    print(f"SUMMARY: {count} total sampled pairs | {allowed_count} Genuine Matches | {vetoed_count} Blocked by Veto")
    print("="*90)

if __name__ == "__main__":
    main()
