"""
Prepare ICEWS dataset tensor for einfact.

This script parses ICEWS tab-separated annual event files and constructs
a 4-mode tensor Y of shape (num_countries, num_countries, 20, num_months).
Y[i, j, a, t] counts the number of events from country i to country j
of CAMEO top-level action a in month t.
"""

import argparse
import csv
import glob
import os
import pathlib
import sys
from datetime import datetime
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ICEWS dataset tensor")
    parser.add_argument("--input-dir", type=str, required=True, 
                        help="Directory containing ICEWS .tab files")
    parser.add_argument("--out", type=str, default="DATA/icews.npy", 
                        help="Output tensor path (numpy .npy)")
    parser.add_argument("--heldout-out", type=str, default="DATA/icews_heldout.npy", 
                        help="Output heldout mask path")
    parser.add_argument("--seed", type=int, default=29482,
                        help="Seed for heldout mask generation")
    parser.add_argument("--heldout-frac", type=float, default=0.05,
                        help="Fraction of data to hold out")
    parser.add_argument("--no-heldout", action="store_true", 
                        help="Skip generating heldout mask")
    parser.add_argument("--start-year", type=int, default=1995,
                        help="Start year for processing")
    parser.add_argument("--end-year", type=int, default=2013,
                        help="End year for processing")
    return parser.parse_args()

def get_cameo_top_level(code_str):
    """
    Map CAMEO code to top-level category index 0-19.
    CAMEO top-level categories are 01-20.
    """
    code_str = code_str.strip()
    if not code_str:
        return None
    
    # Handle single digit strings, just in case
    if len(code_str) < 2 and code_str.isdigit():
        code_str = code_str.zfill(2)
        
    top_level = code_str[:2]
    if top_level.isdigit():
        top_val = int(top_level)
        if 1 <= top_val <= 20:
            return top_val - 1
    return None

def find_columns(header):
    """
    Find indices for Date, Source, Target, and CAMEO code columns.
    """
    header_lower = [h.strip().lower() for h in header]
    
    date_col = -1
    source_col = -1
    target_col = -1
    cameo_col = -1
    
    for i, h in enumerate(header_lower):
        if h == 'event date' or h == 'date':
            date_col = i
        elif h == 'source country' or h == 'source name' or 'source country' in h:
            source_col = i
        elif h == 'target country' or h == 'target name' or 'target country' in h:
            target_col = i
        elif 'cameo code' in h or h == 'cameo':
            cameo_col = i
            
    return date_col, source_col, target_col, cameo_col

def parse_date(date_str):
    date_str = date_str.strip()
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    return None

def main():
    args = parse_args()
    
    input_path = pathlib.Path(args.input_dir)
    if not input_path.is_dir():
        print(f"Error: Input directory {args.input_dir} does not exist.")
        sys.exit(1)
        
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    files = glob.glob(str(input_path / '*.tab'))
    if not files:
        files = glob.glob(str(input_path / '*.tsv'))
        
    if not files:
        print(f"Error: No .tab or .tsv files found in {args.input_dir}.")
        sys.exit(1)
        
    events = []
    countries = set()
    
    print(f"Reading files from {args.input_dir}...")
    
    # Enable parsing large fields in case some records are long
    csv.field_size_limit(sys.maxsize)
    
    total_rows = 0
    valid_events = 0
    
    for f in sorted(files):
        print(f"Processing {os.path.basename(f)}...")
        with open(f, 'r', encoding='utf-8', errors='replace') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            try:
                header = next(reader)
            except StopIteration:
                continue
                
            d_col, s_col, t_col, c_col = find_columns(header)
            if -1 in (d_col, s_col, t_col, c_col):
                print(f"Warning: Missing columns in {f}. Header: {header}")
                continue
                
            for row in reader:
                total_rows += 1
                if len(row) <= max(d_col, s_col, t_col, c_col):
                    continue
                    
                date_str = row[d_col]
                src = row[s_col].strip()
                tgt = row[t_col].strip()
                cameo = row[c_col].strip()
                
                if not src or not tgt:
                    continue
                    
                dt = parse_date(date_str)
                if dt is None:
                    continue
                    
                if not (args.start_year <= dt.year <= args.end_year):
                    continue
                    
                month_idx = (dt.year - args.start_year) * 12 + (dt.month - 1)
                
                action_idx = get_cameo_top_level(cameo)
                if action_idx is None:
                    continue
                    
                countries.add(src)
                countries.add(tgt)
                events.append((src, tgt, action_idx, month_idx))
                valid_events += 1

    country_list = sorted(list(countries))
    country_to_idx = {c: i for i, c in enumerate(country_list)}
    
    num_countries = len(country_list)
    num_actions = 20
    num_months = (args.end_year - args.start_year + 1) * 12
    
    print("\nDataset Statistics:")
    print(f"Total rows parsed: {total_rows}")
    print(f"Valid events matching criteria: {valid_events}")
    print(f"Unique countries: {num_countries}")
    print(f"Time span: {args.start_year} to {args.end_year} ({num_months} months)")
    
    print("\nConstructing tensor Y...")
    Y = np.zeros((num_countries, num_countries, num_actions, num_months), dtype=np.float32)
    
    for src, tgt, act_idx, time_idx in events:
        s_idx = country_to_idx[src]
        t_idx = country_to_idx[tgt]
        Y[s_idx, t_idx, act_idx, time_idx] += 1
        
    # Ensure C-contiguous
    Y = np.ascontiguousarray(Y)
    
    print(f"Saving tensor to {out_path} ...")
    np.save(out_path, Y)
    
    # Save auxiliary files
    countries_file = out_path.parent / f"{out_path.stem}_countries.txt"
    with open(countries_file, 'w', encoding='utf-8') as f:
        for c in country_list:
            f.write(c + '\n')
            
    actions_file = out_path.parent / f"{out_path.stem}_actions.txt"
    with open(actions_file, 'w', encoding='utf-8') as f:
        for i in range(20):
            f.write(f"{i+1:02d}\n")
            
    if not args.no_heldout:
        print("\nGenerating heldout mask...")
        heldout_path = pathlib.Path(args.heldout_out)
        heldout_path.parent.mkdir(parents=True, exist_ok=True)
        
        flat = np.empty(Y.size, dtype=np.bool_)
        multiplier = np.uint64(11400714819323198485)
        block = 10_000_000
        for start in range(0, Y.size, block):
            ind = np.arange(start, min(start+block, Y.size), dtype=np.uint64)
            flat[start:start+ind.size] = ((ind * multiplier + np.uint64(args.seed)) % 100) < int(args.heldout_frac * 100)
            
        heldout = flat.reshape(Y.shape)
        print(f"Saving heldout mask to {heldout_path} ...")
        np.save(heldout_path, heldout)
        
    print("Done!")

if __name__ == '__main__':
    main()
