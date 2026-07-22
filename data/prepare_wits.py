"""
Prepare the WITS (World Integrated Trade Solution) tensor for the einfact repository.

The tensor Y[e, i, g, t] = trade value (in thousand USD) where:
- e: country (exporter, 196 countries)
- i: country (importer, 196 countries)
- g: good (HS2 category, 96 product categories)
- t: year (1996-2024, 29 years: t=0 is 1996, t=28 is 2024)

Data can be downloaded from WITS Bulk Download or UN Comtrade.
Ensure your data covers the years you request and contains the necessary columns.
"""

import argparse
import csv
import glob
import os
import sys
import numpy as np


def generate_heldout_mask(y_shape, seed, frac):
    """Generate heldout mask matching the repo's convention."""
    size = np.prod(y_shape)
    flat = np.empty(size, dtype=np.bool_)
    multiplier = np.uint64(11400714819323198485)
    block = 10_000_000
    for start in range(0, size, block):
        ind = np.arange(start, min(start + block, size), dtype=np.uint64)
        flat[start:start + ind.size] = ((ind * multiplier + np.uint64(seed)) % 100) < int(frac * 100)
    return flat.reshape(y_shape)


def get_hs2_index(code_str):
    """Map HS2 code string to index (0-95). Returns -1 if invalid."""
    try:
        # Sometimes codes have 'H2' prefixes or are longer, just take the first 2 digits
        # but in standard WITS they are 1-2 digits, occasionally padded.
        code = int(str(code_str)[:2])
    except ValueError:
        return -1
    
    if 1 <= code <= 76:
        return code - 1
    elif 78 <= code <= 97:
        return code - 2
    return -1


def hs2_index_to_code(index):
    """Map index (0-95) back to 2-digit string."""
    if 0 <= index <= 75:
        return f"{index + 1:02d}"
    elif 76 <= index <= 95:
        return f"{index + 2:02d}"
    return "00"


def find_column(header, candidates):
    header_lower = [h.lower().strip() for h in header]
    # First pass: exact matches
    for c in candidates:
        c_lower = c.lower()
        for i, h in enumerate(header_lower):
            if c_lower == h:
                return i
    # Second pass: substring matches
    for c in candidates:
        c_lower = c.lower()
        for i, h in enumerate(header_lower):
            if c_lower in h:
                return i
    return -1


def process_files(files, start_year, end_year):
    """
    Process all CSV files, extracting trade records.
    Returns:
        records: list of tuples (exporter, importer, hs2_index, year_index, trade_value)
        countries: set of country names/codes
    """
    records = []
    countries = set()
    
    reporter_cands = ['reporteriso', 'reporteriso3', 'exporteriso3', 'reporter', 'exporter']
    partner_cands = ['partneriso', 'partneriso3', 'importeriso3', 'partner', 'importer']
    product_cands = ['cmdcode', 'product code', 'commodity code', 'productcode', 'hs code', 'commodity']
    year_cands = ['period', 'year']
    value_cands = ['primaryvalue', 'trade value', 'tradevalue', 'export (us$ thousand)', 'value']

    for filepath in files:
        print(f"Reading {filepath}...")
        
        # Try utf-8, fallback to latin1
        try:
            f = open(filepath, 'r', encoding='utf-8', errors='replace')
            f.read(1024)
            f.seek(0)
        except UnicodeDecodeError:
            f.close()
            f = open(filepath, 'r', encoding='latin1', errors='replace')
            
        try:
            # Check dialect
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                # Default to comma separated if sniffing fails
                dialect = csv.excel
                
            reader = csv.reader(f, dialect=dialect)
            
            try:
                header = next(reader)
            except StopIteration:
                continue
                
            idx_rep = find_column(header, reporter_cands)
            idx_par = find_column(header, partner_cands)
            idx_prd = find_column(header, product_cands)
            idx_yr = find_column(header, year_cands)
            idx_val = find_column(header, value_cands)
            
            if any(i == -1 for i in [idx_rep, idx_par, idx_prd, idx_yr, idx_val]):
                print(f"Warning: Could not identify all required columns in {filepath}. Header: {header}")
                continue
                
            row_count = 0
            for row in reader:
                if len(row) <= max(idx_rep, idx_par, idx_prd, idx_yr, idx_val):
                    continue
                    
                exporter = row[idx_rep].strip()
                importer = row[idx_par].strip()
                product_str = row[idx_prd].strip()
                year_str = row[idx_yr].strip()
                value_str = row[idx_val].strip()
                
                try:
                    year = int(year_str)
                except ValueError:
                    continue
                    
                if year < start_year or year > end_year:
                    #print(f"Skipped because year {year} not in range")
                    continue
                    
                hs2_idx = get_hs2_index(product_str)
                if hs2_idx == -1:
                    #print(f"Skipped because hs2_idx == -1 for {product_str}")
                    continue
                    
                # Clean value (remove commas, handle missing)
                value_str = value_str.replace(',', '')
                if not value_str:
                    val = 0.0
                else:
                    try:
                        val = float(value_str)
                    except ValueError:
                        val = 0.0
                        
                year_idx = year - start_year
                
                # Check for Export flow if column exists
                if 'flowcode' in [h.lower() for h in header] or 'flowdesc' in [h.lower() for h in header]:
                    flow_idx = find_column(header, ['flowdesc', 'flowcode'])
                    if flow_idx != -1 and 'import' in row[flow_idx].lower():
                        continue
                
                records.append((exporter, importer, hs2_idx, year_idx, val))
                countries.add(exporter)
                countries.add(importer)
                
                row_count += 1
                if row_count % 1000000 == 0:
                    print(f"  Processed {row_count} rows...")
                    
        finally:
            f.close()
            
    return records, countries


def main():
    parser = argparse.ArgumentParser(description="Prepare WITS tensor for einfact.")
    parser.add_argument('--input-dir', help="Directory containing input CSV/TSV files")
    parser.add_argument('--input-file', help="Single input CSV/TSV file")
    parser.add_argument('--out', default='DATA/wits.npy', help="Output file path for tensor")
    parser.add_argument('--heldout-out', default='DATA/wits_heldout.npy', help="Output file path for heldout mask")
    parser.add_argument('--seed', type=int, default=29482, help="Seed for heldout mask generation")
    parser.add_argument('--heldout-frac', type=float, default=0.05, help="Fraction of data to hold out")
    parser.add_argument('--no-heldout', action='store_true', help="Skip generating heldout mask")
    parser.add_argument('--start-year', type=int, default=1996, help="Start year (inclusive)")
    parser.add_argument('--end-year', type=int, default=2024, help="End year (inclusive)")
    args = parser.parse_args()

    if not args.input_dir and not args.input_file:
        print("Error: Must specify --input-dir or --input-file", file=sys.stderr)
        sys.exit(1)

    files = []
    if args.input_file:
        files.append(args.input_file)
    if args.input_dir:
        files.extend(glob.glob(os.path.join(args.input_dir, '*.csv')))
        files.extend(glob.glob(os.path.join(args.input_dir, '*.tsv')))
        files.extend(glob.glob(os.path.join(args.input_dir, '*.txt')))

    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    num_years = args.end_year - args.start_year + 1
    if num_years <= 0:
        print("Error: end-year must be >= start-year", file=sys.stderr)
        sys.exit(1)

    # 1. Read data and collect unique entities
    print(f"Processing data for years {args.start_year}-{args.end_year}...")
    records, countries = process_files(files, args.start_year, args.end_year)
    
    # Sort countries for consistent indexing
    country_list = sorted(list(countries))
    
    # By request, the target tensor size is (196, 196, 96, 29). We'll trim or pad the countries list.
    target_num_countries = 196
    
    if len(country_list) > target_num_countries:
        print(f"Warning: Found {len(country_list)} countries, keeping the first {target_num_countries}.")
        country_list = country_list[:target_num_countries]
    elif len(country_list) < target_num_countries:
        print(f"Warning: Found {len(country_list)} countries, missing ones will be empty.")
        
    country_to_idx = {c: i for i, c in enumerate(country_list)}
    
    print(f"Loaded {len(records)} records.")
    print(f"Entities: {len(country_list)} countries, 96 products, {num_years} years.")
    
    # 2. Build tensor
    shape = (target_num_countries, target_num_countries, 96, num_years)
    Y = np.zeros(shape, dtype=np.float32)
    
    valid_count = 0
    for exporter, importer, hs2_idx, year_idx, val in records:
        e_idx = country_to_idx.get(exporter, -1)
        i_idx = country_to_idx.get(importer, -1)
        
        if e_idx != -1 and i_idx != -1:
            Y[e_idx, i_idx, hs2_idx, year_idx] = val
            valid_count += 1
            
    # Ensure C-contiguous
    Y = np.ascontiguousarray(Y, dtype=np.float32)

    # 3. Create output directory if it doesn't exist
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    # 4. Save outputs
    print(f"Saving tensor to {args.out}...")
    np.save(args.out, Y)
    
    countries_out = os.path.join(out_dir if out_dir else '', 'wits_countries.txt')
    print(f"Saving country list to {countries_out}...")
    with open(countries_out, 'w', encoding='utf-8') as f:
        for c in country_list:
            f.write(c + '\n')
            
    products_out = os.path.join(out_dir if out_dir else '', 'wits_products.txt')
    print(f"Saving product list to {products_out}...")
    with open(products_out, 'w', encoding='utf-8') as f:
        for i in range(96):
            f.write(hs2_index_to_code(i) + '\n')

    # 5. Heldout mask
    if not args.no_heldout:
        print(f"Generating heldout mask (seed={args.seed}, frac={args.heldout_frac})...")
        heldout_dir = os.path.dirname(args.heldout_out)
        if heldout_dir:
            os.makedirs(heldout_dir, exist_ok=True)
            
        heldout_mask = generate_heldout_mask(Y.shape, args.seed, args.heldout_frac)
        print(f"Saving heldout mask to {args.heldout_out}...")
        np.save(args.heldout_out, heldout_mask)

    # 6. Summary stats
    nnz = np.count_nonzero(Y)
    sparsity = 1.0 - (nnz / Y.size)
    print("\n--- Summary Statistics ---")
    print(f"Shape: {Y.shape}")
    print(f"Total elements: {Y.size}")
    print(f"Non-zero elements: {nnz}")
    print(f"Sparsity: {sparsity:.4f}")
    print(f"Min value: {np.min(Y):.4f}")
    print(f"Max value: {np.max(Y):.4f}")
    print(f"Mean value: {np.mean(Y):.4f}")
    print("--------------------------")


if __name__ == '__main__':
    main()
