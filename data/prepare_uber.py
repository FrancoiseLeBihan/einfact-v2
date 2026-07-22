import argparse
import os
import sys
import gzip
import urllib.request
import hashlib
from datetime import datetime, timedelta
import csv
import numpy as np

def download_file(url, out_path):
    """Downloads a file from url to out_path, skipping if it already exists."""
    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {out_path} ...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    urllib.request.urlretrieve(url, out_path)

def get_sha256(path):
    """Computes the SHA-256 hash of a file."""
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(path, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

def to_grid(val, min_val, max_val, num_bins):
    """Maps a real value to a uniform grid bin index [0, num_bins - 1]."""
    if max_val == min_val:
        return 0
    bin_idx = int((val - min_val) / (max_val - min_val) * num_bins)
    return min(max(bin_idx, 0), num_bins - 1)

def load_frostt(cache_dir):
    """Downloads and parses the Uber FROSTT tensor."""
    base_url = "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/uber-pickups"
    tns_url = f"{base_url}/uber.tns.gz"
    map_urls = [
        f"{base_url}/mode-1-dates.map.gz",
        f"{base_url}/mode-2-hours.map.gz",
        f"{base_url}/mode-3-latitudes.map.gz",
        f"{base_url}/mode-4-longitudes.map.gz",
    ]
    
    tns_path = os.path.join(cache_dir, "uber.tns.gz")
    download_file(tns_url, tns_path)
    print(f"uber.tns.gz SHA-256: {get_sha256(tns_path)}")
    
    map_paths = []
    for url in map_urls:
        path = os.path.join(cache_dir, os.path.basename(url))
        download_file(url, path)
        print(f"{os.path.basename(path)} SHA-256: {get_sha256(path)}")
        map_paths.append(path)
        
    print("Loading mappings...")
    dates = []
    with gzip.open(map_paths[0], 'rt') as f:
        for line in f:
            dates.append(datetime.strptime(line.strip(), '%Y-%m-%d'))
            
    lats = []
    with gzip.open(map_paths[2], 'rt') as f:
        for line in f:
            lats.append(float(line.strip()))
            
    lons = []
    with gzip.open(map_paths[3], 'rt') as f:
        for line in f:
            lons.append(float(line.strip()))
            
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    earliest_date = min(dates).date()
    start_date = earliest_date - timedelta(days=earliest_date.weekday())
    print(f"Earliest date in dataset: {earliest_date}. Start of week 0: {start_date}.")
    
    date_to_wd = []
    for dt in dates:
        delta = dt.date() - start_date
        w = delta.days // 7
        d = dt.weekday()
        date_to_wd.append((w, d))
        
    lat_to_i = [to_grid(lat, min_lat, max_lat, 100) for lat in lats]
    lon_to_j = [to_grid(lon, min_lon, max_lon, 100) for lon in lons]
    
    Y = np.zeros((27, 7, 24, 100, 100), dtype=np.float32)
    
    print("Parsing tensor and aggregating...")
    with gzip.open(tns_path, 'rt') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            i1, i2, i3, i4, val = parts
            
            date_idx = int(i1) - 1
            hour_idx = int(i2) - 1
            lat_idx = int(i3) - 1
            lon_idx = int(i4) - 1
            v = float(val)
            
            w, d = date_to_wd[date_idx]
            if w < 27:
                h = hour_idx
                i = lat_to_i[lat_idx]
                j = lon_to_j[lon_idx]
                Y[w, d, h, i, j] += v
                
    return Y

def load_csvs(csv_dir):
    """Loads and parses raw Uber FiveThirtyEight CSV files."""
    Y = np.zeros((27, 7, 24, 100, 100), dtype=np.float32)
    files = [f for f in os.listdir(csv_dir) if f.startswith('uber-raw-data') and f.endswith('.csv')]
    if not files:
        raise ValueError(f"No CSV files found in {csv_dir} matching 'uber-raw-data*.csv'")
    
    print("Pass 1: Finding geographic bounds and earliest date...")
    min_lat, max_lat = float('inf'), float('-inf')
    min_lon, max_lon = float('inf'), float('-inf')
    earliest_date = None
    
    for fname in files:
        with open(os.path.join(csv_dir, fname), 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                dt_str, lat_str, lon_str, _ = row
                try:
                    dt = datetime.strptime(dt_str, '%m/%d/%Y %H:%M:%S')
                except ValueError:
                    dt = datetime.strptime(dt_str, '%m/%d/%Y %H:%M')
                    
                lat, lon = float(lat_str), float(lon_str)
                
                min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)
                min_lon, max_lon = min(min_lon, lon), max(max_lon, lon)
                
                if earliest_date is None or dt.date() < earliest_date:
                    earliest_date = dt.date()
                    
    start_date = earliest_date - timedelta(days=earliest_date.weekday())
    print(f"Earliest date in dataset: {earliest_date}. Start of week 0: {start_date}.")
    
    print("Pass 2: Populating tensor...")
    for fname in files:
        with open(os.path.join(csv_dir, fname), 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                dt_str, lat_str, lon_str, _ = row
                try:
                    dt = datetime.strptime(dt_str, '%m/%d/%Y %H:%M:%S')
                except ValueError:
                    dt = datetime.strptime(dt_str, '%m/%d/%Y %H:%M')
                    
                lat, lon = float(lat_str), float(lon_str)
                
                delta = dt.date() - start_date
                w = delta.days // 7
                d = dt.weekday()
                h = dt.hour
                
                if w < 27:
                    i = to_grid(lat, min_lat, max_lat, 100)
                    j = to_grid(lon, min_lon, max_lon, 100)
                    Y[w, d, h, i, j] += 1
                    
    return Y

def main():
    parser = argparse.ArgumentParser(description="Prepare Uber pickups tensor.")
    parser.add_argument('--out', default='DATA/uber.npy', help="Output path for the tensor")
    parser.add_argument('--heldout-out', default='DATA/uber_heldout.npy', help="Output path for heldout mask")
    parser.add_argument('--seed', type=int, default=29482, help="Seed for the heldout split")
    parser.add_argument('--heldout-frac', type=float, default=0.05, help="Fraction of entries for heldout")
    parser.add_argument('--from-csv', metavar='DIR', help="Load from raw CSVs in DIR instead of FROSTT")
    parser.add_argument('--cache-dir', default='data/.cache', help="Directory to cache downloaded files")
    parser.add_argument('--no-heldout', action='store_true', help="Skip generating heldout mask")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    
    if args.from_csv:
        print(f"Loading data from CSVs in {args.from_csv}...")
        Y = load_csvs(args.from_csv)
    else:
        print("Loading data from FROSTT...")
        Y = load_frostt(args.cache_dir)
        
    print(f"Tensor shape: {Y.shape}")
    print(f"Tensor dtype: {Y.dtype}")
    print(f"Total pickups: {Y.sum():.0f}")
    print(f"Min value: {Y.min():.0f}, Max value: {Y.max():.0f}")
    print(f"Nonzero fraction: {np.count_nonzero(Y) / Y.size:.5f}")
    
    # Ensure C-contiguous
    Y = np.ascontiguousarray(Y)
    np.save(args.out, Y)
    print(f"Saved tensor to {args.out}")
    
    if not args.no_heldout:
        os.makedirs(os.path.dirname(os.path.abspath(args.heldout_out)), exist_ok=True)
        # Deterministic random selection for heldout matching common reproduce/common.py split convention
        rng = np.random.RandomState(args.seed)
        heldout_mask = (rng.rand(*Y.shape) < args.heldout_frac)
        np.save(args.heldout_out, heldout_mask)
        print(f"Saved heldout mask to {args.heldout_out}")
        print(f"Heldout fraction generated: {np.count_nonzero(heldout_mask) / heldout_mask.size:.5f}")

if __name__ == '__main__':
    main()
