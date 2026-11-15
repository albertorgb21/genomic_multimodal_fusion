import os
import glob
import subprocess
import zipfile
import shutil
import time
import pandas as pd
from pathlib import Path
from Bio import SeqIO, Entrez
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR      = Path("C:\\Users\\rodri\\Desktop\\MASTER\\EMMBIOME\\Grecia\\Courses\\Bioinformatics\\Project\\genomic_multimodal_fusion\\data")
RAW_DIR       = DATA_DIR / "raw"
FASTA_DIR     = DATA_DIR / "fasta"
PROCESSED_DIR = DATA_DIR / "processed"

DATA_VIRUSES_DIR = Path("C:\\Users\\rodri\\Desktop\\MASTER\\EMMBIOME\\Grecia\\Courses\\Bioinformatics\\Project\\genomic_multimodal_fusion\\Data_Viruses\\Data_Viruses.xlsx")

for d in [RAW_DIR, FASTA_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Taxonomy ID map ──────────────────────────────────────────────
VIRUS_TAXONOMY = {
    "Corona"            : (694009,   0),
    "Flu"               : (197911,   1),
    "Norwalk"           : (142786,   2),
    "Coxsackievirus"    : (12059,    3),
    "Crimean-Congo"     : (3052518,  4),
    "Cytomegalovirus"   : (10358,    5),
    "Ebola"             : (186538,   6),
    "Encephalitis"      : (11072,    7),
    "Hepatitis B"       : (10407,    8),
    "Hepatitis C"       : (11103,    9),
    "Varicella"         : (10335,    10),
    "HIV"               : (11676,    11),
    "Lassa"             : (11620,    12),
    "Marburg"           : (11269,    13),
    "Mastadenovirus"    : (10509,    14),
    "Measles"           : (11234,    15),
    "Metapneumovirus"   : (162145,   16),
    "Mumps"             : (11161,    17),
    "Papilloma"         : (10566,    18),
    "Parainfluenza"     : (11552,    19),
    "Poliovirus"        : (12110,    20),
    "Poxmonkey"         : (10244,    21),
    "Respirovirus"      : (11552,    22),
    "Rhinovirus"        : (12110,    23),
    "Rotavirus"         : (12912,    24),
    "RSV"               : (11250,    25),
    "Rubella"           : (11041,    26),
    "Zika"              : (64320,    27),
    "Yellow Fever"      : (11089,    28),
    "Dengue"            : (12637,    29),
}

def clean_accession(acc):
    """Remove quotes and extra characters from accession numbers"""
    if isinstance(acc, str):
        return acc.strip("'\"")
    return str(acc)

# Set Entrez email
Entrez.email = "rodriguezgbalberto@gmail.com"

# Thread-safe lock for Entrez requests (NCBI prefers <3 requests/sec)
entrez_lock = threading.Lock()
time_between_requests = 0.5  # seconds

def download_sequence(accession, virus, virus_fasta_dir, max_retries=3):
    """Download a single sequence with retry logic and rate limiting"""
    fasta_path = virus_fasta_dir / f"{accession}.fasta"
    
    # Skip if file already exists
    if fasta_path.exists():
        return accession, "exists", None
    
    for attempt in range(max_retries):
        try:
            # Rate limiting: ensure minimum time between NCBI requests
            with entrez_lock:
                time.sleep(time_between_requests)
                handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
                record = SeqIO.read(handle, "fasta")
                handle.close()
            
            # Save the sequence
            SeqIO.write(record, fasta_path, "fasta")
            return accession, "downloaded", None
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff
                time.sleep(wait_time)
            else:
                return accession, "failed", str(e)
    
    return accession, "failed", "Max retries exceeded"

# Read all virus data
xls = pd.ExcelFile(DATA_VIRUSES_DIR)
print(f"Available sheets: {xls.sheet_names}")

# Process each virus
for virus in VIRUS_TAXONOMY:
    # Find the sheet name case-insensitively
    sheet_name = next((s for s in xls.sheet_names if virus.lower() in s.lower()), virus)
    
    try:
        virus_df = pd.read_excel(DATA_VIRUSES_DIR, sheet_name=sheet_name)
        print(f"\n{'='*60}")
        print(f"Processing {virus} sheet")
        print(f"Shape: {virus_df.shape}")
        print(f"Column: {virus_df.columns[0]}")
        
        # Clean accession numbers
        virus_df['accession_clean'] = virus_df.iloc[:, 0].apply(clean_accession)
        
        # Filter out empty/invalid accessions
        valid_accessions = virus_df['accession_clean'].dropna()
        valid_accessions = valid_accessions[valid_accessions.str.len() > 0]
        
        print(f"Valid accessions to download: {len(valid_accessions)}")
        
        # Create virus folder
        virus_fasta_dir = FASTA_DIR / virus
        virus_fasta_dir.mkdir(parents=True, exist_ok=True)
        
        # Check how many already exist
        existing_files = len(list(virus_fasta_dir.glob("*.fasta")))
        print(f"Files already present: {existing_files}")
        
        # Download with parallel execution (4 workers for rate limiting)
        results = {"downloaded": 0, "exists": 0, "failed": 0}
        failed_accessions = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(download_sequence, acc, virus, virus_fasta_dir): acc 
                for acc in valid_accessions
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {virus}"):
                acc, status, error = future.result()
                results[status] += 1
                
                if status == "failed":
                    failed_accessions.append((acc, error))
        
        # Print summary
        print(f"\nDownload Summary for {virus}:")
        print(f"  ✓ Downloaded: {results['downloaded']}")
        print(f"  ↻ Already existed: {results['exists']}")
        print(f"  ✗ Failed: {results['failed']}")
        
        if failed_accessions:
            pd.DataFrame(failed_accessions, columns=["accession", "error"]).to_csv(DATA_DIR / "failed_accesions.csv", index=False)
            print(f"\nFailed accessions:")
            for acc, error in failed_accessions[:5]:  # Show first 5
                print(f"  - {acc}: {error}")
            if len(failed_accessions) > 5:
                print(f"  ... and {len(failed_accessions) - 5} more")
    
    except Exception as e:
        print(f"Error processing {virus}: {e}")
        continue

print("\n" + "="*60)
print("All downloads completed!")