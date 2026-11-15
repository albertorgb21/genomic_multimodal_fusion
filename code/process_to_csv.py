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

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR      = Path("##DATA_DIR##")
RAW_DIR       = DATA_DIR / "raw"
FASTA_DIR     = DATA_DIR / "fasta"
PROCESSED_DIR = DATA_DIR / "processed"

# ── Taxonomy ID map ──────────────────────────────────────────────
# NCBI Taxonomy IDs for the 30 viruses in Table 1 of the paper.
# You can verify any ID at: https://www.ncbi.nlm.nih.gov/taxonomy
VIRUS_TAXONOMY = {
    # Virus name        : (taxonomy_id, label_int)
    "corona"            : (694009,   0),   # Betacoronavirus
    "flu"               : (197911,   1),   # Alphainfluenzavirus
    "norwalk"           : (142786,   2),   # Norwalk virus
    "coxsackievirus"    : (12059,    3),
    "crimean-congo"     : (3052518,  4),   # CCHF virus
    "cytomegalovirus"   : (10358,    5),   # Human cytomegalovirus
    "ebola"             : (186538,   6),   # Zaire ebolavirus
    "encephalitis"      : (11072,    7),   # Eastern equine encephalitis
    "hepatitis b"       : (10407,    8),
    "hepatitis c"       : (11103,    9),
    "varicella"         : (10335,    10),  # Varicella-zoster virus
    "hiv"               : (11676,    11),  # HIV-1
    "lassa"             : (11620,    12),
    "marburg"           : (11269,    13),
    "mastadenovirus"    : (10509,    14),
    "measles"           : (11234,    15),
    "metapneumovirus"   : (162145,   16),  # Human metapneumovirus
    "mumps"             : (11161,    17),
    "papilloma"         : (10566,    18),  # Human papillomavirus
    "parainfluenza"     : (11552,    19),
    "poliovirus"        : (12110,    20),
    "poxmonkey"         : (10244,    21),  # Monkeypox virus
    "respirovirus"      : (11552,    22),  # Sendai virus family
    "rhinovirus"        : (12110,    23),  # Enterovirus (rhinovirus)
    "rotavirus"         : (10912,    24),
    "rsv"               : (11250,    25),  # Respiratory syncytial virus
    "rubella"           : (11041,    26),
    "zika"              : (64320,    27),
    "yellow fever"      : (11089,    28),
    "dengue"            : (12637,    29),
}


def process_fasta_to_csv():
    """
    Process all FASTA files from virus subfolders and combine into a single CSV.
    Creates a DataFrame with columns: accession, sequence, length, virus, label
    """
    all_records = []
    
    # Get all virus subfolders in FASTA_DIR
    virus_folders = [f for f in FASTA_DIR.iterdir() if f.is_dir()]
    
    print(f"Found {len(virus_folders)} virus folders: {[f.name for f in virus_folders]}")
    
    for virus_folder in tqdm(virus_folders, desc="Processing virus folders"):
        virus_name = virus_folder.name.lower()

        # Get the label from VIRUS_TAXONOMY if available
        label = VIRUS_TAXONOMY.get(virus_name, -1)  # Default to -1 if not found
        if label != -1:
            label = label[1]    # Labels are stored from the Taxonomy list as (tax_id, label)
        
        # Find all FASTA files in this virus folder
        fasta_files = list(virus_folder.glob("*.fasta")) + list(virus_folder.glob("*.fa"))
        
        print(f"Processing {virus_name}: {len(fasta_files)} FASTA files")
        
        for fasta_file in tqdm(fasta_files, desc=f"Reading {virus_name} sequences", leave=False):
            try:
                # Parse the FASTA file
                for record in SeqIO.parse(fasta_file, "fasta"):
                    # Clean the sequence (keep only ACGT)
                    seq_str = str(record.seq).upper()
                    clean_seq = "".join(c for c in seq_str if c in "ACGT")
                    
                    # Skip very short sequences
                    if len(clean_seq) < 100:
                        continue
                    
                    # Create record
                    record_data = {
                        "accession": record.id,
                        "sequence": clean_seq,
                        "length": len(clean_seq),
                        "virus": virus_name,
                        "label": label
                    }
                    
                    all_records.append(record_data)
                    
            except Exception as e:
                print(f"Error processing {fasta_file}: {e}")
                continue
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Save to CSV
    output_path = PROCESSED_DIR / "all_viruses_sequences.csv"
    df.to_csv(output_path, index=False)
    
    print("✓ Processing complete!")
    print(f"Total sequences: {len(df)}")
    print(f"Saved to: {output_path}")
    
    # Show summary by virus
    if len(df) > 0:
        print("\n── Dataset Summary ─────────────────────────────")
        summary = df.groupby("virus").agg(
            count=("accession", "count"),
            mean_length=("length", "mean"),
            min_length=("length", "min"),
            max_length=("length", "max"),
        ).round(0)
        print(summary.to_string())
    
    return df

# Run the processing
df_all = process_fasta_to_csv()

# Show sample of the final dataset
print("\n── Sample of Final Dataset ─────────────────────")
print(df_all.head(10))
