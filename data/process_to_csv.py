"""
process_to_csv.py

Combines the per-sequence FASTA files written by data_loader.py into one CSV.

Input : DATA_DIR/fasta/<virus>/*.fasta
Output: DATA_DIR/processed/all_viruses_sequences.csv
        with columns: accession, sequence, length, virus, label

Only unambiguous nucleotides (A, C, G, T) are kept, and sequences shorter than
100 nt after cleaning are skipped. Edit DATA_DIR before running.
"""

from pathlib import Path

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# ── Configuration (edit this) ────────────────────────────────────
DATA_DIR = Path("##DATA_DIR##")   # same folder as in data_loader.py

if "##" in str(DATA_DIR):
    raise SystemExit("Edit DATA_DIR at the top of process_to_csv.py first.")

FASTA_DIR     = DATA_DIR / "fasta"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_LENGTH = 100   # nt, after removing non-ACGT characters

# ── Class labels (order of the 30 viruses in Table 1 of the paper) ──
# Keys are the lower-cased folder names created by data_loader.py.
VIRUS_LABELS = {
    "corona": 0, "flu": 1, "norwalk": 2, "coxsackievirus": 3, "crimean-congo": 4,
    "cytomegalovirus": 5, "ebola": 6, "encephalitis": 7, "hepatitis b": 8,
    "hepatitis c": 9, "varicella": 10, "hiv": 11, "lassa": 12, "marburg": 13,
    "mastadenovirus": 14, "measles": 15, "metapneumovirus": 16, "mumps": 17,
    "papilloma": 18, "parainfluenza": 19, "poliovirus": 20, "poxmonkey": 21,
    "respirovirus": 22, "rhinovirus": 23, "rotavirus": 24, "rsv": 25,
    "rubella": 26, "zika": 27, "yellow fever": 28, "dengue": 29,
}


def process_fasta_to_csv():
    """Read all FASTA files under FASTA_DIR and write a single CSV."""
    all_records = []

    virus_folders = sorted(f for f in FASTA_DIR.iterdir() if f.is_dir())
    print(f"Found {len(virus_folders)} virus folders: {[f.name for f in virus_folders]}")

    for virus_folder in tqdm(virus_folders, desc="Processing virus folders"):
        virus_name = virus_folder.name.lower()

        if virus_name not in VIRUS_LABELS:
            print(f"WARNING: '{virus_folder.name}' is not in VIRUS_LABELS; skipping this folder.")
            continue
        label = VIRUS_LABELS[virus_name]

        fasta_files = list(virus_folder.glob("*.fasta")) + list(virus_folder.glob("*.fa"))
        print(f"Processing {virus_name}: {len(fasta_files)} FASTA files")

        for fasta_file in tqdm(fasta_files, desc=f"Reading {virus_name} sequences", leave=False):
            try:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    clean_seq = "".join(c for c in str(record.seq).upper() if c in "ACGT")

                    if len(clean_seq) < MIN_LENGTH:
                        continue

                    all_records.append({
                        "accession": record.id,
                        "sequence": clean_seq,
                        "length": len(clean_seq),
                        "virus": virus_name,
                        "label": label,
                    })

            except Exception as e:
                print(f"Error processing {fasta_file}: {e}")
                continue

    df = pd.DataFrame(all_records)

    output_path = PROCESSED_DIR / "all_viruses_sequences.csv"
    df.to_csv(output_path, index=False)

    print("✓ Processing complete!")
    print(f"Total sequences: {len(df)}")
    print(f"Saved to: {output_path}")

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


if __name__ == "__main__":
    df_all = process_fasta_to_csv()

    print("\n── Sample of Final Dataset ─────────────────────")
    print(df_all.head(10))
