"""
data_loader.py

Downloads the genomic sequences listed in an Excel workbook from NCBI.

Input : an Excel workbook with one sheet per virus; the first column of each
        sheet holds the NCBI accession numbers.
Output: DATA_DIR/fasta/<virus>/<accession>.fasta  (one file per sequence)
        DATA_DIR/failed_accessions.csv            (only if some downloads failed)

Already-downloaded files are skipped, so the script can be stopped and resumed.
Edit the configuration block below before running.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from Bio import SeqIO, Entrez
from tqdm import tqdm

# ── Configuration (edit these) ───────────────────────────────────
DATA_DIR         = Path("##DATA_DIR##")           # e.g. Path("../data")
DATA_VIRUSES_DIR = Path("##DATA_VIRUSES_DIR##")   # path to the Excel workbook with the accession lists
Entrez.email     = "##YOUR_EMAIL##"               # NCBI asks for a contact e-mail

if any("##" in str(v) for v in (DATA_DIR, DATA_VIRUSES_DIR, Entrez.email)):
    raise SystemExit("Edit the configuration block at the top of data_loader.py first.")

FASTA_DIR = DATA_DIR / "fasta"
FASTA_DIR.mkdir(parents=True, exist_ok=True)

# ── Viruses (30 classes) ─────────────────────────────────────────
# Each name is used as the output folder and to find the matching sheet in the
# workbook (case-insensitive substring match on the sheet name).
VIRUSES = [
    "Corona", "Flu", "Norwalk", "Coxsackievirus", "Crimean-Congo",
    "Cytomegalovirus", "Ebola", "Encephalitis", "Hepatitis B", "Hepatitis C",
    "Varicella", "HIV", "Lassa", "Marburg", "Mastadenovirus",
    "Measles", "Metapneumovirus", "Mumps", "Papilloma", "Parainfluenza",
    "Poliovirus", "Poxmonkey", "Respirovirus", "Rhinovirus", "Rotavirus",
    "RSV", "Rubella", "Zika", "Yellow Fever", "Dengue",
]

# NCBI allows ~3 requests/second without an API key; requests are serialised
# through a lock and spaced by this many seconds.
TIME_BETWEEN_REQUESTS = 0.5
entrez_lock = threading.Lock()


def clean_accession(acc):
    """Remove quotes and extra characters from accession numbers."""
    if isinstance(acc, str):
        return acc.strip("'\"")
    return str(acc)


def download_sequence(accession, virus_fasta_dir, max_retries=3):
    """Download one sequence with retries (exponential back-off) and rate limiting.

    Returns (accession, status, error) where status is 'downloaded', 'exists' or 'failed'.
    """
    fasta_path = virus_fasta_dir / f"{accession}.fasta"

    if fasta_path.exists():
        return accession, "exists", None

    for attempt in range(max_retries):
        try:
            with entrez_lock:
                time.sleep(TIME_BETWEEN_REQUESTS)
                handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
                record = SeqIO.read(handle, "fasta")
                handle.close()

            SeqIO.write(record, fasta_path, "fasta")
            return accession, "downloaded", None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return accession, "failed", str(e)

    return accession, "failed", "Max retries exceeded"


def main():
    xls = pd.ExcelFile(DATA_VIRUSES_DIR)
    print(f"Available sheets: {xls.sheet_names}")

    all_failed = []   # (virus, accession, error) across all viruses

    for virus in VIRUSES:
        sheet_name = next((s for s in xls.sheet_names if virus.lower() in s.lower()), virus)

        try:
            virus_df = pd.read_excel(DATA_VIRUSES_DIR, sheet_name=sheet_name)
            print(f"\n{'=' * 60}")
            print(f"Processing {virus} sheet")
            print(f"Shape: {virus_df.shape}")
            print(f"Column: {virus_df.columns[0]}")

            accessions = virus_df.iloc[:, 0].apply(clean_accession).dropna()
            accessions = accessions[accessions.str.len() > 0]
            print(f"Valid accessions to download: {len(accessions)}")

            virus_fasta_dir = FASTA_DIR / virus
            virus_fasta_dir.mkdir(parents=True, exist_ok=True)
            print(f"Files already present: {len(list(virus_fasta_dir.glob('*.fasta')))}")

            results = {"downloaded": 0, "exists": 0, "failed": 0}
            failed = []

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(download_sequence, acc, virus_fasta_dir): acc
                    for acc in accessions
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {virus}"):
                    acc, status, error = future.result()
                    results[status] += 1
                    if status == "failed":
                        failed.append((acc, error))

            print(f"\nDownload summary for {virus}:")
            print(f"  ✓ Downloaded: {results['downloaded']}")
            print(f"  ↻ Already existed: {results['exists']}")
            print(f"  ✗ Failed: {results['failed']}")

            if failed:
                all_failed.extend((virus, acc, err) for acc, err in failed)
                print("\nFailed accessions:")
                for acc, error in failed[:5]:
                    print(f"  - {acc}: {error}")
                if len(failed) > 5:
                    print(f"  ... and {len(failed) - 5} more")

        except Exception as e:
            print(f"Error processing {virus}: {e}")
            continue

    if all_failed:
        failed_path = DATA_DIR / "failed_accessions.csv"
        pd.DataFrame(all_failed, columns=["virus", "accession", "error"]).to_csv(failed_path, index=False)
        print(f"\n{len(all_failed)} downloads failed in total; see {failed_path}")

    print("\n" + "=" * 60)
    print("All downloads completed!")


if __name__ == "__main__":
    main()
