# Genomic Multimodal Fusion

Classification of 30 virus types from whole genomic sequences by fusing two complementary representations of each sequence:

- **Stream A — CPNR · GASF · CNN.** Each sequence is mapped to a codon-based signal (CPNR), turned into a 128×128 Gramian Angular Summation Field image and encoded by a small CNN into a 256-dim embedding.
- **Stream B — PRCFX-DT (PageRank).** A 64×64 codon-transition Markov graph is built per sequence and its PageRank centrality gives a 64-dim feature vector. A Decision Tree on this vector reproduces the method of Khodaei et al. (2025).
- **Fusion.** The two vectors are standardised and concatenated (320-dim) and classified by either a standard MLP or a Mixture-of-Experts (MoE) head.

This project was developed as part of the Research Methodology in Biomedical Engineering course at the University of Patras, exploring how signal processing, graph-based representations and deep learning can be combined for genomic sequence classification. The central research question is whether complementary representations of viral genomes can improve classification performance compared with individual feature extraction approaches.

## Results (test set, 30 classes)

Test set = 20 % of the data, evaluated once with the checkpoint selected on the validation set.

| System | Accuracy | Macro F1 |
|---|---|---|
| Stream A — CPNR-GASF-CNN | 0.9753 | 0.8790 |
| Stream B — PRCFX-DT (PageRank + Decision Tree) | 0.9976 | 0.9829 |
| Fusion — Standard MLP (A+B) | 0.9967 | 0.9585 |
| Fusion — MoE (E=4) | 0.9957 | 0.9556 |

Stream B achieves the best overall performance, with 99.76% accuracy and a macro F1-score of 0.9829, outperforming both Stream A and the fusion architectures. Although the CNN-based Stream A provides a complementary representation, neither the standard MLP nor the four-expert MoE improves upon the PageRank-based classifier, suggesting that the graph-derived features capture highly discriminative information for this dataset.

## Data

173,182 sequences after cleaning (non-ACGT characters removed, sequences shorter than 500 nt dropped), split 70 % train / 10 % validation / 20 % test (stratified).

The dataset was assembled from viral genome records indexed in the NCBI GenBank database. The accession lists are maintained in an Excel workbook, with one worksheet per virus class, and are used to retrieve the corresponding nucleotide sequences through NCBI Entrez. The accession workbook is not included in this repository; users must obtain the original workbook from the project author before running the data acquisition pipeline. Once retrieved, the sequences can be downloaded directly from NCBI using their accession identifiers.

The data is not stored in the repository. It is downloaded from NCBI and written to a `data/` folder at the repository root.

## Repository structure

```
genomic_multimodal_fusion/
├── README.md
├── requirements.txt
├── data/                          # created by the scripts, not versioned
│   ├── raw/
│   ├── fasta/                     # one folder per virus
│   └── processed/
│       └── all_viruses_sequences.csv
└── src/
    ├── data_loader.py             # 1. download sequences from NCBI
    ├── process_to_csv.py          # 2. merge FASTA files into one CSV
    ├── StreamA_cpnr_GAF_CNN_wholeseq_VAL.ipynb
    ├── StreamB_PRCFXDT_PageRank_DT.ipynb
    ├── Fusion_Standard_VAL.ipynb
    ├── Fusion_MoE_VAL.ipynb
    └── preprocessing.py           # optional: windowed train/test CSVs (not used by the notebooks)
```

## Installation

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# CPU build of PyTorch (smaller download)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The notebooks were run with Python 3.13 on CPU.

## How to run

Run everything from inside `src/`.

1. **Download the sequences.** Edit the paths and your e-mail at the top of `data_loader.py`, then run `python data_loader.py`. This fetches every accession from NCBI Entrez and writes one FASTA file per sequence under `data/fasta/<virus>/`. It is rate-limited and takes a long time (on the order of a day for ~173k sequences). Already-downloaded files are skipped, so it can be resumed.
2. **Build the CSV.** Edit the paths in `process_to_csv.py`, then run `python process_to_csv.py`. This creates `data/processed/all_viruses_sequences.csv` with the columns `accession, sequence, length, virus, label`.
3. **Run the notebooks in this order:**

   | Order | Notebook | Needs | Produces |
   |---|---|---|---|
   | 1 | `StreamA_cpnr_GAF_CNN_wholeseq_VAL.ipynb` | the CSV | `output_streamA_val/` (CNN, train/val/test embeddings, summary) |
   | 2 | `StreamB_PRCFXDT_PageRank_DT.ipynb` | the CSV | `output_streamB/` (PageRank features, DT model, summary) |
   | 3 | `Fusion_Standard_VAL.ipynb` | outputs of 1 and 2 | `output_fusion_standard_val/` |
   | 4 | `Fusion_MoE_VAL.ipynb` | outputs of 1, 2 and 3 | `output_fusion_moe_val/` |

   Notebooks 1 and 2 are independent of each other. Notebook 4 reads the summary written by notebook 3 for its ablation table.

## Methodology notes

- **Split.** Step 1 splits all sequences 80 / 20 (train+val / test, `random_state=42`, stratified). Step 2 splits the 80 % into 87.5 / 12.5 (`random_state=43`, stratified), giving about 70 / 10 / 20 %.
- **Model selection.** The learning-rate scheduler, early stopping (patience 8) and best-checkpoint selection use validation macro F1 only. The test set is evaluated once at the end.
- **Row alignment between streams.** The fusion notebooks rebuild the same splits and slice Stream B's `X_train_sequences.npy` with the same indices to obtain its validation features. They assert at runtime that the labels agree, so a mismatch fails loudly.
- **Class imbalance.** A `WeightedRandomSampler` is used on the training loader only (Stream A and the fusion heads). Stream B uses `class_weight='balanced'`.

## References

- Khodaei A. et al. *PRCFX-DT: a new graph-based approach for feature selection and classification of genomic sequences.* BMC Bioinformatics 26, 159 (2025). https://doi.org/10.1186/s12859-025-06183-4
- Chen D., Wang J., Yan M., Bao F. S. A Complex Prime Numerical Representation of Amino Acids for Protein Function Comparison. Journal of Computational Biology 23(8), 669–677 (2016). https://doi.org/10.1089/cmb.2015.0178
- Wang Z., Oates T. Imaging Time-Series to Improve Classification and Imputation. Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI), 3939–3944 (2015). https://arxiv.org/abs/1506.00327
- Jacobs R. A., Jordan M. I., Nowlan S. J., Hinton G. E. Adaptive Mixtures of Local Experts. Neural Computation 3(1), 79–87 (1991). https://doi.org/10.1162/neco.1991.3.1.79


> TODO: add the reference for the CPNR encoding and the MoE reference you use, after checking them.

## License

> TODO: choose a license (for example MIT) and add a `LICENSE` file.
