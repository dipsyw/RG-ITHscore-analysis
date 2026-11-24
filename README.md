
# RG-ITHscore analysis code

This repository contains the custom data-analysis code described in the final manuscript. It intentionally excludes figure-only scripts, demonstration or synthetic-data code, manuscript builders, local batch monitors, and superseded analyses.

## Modules

1. `01_molecular_subtyping`: NMF programme discovery, external projection, ECM/IR reference clustering, and survival analyses.
2. `02_model_development`: habitat-number assessment, frozen deep-encoder feature extraction, 12-algorithm molecular-guided feature selection, and elastic-net Cox modelling.
3. `03_external_validation`: fixed-score survival analysis, IPCW time-dependent AUC and operating metrics, calibration, and decision-curve analysis.
4. `04_bulk_transcriptomics`: limma differential expression, preranked GSEA, GSVA, and tumour-microenvironment association statistics.
5. `05_single_cell`: Seurat/Harmony processing, cell-type composition, within-cell-type differential expression, Hallmark GSEA, and patient-level module-score comparisons.
6. `06_pathomics_validation`: CellViT-output feature extraction, 52-descriptor patch aggregation into 734 patient-level features, six-phenotype composition, and group/correlation statistics.

Each module contains a `packages.txt` file listing its R or Python dependencies. Scripts use command-line paths and do not contain workstation-specific absolute paths. Run a script with `--help` (Python) or without arguments (R) to see its required inputs.

## Reproducibility boundary

Institutional clinical data, CT/WSI images, single-cell count matrices, and identifier-linkage tables are not distributed. The code therefore starts from authorized, de-identified analysis inputs. Third-party model inference (3D-RPNUNet/3D U-Net, IGNITE nnU-Net and CellViT/CellViT++) and licensed resources such as CIBERSORT must be obtained from their original providers; this repository contains the downstream custom analyses reported in the manuscript.

Repository: <https://github.com/75g8gf2df7-tech/RG-ITHscore-analysis>
