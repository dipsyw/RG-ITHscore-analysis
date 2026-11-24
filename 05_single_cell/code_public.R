# Public single-cell RNA-seq analysis for RG-ITHscore
#
# Expected layout:
#   05_single_cell/
#     code_public.R
#     sample_metadata.csv
#     data/<sample_id>/{barcodes.tsv.gz,features.tsv.gz,matrix.mtx.gz}
#
# Run from the 05_single_cell directory with:
#   Rscript code_public.R

suppressPackageStartupMessages({
  library(Seurat)
  library(harmony)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(msigdbr)
  library(fgsea)
})

set.seed(20260613)

project_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
data_dir <- file.path(project_dir, "data")
metadata_file <- file.path(project_dir, "sample_metadata.csv")
output_dir <- file.path(project_dir, "results_public")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

required_files <- c("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")
qc_thresholds <- list(
  nFeature_lower = 500,
  nFeature_upper = 10000,
  nCount_lower = 1000,
  nCount_upper = 100000,
  mitochondrial_upper = 30,
  haemoglobin_upper = 5
)

if (!dir.exists(data_dir)) {
  stop("Missing data directory: ", data_dir)
}
if (!file.exists(metadata_file)) {
  stop("Missing sample metadata: ", metadata_file)
}

sample_metadata <- read_csv(metadata_file, show_col_types = FALSE) %>%
  mutate(
    sample_id = as.character(sample_id),
    RG_ITH_group = factor(RG_ITH_group, levels = c("Low", "High"))
  )

if (!all(c("sample_id", "RG_ITH_group") %in% colnames(sample_metadata))) {
  stop("sample_metadata.csv must contain sample_id and RG_ITH_group columns.")
}
if (anyDuplicated(sample_metadata$sample_id)) {
  stop("sample_id values must be unique.")
}
if (any(is.na(sample_metadata$RG_ITH_group))) {
  stop("RG_ITH_group must contain only Low or High.")
}

sample_dirs <- file.path(data_dir, sample_metadata$sample_id)
names(sample_dirs) <- sample_metadata$sample_id
file_audit <- bind_rows(lapply(names(sample_dirs), function(sample_id) {
  data.frame(
    sample_id = sample_id,
    file = required_files,
    present = file.exists(file.path(sample_dirs[[sample_id]], required_files))
  )
}))
write_csv(file_audit, file.path(output_dir, "input_file_audit.csv"))
if (any(!file_audit$present)) {
  stop("One or more samples are missing required 10X files; see input_file_audit.csv.")
}

infer_gene_column <- function(features_file) {
  features <- read.delim(
    gzfile(features_file), header = FALSE, stringsAsFactors = FALSE,
    quote = "", comment.char = ""
  )
  candidate_columns <- seq_len(min(2L, ncol(features)))
  marker_genes <- c(
    "EPCAM", "KRT19", "PTPRC", "CD3D", "CD3E", "MS4A1", "LYZ",
    "C1QA", "COL1A1", "PECAM1", "MKI67", "MT-CO1", "MT-ND1"
  )
  scores <- vapply(candidate_columns, function(column_index) {
    genes <- features[[column_index]]
    sum(marker_genes %in% genes) + sum(grepl("^MT-", genes)) -
      10 * mean(grepl("^ENSG", genes))
  }, numeric(1))
  candidate_columns[which.max(scores)]
}

gene_columns <- vapply(sample_dirs, function(sample_dir) {
  infer_gene_column(file.path(sample_dir, "features.tsv.gz"))
}, integer(1))
write_csv(
  data.frame(sample_id = names(gene_columns), gene_column = gene_columns),
  file.path(output_dir, "gene_column_audit.csv")
)
if (length(unique(gene_columns)) != 1L) {
  stop("The inferred gene-name column differs among samples; inspect gene_column_audit.csv.")
}

gene_column <- unname(gene_columns[[1]])
objects <- lapply(names(sample_dirs), function(sample_id) {
  counts <- Read10X(data.dir = sample_dirs[[sample_id]], gene.column = gene_column)
  if (is.list(counts)) {
    if (!"Gene Expression" %in% names(counts)) {
      stop("No Gene Expression matrix was found for sample ", sample_id)
    }
    counts <- counts[["Gene Expression"]]
  }
  CreateSeuratObject(
    counts = counts,
    project = sample_id,
    min.cells = 3,
    min.features = 200
  )
})
names(objects) <- names(sample_dirs)

seu <- merge(
  x = objects[[1]],
  y = objects[-1],
  add.cell.ids = names(objects),
  project = "RG_ITHscore_single_cell"
)

seu[["pMT"]] <- PercentageFeatureSet(seu, pattern = "^MT-")
haemoglobin_genes <- intersect(
  c("HBA1", "HBA2", "HBB", "HBD", "HBE1", "HBG1", "HBG2", "HBM", "HBQ1", "HBZ"),
  rownames(seu)
)
seu[["pHB"]] <- PercentageFeatureSet(seu, features = haemoglobin_genes)

initial_cells <- ncol(seu)
qc_by_sample_before <- seu@meta.data %>% count(orig.ident, name = "cells_before_qc")

seu <- subset(
  seu,
  subset = nFeature_RNA > qc_thresholds$nFeature_lower &
    nFeature_RNA < qc_thresholds$nFeature_upper &
    nCount_RNA > qc_thresholds$nCount_lower &
    nCount_RNA < qc_thresholds$nCount_upper &
    pMT < qc_thresholds$mitochondrial_upper &
    pHB < qc_thresholds$haemoglobin_upper
)

retained_cells <- ncol(seu)
qc_by_sample_after <- seu@meta.data %>% count(orig.ident, name = "cells_after_qc")
qc_summary <- qc_by_sample_before %>%
  full_join(qc_by_sample_after, by = "orig.ident") %>%
  mutate(retention_fraction = cells_after_qc / cells_before_qc)
write_csv(qc_summary, file.path(output_dir, "qc_counts_by_sample.csv"))
write_csv(
  data.frame(initial_cells = initial_cells, retained_cells = retained_cells),
  file.path(output_dir, "qc_counts_total.csv")
)
message("QC retained ", format(retained_cells, big.mark = ","), " of ",
        format(initial_cells, big.mark = ","), " cells.")

seu <- NormalizeData(seu, verbose = FALSE)
seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 3000)
s_genes <- intersect(cc.genes.updated.2019$s.genes, rownames(seu))
g2m_genes <- intersect(cc.genes.updated.2019$g2m.genes, rownames(seu))
seu <- CellCycleScoring(seu, s.features = s_genes, g2m.features = g2m_genes)
seu <- ScaleData(
  seu,
  features = VariableFeatures(seu),
  vars.to.regress = c("pMT", "S.Score", "G2M.Score"),
  verbose = FALSE
)
seu <- RunPCA(seu, npcs = 50, verbose = FALSE)
seu <- RunHarmony(seu, group.by.vars = "orig.ident", plot_convergence = FALSE)
seu <- FindNeighbors(seu, reduction = "harmony", dims = 1:20, verbose = FALSE)
seu <- FindClusters(seu, resolution = 1.0, random.seed = 20260613, verbose = FALSE)
seu <- RunUMAP(
  seu, reduction = "harmony", dims = 1:20,
  seed.use = 20260613, verbose = FALSE
)

cluster_column <- if ("RNA_snn_res.1" %in% colnames(seu@meta.data)) {
  "RNA_snn_res.1"
} else {
  "seurat_clusters"
}

main_cell_type <- c(
  "0" = "CD4 T cells", "1" = "CD8 T cells", "2" = "Macrophages",
  "3" = "NK cells", "4" = "Epithelial cells", "5" = "Epithelial cells",
  "6" = "Other immune cells", "7" = "Dendritic cells",
  "8" = "Other immune cells", "9" = "Monocytes", "10" = "Macrophages",
  "11" = "CD4 T cells", "12" = "Monocytes", "13" = "Macrophages",
  "14" = "CD8 T cells", "15" = "Endothelial cells",
  "16" = "Epithelial cells", "17" = "Epithelial cells",
  "18" = "Dendritic cells", "19" = "Fibroblasts", "20" = "Macrophages",
  "21" = "Epithelial cells", "22" = "CD4 T cells",
  "23" = "Epithelial cells", "24" = "Dendritic cells",
  "25" = "Other immune cells", "26" = "Epithelial cells",
  "27" = "Other immune cells", "28" = "Macrophages",
  "29" = "Epithelial cells", "30" = "Epithelial cells",
  "31" = "Epithelial cells"
)

cluster_detail <- c(
  "0" = "CD4/memory T cells", "1" = "CD8 cytotoxic T cells",
  "2" = "SPP1+ macrophages", "3" = "NK cells",
  "4" = "Secretory epithelial cells", "5" = "AT2-like/LUAD epithelial cells",
  "6" = "Mast cells", "7" = "CCL17+ dendritic cells", "8" = "B cells",
  "9" = "Inflammatory monocytes", "10" = "Alveolar macrophages",
  "11" = "Treg-like CD4 T cells", "12" = "Inflammatory monocytes",
  "13" = "Cycling SPP1+ macrophages", "14" = "Cytotoxic T cells",
  "15" = "Endothelial cells", "16" = "Ciliated epithelial cells",
  "17" = "Secretory epithelial cells", "18" = "cDC2/Langerhans-like DCs",
  "19" = "Fibroblasts", "20" = "Macrophages",
  "21" = "AT2 epithelial cells", "22" = "CD4 T cells",
  "23" = "Proliferative/secretory epithelial cells",
  "24" = "Mature/LAMP3+ dendritic cells", "25" = "Plasma cells",
  "26" = "AT1 epithelial cells", "27" = "Mast cells",
  "28" = "APOE/SPP1+ macrophages", "29" = "LUAD-like epithelial cells",
  "30" = "AT2 epithelial cells",
  "31" = "Inflammatory/stressed epithelial cells"
)

cluster_id <- as.character(seu@meta.data[[cluster_column]])
if (!all(unique(cluster_id) %in% names(main_cell_type))) {
  stop("The public cluster annotation does not cover every detected cluster.")
}
seu$main_cell_type <- unname(main_cell_type[cluster_id])
seu$cell_type_detail <- unname(cluster_detail[cluster_id])
seu$RG_ITH_group <- sample_metadata$RG_ITH_group[
  match(as.character(seu$orig.ident), sample_metadata$sample_id)
]
if (any(is.na(seu$RG_ITH_group))) {
  stop("One or more cells could not be linked to sample_metadata.csv.")
}

cell_type_order <- c(
  "Epithelial cells", "Endothelial cells", "Fibroblasts", "Macrophages",
  "Monocytes", "Dendritic cells", "CD4 T cells", "CD8 T cells", "NK cells",
  "Other immune cells"
)
seu$main_cell_type <- factor(seu$main_cell_type, levels = cell_type_order)
seu$RG_ITH_group <- factor(seu$RG_ITH_group, levels = c("Low", "High"))

annotation_summary <- seu@meta.data %>%
  count(orig.ident, RG_ITH_group, main_cell_type, name = "n_cells")
write_csv(annotation_summary, file.path(output_dir, "cell_counts_by_sample_and_type.csv"))
saveRDS(seu, file.path(output_dir, "single_cell_processed.rds"), compress = FALSE)

composition <- annotation_summary %>%
  group_by(orig.ident, RG_ITH_group) %>%
  mutate(fraction = n_cells / sum(n_cells)) %>%
  ungroup()
composition_tests <- composition %>%
  group_by(main_cell_type) %>%
  summarise(
    n_low = sum(RG_ITH_group == "Low"),
    n_high = sum(RG_ITH_group == "High"),
    p_value = ifelse(
      n_low >= 2 & n_high >= 2,
      wilcox.test(fraction ~ RG_ITH_group, exact = FALSE)$p.value,
      NA_real_
    ),
    .groups = "drop"
  ) %>%
  mutate(p_adjusted = p.adjust(p_value, method = "BH"))
write_csv(composition, file.path(output_dir, "cell_type_composition_by_sample.csv"))
write_csv(composition_tests, file.path(output_dir, "cell_type_composition_tests.csv"))

DefaultAssay(seu) <- "RNA"
if (exists("JoinLayers", mode = "function")) {
  seu <- JoinLayers(seu)
}

# Compare RG-ITHscore groups within each annotated cell population.
# Positive fold changes denote higher expression in the High-risk group.
run_high_low_de <- function(logfc_threshold, min_detected_fraction) {
  bind_rows(lapply(cell_type_order, function(cell_type) {
    cells_keep <- Cells(seu)[as.character(seu$main_cell_type) == cell_type]
    cell_subset <- subset(seu, cells = cells_keep)
    Idents(cell_subset) <- "RG_ITH_group"
    if (!all(c("Low", "High") %in% levels(Idents(cell_subset)))) return(NULL)

    differential_table <- FindMarkers(
      cell_subset,
      ident.1 = "High", ident.2 = "Low",
      only.pos = FALSE, test.use = "wilcox",
      logfc.threshold = logfc_threshold,
      min.pct = min_detected_fraction
    )
    differential_table$gene <- rownames(differential_table)
    differential_table$cell_type <- cell_type
    differential_table$comparison <- "High versus Low"
    differential_table
  }))
}

markers <- run_high_low_de(logfc_threshold = 0.4, min_detected_fraction = 0.21)
write_csv(
  markers,
  file.path(output_dir, "cell_type_high_vs_low_differential_expression.csv")
)

markers_for_gsea <- run_high_low_de(
  logfc_threshold = 0.3,
  min_detected_fraction = 0.25
)
write_csv(
  markers_for_gsea,
  file.path(output_dir, "cell_type_high_vs_low_differential_expression_for_gsea.csv")
)

hallmark_table <- tryCatch(
  msigdbr(species = "Homo sapiens", collection = "H"),
  error = function(e) msigdbr(species = "Homo sapiens", category = "H")
)
hallmark_sets <- split(hallmark_table$gene_symbol, hallmark_table$gs_name)
fold_change_column <- if ("avg_log2FC" %in% colnames(markers_for_gsea)) {
  "avg_log2FC"
} else {
  "avg_logFC"
}

gsea_results <- bind_rows(lapply(unique(markers_for_gsea$cell_type), function(cell_type) {
  ranked_table <- markers_for_gsea %>%
    filter(cell_type == .env$cell_type, !is.na(.data[[fold_change_column]]), !is.na(p_val)) %>%
    mutate(
      rank_score = sign(.data[[fold_change_column]]) *
        abs(.data[[fold_change_column]]) *
        -log10(pmax(p_val, .Machine$double.xmin))
    ) %>%
    arrange(desc(rank_score)) %>%
    distinct(gene, .keep_all = TRUE)
  ranks <- ranked_table$rank_score
  names(ranks) <- ranked_table$gene
  ranks <- sort(ranks[is.finite(ranks)], decreasing = TRUE)
  if (length(ranks) < 100) return(NULL)
  fgseaMultilevel(
    pathways = hallmark_sets, stats = ranks,
    minSize = 10, maxSize = 500
  ) %>%
    as.data.frame() %>%
    mutate(
      cell_type = as.character(cell_type),
      comparison = "High versus Low",
      enriched_group = ifelse(NES > 0, "High", "Low")
    ) %>%
    select(-any_of("leadingEdge"))
}))
write_csv(
  gsea_results,
  file.path(output_dir, "cell_type_high_vs_low_hallmark_gsea.csv")
)

clean_gene_symbols <- function(x) {
  unique(trimws(unlist(strsplit(x, "///", fixed = TRUE))))
}

ecm_genes <- c(
  "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2",
  "COL6A3", "FN1", "LAMA2", "LAMB1", "LAMC1", "THBS1", "THBS2", "TNC",
  "VCAN", "SPARC", "MMP2", "MMP9", "MMP11", "MMP14", "TIMP1", "TIMP2",
  "LOX", "LOXL2", "POSTN", "CTHRC1", "FAP", "ACTA2", "TAGLN", "SERPINE1",
  "ITGA5", "ITGB1"
)
cytotoxic_genes <- c(
  "CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "TRAC", "TRBC1", "TRBC2",
  "GZMA", "GZMB", "GZMK", "PRF1", "NKG7", "GNLY", "IFNG", "CXCL9",
  "CXCL10", "CXCL11", "CCL5", "STAT1", "IRF1", "HLA-A", "HLA-B", "HLA-C",
  "HLA-DRA", "HLA-DRB1", "PDCD1", "CD274", "LAG3", "HAVCR2", "TIGIT", "CTLA4"
)
myeloid_genes <- c(
  "LYZ", "LST1", "FCN1", "S100A8", "S100A9", "S100A12", "AIF1", "TYROBP",
  "IL1B", "IL6", "TNF", "NLRP3", "CXCL8", "CXCL2", "CXCL3", "CCL2", "CCL3",
  "CCL4", "PTGS2", "PLAUR", "FCGR3A", "FCGR2A", "CD14", "ITGAM", "TLR2",
  "TLR4", "BCL2A1", "EREG", "NAMPT", "OSM", "SERPINE1"
)
ir_subclone_raw <- c(
  "SFTPC", "ADH1B", "WIF1", "PIR-FIGF///FIGF", "CLDN18", "CHRDL1", "CPB2",
  "SCGB3A2", "FOSB", "C7", "AOC3", "MFAP4", "MAMDC2", "CYP4B1", "ADAMTS8",
  "C2orf40", "SFTPD", "INMT", "AGER", "ACKR1", "COL6A6", "SCGB1A1", "HBB",
  "TMEM100", "ABCA8", "SDPR", "TCF21", "BCHE", "SCN4B", "FMO2", "FHL1", "CFD",
  "HBA2///HBA1", "TGFBR3", "GPM6A", "SCARA5", "SCN7A", "LRRK2", "CACNA2D2",
  "ABI3BP", "PTGDS", "ADAMTS1", "CAV1", "CYP2B7P///CYP2B6", "SOX7", "LAMP3",
  "AQP4", "SFTPA2///SFTPA1", "IL33", "ZNF385B", "SPARCL1", "PGC", "SELE",
  "TSPAN7", "IGSF10", "SFTPB", "ANKRD29", "GPX3", "VWF", "LINC00968", "KLF4",
  "SLIT2", "PDK4", "LOC100653057///CES1", "CD69", "LOC101926959", "SUSD2",
  "LRRN3", "FABP4", "CLIC5", "EDNRB", "LPL", "PEBP4", "LOC105379426", "RGCC",
  "FAM150B", "TNXB///TNXA", "TNNC1", "ZBTB16", "SLC6A4", "GPIHBP1", "FAM107A",
  "C4BPA", "OGN", "ZFP36", "EXOSC7///CLEC3B", "HSD17B6", "CLDN5", "MCEMP1",
  "ANGPT1", "MRC1", "NR2F1", "CCL15-CCL14///CCL14", "S1PR1", "NR4A2", "MMRN1",
  "FAM13C", "MYH10", "FOS"
)
hypoxia_genes <- hallmark_table %>%
  filter(gs_name == "HALLMARK_HYPOXIA") %>%
  pull(gene_symbol) %>%
  unique()

module_sets_raw <- list(
  Myeloid_inflammatory = myeloid_genes,
  ECM = ecm_genes,
  Cytotoxic_T_cell_inflamed = cytotoxic_genes,
  IR_subclone = clean_gene_symbols(ir_subclone_raw),
  Hypoxia = hypoxia_genes
)
module_sets_used <- lapply(module_sets_raw, function(genes) {
  intersect(unique(genes), rownames(seu))
})
module_sets_missing <- lapply(module_sets_raw, function(genes) {
  setdiff(unique(genes), rownames(seu))
})

gene_set_audit <- bind_rows(lapply(names(module_sets_raw), function(module_name) {
  data.frame(
    module = module_name,
    gene = module_sets_raw[[module_name]],
    represented = module_sets_raw[[module_name]] %in% rownames(seu)
  )
}))
write_csv(gene_set_audit, file.path(output_dir, "module_gene_audit.csv"))
module_sets_used <- module_sets_used[lengths(module_sets_used) >= 2]
if (length(module_sets_used) == 0L) {
  stop("No functional module retained at least two represented genes.")
}

seu <- AddModuleScore(
  seu, features = module_sets_used, name = "SelectedScore",
  assay = "RNA", seed = 123
)
score_columns_old <- paste0("SelectedScore", seq_along(module_sets_used))
score_columns_new <- paste0(names(module_sets_used), "_Score")
colnames(seu@meta.data)[match(score_columns_old, colnames(seu@meta.data))] <-
  score_columns_new

cell_scores <- seu@meta.data %>%
  mutate(cell_id = rownames(seu@meta.data)) %>%
  select(
    cell_id, patient = orig.ident, RG_ITH_group, main_cell_type,
    all_of(score_columns_new)
  )
write_csv(cell_scores, file.path(output_dir, "module_scores_cell_level.csv.gz"))

patient_scores <- cell_scores %>%
  pivot_longer(
    cols = all_of(score_columns_new),
    names_to = "module", values_to = "score"
  ) %>%
  mutate(module = str_remove(module, "_Score$")) %>%
  group_by(patient, RG_ITH_group, main_cell_type, module) %>%
  summarise(
    mean_score = mean(score, na.rm = TRUE),
    median_score = median(score, na.rm = TRUE),
    n_cells = n(),
    .groups = "drop"
  ) %>%
  filter(n_cells >= 10)

patient_score_tests <- patient_scores %>%
  group_by(module, main_cell_type) %>%
  summarise(
    n_low = sum(RG_ITH_group == "Low"),
    n_high = sum(RG_ITH_group == "High"),
    p_value = ifelse(
      n_low >= 2 & n_high >= 2,
      wilcox.test(mean_score ~ RG_ITH_group, exact = FALSE)$p.value,
      NA_real_
    ),
    .groups = "drop"
  ) %>%
  group_by(module) %>%
  mutate(p_adjusted = p.adjust(p_value, method = "BH")) %>%
  ungroup()
write_csv(patient_scores, file.path(output_dir, "module_scores_patient_level.csv"))
write_csv(patient_score_tests, file.path(output_dir, "module_scores_patient_level_tests.csv"))

cell_score_tests <- cell_scores %>%
  pivot_longer(
    cols = all_of(score_columns_new),
    names_to = "module", values_to = "score"
  ) %>%
  mutate(module = str_remove(module, "_Score$")) %>%
  group_by(module, main_cell_type) %>%
  summarise(
    n_low_cells = sum(RG_ITH_group == "Low"),
    n_high_cells = sum(RG_ITH_group == "High"),
    descriptive_p_value = wilcox.test(score ~ RG_ITH_group, exact = FALSE)$p.value,
    .groups = "drop"
  ) %>%
  group_by(module) %>%
  mutate(descriptive_p_adjusted = p.adjust(descriptive_p_value, method = "BH")) %>%
  ungroup()
write_csv(
  cell_score_tests,
  file.path(output_dir, "module_scores_cell_level_descriptive_tests.csv")
)

saveRDS(seu, file.path(output_dir, "single_cell_processed_with_scores.rds"),
        compress = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))

message("Single-cell analysis completed. Outputs: ", output_dir)
