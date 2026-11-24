suppressPackageStartupMessages({
  library(NMF)
  library(clusterProfiler)
  library(org.Hs.eg.db)
})

parse_args <- function(args) {
  result <- list()
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[[i]])
    if (i == length(args)) stop(sprintf("Missing value for --%s", key))
    result[[key]] <- args[[i + 1]]
    i <- i + 2
  }
  result
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(args$expression) || is.null(args$`output-dir`)) {
  stop("Usage: Rscript nmf_subtyping_and_go.R --expression <gene_by_sample.csv> --output-dir <directory>")
}

expression_path <- normalizePath(args$expression, mustWork = TRUE)
output_dir <- args$`output-dir`
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

expression_df <- read.csv(expression_path, row.names = 1, check.names = FALSE)
expression_matrix <- as.matrix(expression_df)
storage.mode(expression_matrix) <- "double"
if (anyNA(expression_matrix) || any(!is.finite(expression_matrix))) {
  stop("Expression matrix contains missing or non-finite values")
}
if (min(expression_matrix) < 0) {
  stop("NMF input must be non-negative. Supply the finalized non-negative expression matrix.")
}

rank_seed <- 123456
factor_seed <- 2025
rank_estimate <- nmfEstimateRank(
  expression_matrix,
  range = 2:6,
  method = "brunet",
  nrun = 3,
  seed = rank_seed
)

rank_summary <- as.data.frame(summary(rank_estimate))
rank_summary$rank <- rownames(rank_summary)
write.csv(rank_summary, file.path(output_dir, "nmf_rank_diagnostics.csv"), row.names = FALSE)

nmf_k5 <- nmf(
  expression_matrix,
  rank = 5,
  method = "brunet",
  nrun = 10,
  seed = factor_seed
)

basis_matrix <- basis(nmf_k5)
coefficient_matrix <- t(coef(nmf_k5))
colnames(basis_matrix) <- paste0("Subclone", seq_len(ncol(basis_matrix)))
colnames(coefficient_matrix) <- paste0("Subclone", seq_len(ncol(coefficient_matrix)))

row_totals <- rowSums(coefficient_matrix)
if (any(row_totals <= 0)) stop("At least one sample has a non-positive total NMF coefficient")
proportions <- sweep(coefficient_matrix, 1, row_totals, FUN = "/")

write.csv(basis_matrix, file.path(output_dir, "nmf_k5_gene_loadings.csv"))
write.csv(coefficient_matrix, file.path(output_dir, "nmf_k5_sample_coefficients.csv"))
write.csv(proportions, file.path(output_dir, "nmf_k5_sample_proportions.csv"))

top_genes <- lapply(seq_len(ncol(basis_matrix)), function(index) {
  names(sort(basis_matrix[, index], decreasing = TRUE))[seq_len(min(100, nrow(basis_matrix)))]
})
top_gene_table <- do.call(
  rbind,
  lapply(seq_along(top_genes), function(index) {
    data.frame(Subclone = paste0("Subclone", index), rank = seq_along(top_genes[[index]]), gene = top_genes[[index]])
  })
)
write.csv(top_gene_table, file.path(output_dir, "nmf_k5_top100_genes.csv"), row.names = FALSE)

go_results <- lapply(seq_along(top_genes), function(index) {
  mapped <- bitr(
    top_genes[[index]],
    fromType = "SYMBOL",
    toType = "ENTREZID",
    OrgDb = org.Hs.eg.db
  )
  if (nrow(mapped) == 0) return(NULL)
  enriched <- enrichGO(
    gene = unique(mapped$ENTREZID),
    OrgDb = org.Hs.eg.db,
    ont = "BP",
    pAdjustMethod = "BH",
    qvalueCutoff = 0.20,
    readable = TRUE
  )
  result <- as.data.frame(enriched)
  if (nrow(result) == 0) return(NULL)
  result$Subclone <- paste0("Subclone", index)
  result
})
go_combined <- do.call(rbind, Filter(Negate(is.null), go_results))
if (is.null(go_combined)) go_combined <- data.frame()
write.csv(go_combined, file.path(output_dir, "nmf_k5_go_biological_process.csv"), row.names = FALSE)

metadata <- list(
  input = expression_path,
  rank_candidates = 2:6,
  rank_method = "brunet",
  rank_nrun = 3,
  rank_seed = rank_seed,
  selected_rank = 5,
  final_nrun = 10,
  final_seed = factor_seed,
  top_genes_per_component = 100,
  go_ontology = "BP",
  go_adjustment = "BH"
)
writeLines(jsonlite::toJSON(metadata, pretty = TRUE, auto_unbox = TRUE), file.path(output_dir, "nmf_analysis_metadata.json"))
