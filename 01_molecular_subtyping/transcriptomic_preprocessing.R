suppressPackageStartupMessages(library(limma))

parse_args <- function(values) {
  out <- list(); i <- 1L
  while (i <= length(values)) {
    key <- sub("^--", "", values[[i]])
    if (i == length(values)) stop("Missing value for --", key)
    out[[key]] <- values[[i + 1L]]; i <- i + 2L
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(args$expression) || is.null(args$metadata) || is.null(args$`output-dir`)) {
  stop(paste(
    "Usage: Rscript transcriptomic_preprocessing.R",
    "--expression <gene_by_sample.csv> --metadata <sample_id_dataset.csv>",
    "--output-dir <directory> [--transform log2p1|none]"
  ))
}
transform <- if (is.null(args$transform)) "none" else args$transform
if (!transform %in% c("none", "log2p1")) stop("--transform must be none or log2p1")

expression <- read.csv(normalizePath(args$expression, mustWork = TRUE), row.names = 1, check.names = FALSE)
expression <- as.matrix(expression); storage.mode(expression) <- "double"
metadata <- read.csv(normalizePath(args$metadata, mustWork = TRUE), check.names = FALSE)
if (!all(c("sample_id", "dataset") %in% names(metadata))) stop("Metadata must contain sample_id and dataset")
if (!identical(colnames(expression), metadata$sample_id)) stop("Expression columns must match metadata sample_id order")
if (anyNA(expression) || any(!is.finite(expression))) stop("Expression contains missing or non-finite values")
if (transform == "log2p1") {
  if (any(expression < 0)) stop("log2p1 requires non-negative input")
  expression <- log2(expression + 1)
}

minimum_samples <- ceiling(0.20 * ncol(expression))
keep <- rowSums(expression > 1) >= minimum_samples
filtered <- expression[keep, , drop = FALSE]
corrected <- removeBatchEffect(filtered, batch = factor(metadata$dataset))
pca <- prcomp(t(corrected), center = TRUE, scale. = TRUE)
pca_scores <- data.frame(sample_id = rownames(pca$x), dataset = metadata$dataset, pca$x[, seq_len(min(10, ncol(pca$x))), drop = FALSE])

output_dir <- args$`output-dir`; dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(data.frame(gene = rownames(corrected), corrected, check.names = FALSE), file.path(output_dir, "batch_corrected_expression.csv"), row.names = FALSE)
write.csv(pca_scores, file.path(output_dir, "post_correction_pca_scores.csv"), row.names = FALSE)
write.csv(data.frame(
  genes_before = nrow(expression), genes_after = nrow(corrected),
  low_expression_rule = paste0("expression > 1 in at least ", minimum_samples, " samples"),
  transform = transform
), file.path(output_dir, "preprocessing_summary.csv"), row.names = FALSE)
