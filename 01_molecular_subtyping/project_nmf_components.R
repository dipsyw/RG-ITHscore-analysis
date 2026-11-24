suppressPackageStartupMessages({
  library(NMF)
  library(jsonlite)
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

read_numeric_matrix <- function(path) {
  frame <- read.csv(path, row.names = 1, check.names = FALSE)
  matrix <- as.matrix(frame)
  storage.mode(matrix) <- "double"
  if (anyNA(matrix) || any(!is.finite(matrix))) {
    stop(sprintf("Matrix contains missing or non-finite values: %s", path))
  }
  matrix
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
required <- c("basis", "expression", "reference-centers", "output-dir")
missing_args <- required[vapply(required, function(key) is.null(args[[key]]), logical(1))]
if (length(missing_args) > 0) {
  stop(paste(
    "Usage: Rscript project_nmf_components.R",
    "--basis <nmf_k5_gene_loadings.csv>",
    "--expression <gene_by_sample.csv>",
    "--reference-centers <ecm_ir_cluster_centroids.csv>",
    "--output-dir <directory>",
    "[--seed 123] [--minimum-overlap 0.70]"
  ))
}

basis_path <- normalizePath(args$basis, mustWork = TRUE)
expression_path <- normalizePath(args$expression, mustWork = TRUE)
centers_path <- normalizePath(args$`reference-centers`, mustWork = TRUE)
output_dir <- args$`output-dir`
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

seed <- if (is.null(args$seed)) 123L else as.integer(args$seed)
minimum_overlap <- if (is.null(args$`minimum-overlap`)) 0.70 else as.numeric(args$`minimum-overlap`)
if (!is.finite(minimum_overlap) || minimum_overlap <= 0 || minimum_overlap > 1) {
  stop("--minimum-overlap must be in (0, 1]")
}

basis_matrix <- read_numeric_matrix(basis_path)
expression_matrix <- read_numeric_matrix(expression_path)
if (ncol(basis_matrix) != 5) stop("The reference basis must contain five NMF components")
if (is.null(colnames(basis_matrix))) colnames(basis_matrix) <- paste0("Subclone", 1:5)

common_genes <- intersect(rownames(basis_matrix), rownames(expression_matrix))
overlap_fraction <- length(common_genes) / nrow(basis_matrix)
if (length(common_genes) == 0) stop("The reference basis and target expression matrix share no genes")
if (overlap_fraction < minimum_overlap) {
  warning(sprintf(
    "Only %.1f%% of reference-basis genes were found in the target matrix",
    100 * overlap_fraction
  ))
}

reference_basis <- basis_matrix[common_genes, , drop = FALSE]
target_matrix <- expression_matrix[common_genes, , drop = FALSE]
target_matrix <- target_matrix[rownames(reference_basis), , drop = FALSE]
stopifnot(identical(rownames(reference_basis), rownames(target_matrix)))

# This preserves the non-negative shift recorded in the original external-projection script.
minimum_value <- min(target_matrix)
shift_applied <- if (minimum_value < 0) -minimum_value else 0
target_nonnegative <- target_matrix + shift_applied + 1e-6

model_reference <- nmfModel(rank = 5, W = reference_basis)
set.seed(seed)
projection_fit <- nmf(
  x = target_nonnegative,
  rank = 5,
  model = model_reference,
  nrun = 1,
  seed = seed
)

coefficients <- coef(projection_fit)
rownames(coefficients) <- paste0("Subclone", 1:5)
column_totals <- colSums(coefficients)
if (any(column_totals <= 0)) stop("At least one projected sample has non-positive total coefficients")
proportions <- sweep(coefficients, 2, column_totals, FUN = "/")

centers <- read.csv(centers_path, check.names = FALSE)
required_center_columns <- c("Cluster", "ECM_center", "IR_center")
if (!all(required_center_columns %in% colnames(centers))) {
  stop(sprintf("Reference-centre table must contain: %s", paste(required_center_columns, collapse = ", ")))
}
centers <- centers[order(centers$Cluster), required_center_columns]
if (nrow(centers) != 2) stop("Reference-centre table must contain exactly two clusters")

ecm_ir <- t(proportions[c("Subclone4", "Subclone5"), , drop = FALSE])
colnames(ecm_ir) <- c("ECM_proportion", "IR_proportion")
predict_cluster <- function(sample_values) {
  distances <- apply(
    centers[, c("ECM_center", "IR_center"), drop = FALSE],
    1,
    function(center) sum((sample_values - center)^2)
  )
  centers$Cluster[[which.min(distances)]]
}
cluster <- apply(ecm_ir, 1, predict_cluster)

coefficient_output <- as.data.frame(t(coefficients), check.names = FALSE)
coefficient_output$SampleID <- rownames(coefficient_output)
coefficient_output <- coefficient_output[, c("SampleID", paste0("Subclone", 1:5))]
proportion_output <- as.data.frame(t(proportions), check.names = FALSE)
proportion_output$SampleID <- rownames(proportion_output)
proportion_output <- proportion_output[, c("SampleID", paste0("Subclone", 1:5))]
assignment_output <- data.frame(
  SampleID = rownames(ecm_ir),
  ECM_proportion = ecm_ir[, "ECM_proportion"],
  IR_proportion = ecm_ir[, "IR_proportion"],
  Cluster = as.integer(cluster),
  row.names = NULL
)

write.csv(coefficient_output, file.path(output_dir, "projected_nmf_coefficients.csv"), row.names = FALSE)
write.csv(proportion_output, file.path(output_dir, "projected_nmf_proportions.csv"), row.names = FALSE)
write.csv(assignment_output, file.path(output_dir, "projected_ecm_ir_clusters.csv"), row.names = FALSE)

metadata <- list(
  basis = basis_path,
  expression = expression_path,
  reference_centers = centers_path,
  reference_gene_count = nrow(basis_matrix),
  target_gene_count = nrow(expression_matrix),
  common_gene_count = length(common_genes),
  overlap_fraction = overlap_fraction,
  minimum_overlap = minimum_overlap,
  nonnegative_shift = shift_applied,
  projection_seed = seed,
  projection_nrun = 1,
  cluster_distance = "Euclidean",
  clustering_components = c("Subclone4", "Subclone5")
)
writeLines(
  toJSON(metadata, pretty = TRUE, auto_unbox = TRUE),
  file.path(output_dir, "projection_metadata.json")
)
