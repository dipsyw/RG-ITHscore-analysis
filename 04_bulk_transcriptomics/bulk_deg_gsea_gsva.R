suppressPackageStartupMessages({
  library(limma)
  library(fgsea)
  library(GSVA)
})

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
required <- c("expression", "metadata", "gmt", "output-dir")
missing <- required[vapply(required, function(x) is.null(args[[x]]), logical(1))]
if (length(missing)) {
  stop(paste(
    "Usage: Rscript bulk_deg_gsea_gsva.R",
    "--expression <gene_by_sample.csv>",
    "--metadata <metadata.csv>",
    "--gmt <comma-separated GMT files>",
    "--output-dir <directory>"
  ))
}

expression <- read.csv(normalizePath(args$expression, mustWork = TRUE), row.names = 1, check.names = FALSE)
expression <- as.matrix(expression); storage.mode(expression) <- "double"
metadata <- read.csv(normalizePath(args$metadata, mustWork = TRUE), check.names = FALSE)
required_metadata <- c("sample_id", "RG_ITH_group", "RG_ITHscore", "dataset", "histology")
if (!all(required_metadata %in% names(metadata))) stop("Metadata must contain: ", paste(required_metadata, collapse = ", "))
if (!identical(colnames(expression), metadata$sample_id)) stop("Expression columns must exactly match metadata sample_id order")
if (anyNA(expression) || any(!is.finite(expression))) stop("Expression contains missing or non-finite values")
metadata$RG_ITH_group <- factor(metadata$RG_ITH_group, levels = c("Low", "High"))
metadata$dataset <- factor(metadata$dataset)
metadata$histology <- factor(metadata$histology)

design <- model.matrix(~ RG_ITH_group + dataset + histology, data = metadata)
if (qr(design)$rank != ncol(design)) stop("The prespecified limma design is rank deficient")
fit <- eBayes(lmFit(expression, design), trend = TRUE, robust = TRUE)
coefficient <- "RG_ITH_groupHigh"
deg <- topTable(fit, coef = coefficient, number = Inf, sort.by = "none", adjust.method = "BH")
deg$gene <- rownames(deg)
deg$passes_reported_threshold <- deg$adj.P.Val < 0.05 & abs(deg$logFC) > 0.5
ranked <- deg$t; names(ranked) <- deg$gene
ranked <- sort(ranked[is.finite(ranked)], decreasing = TRUE)

gmt_paths <- strsplit(args$gmt, ",", fixed = TRUE)[[1]]
gmt_paths <- trimws(gmt_paths)
if (!all(file.exists(gmt_paths))) stop("One or more GMT files do not exist")
pathways_by_collection <- lapply(gmt_paths, fgsea::gmtPathways)
names(pathways_by_collection) <- tools::file_path_sans_ext(basename(gmt_paths))

gsea <- do.call(rbind, lapply(names(pathways_by_collection), function(collection) {
  result <- as.data.frame(fgseaMultilevel(
    pathways = pathways_by_collection[[collection]], stats = ranked,
    minSize = 10, maxSize = 500, eps = 0
  ))
  result$collection <- collection
  result$leadingEdge <- vapply(result$leadingEdge, paste, collapse = ";", FUN.VALUE = character(1))
  result
}))

all_pathways <- unlist(pathways_by_collection, recursive = FALSE)
names(all_pathways) <- make.unique(names(all_pathways))
if ("gsvaParam" %in% getNamespaceExports("GSVA")) {
  parameter <- gsvaParam(
    expression, all_pathways, kcdf = "Gaussian", tau = 1,
    maxDiff = TRUE, minSize = 10, maxSize = 500
  )
  gsva_scores <- gsva(parameter, verbose = FALSE)
} else {
  gsva_scores <- gsva(
    expression, all_pathways, method = "gsva", kcdf = "Gaussian", tau = 1,
    mx.diff = TRUE, min.sz = 10, max.sz = 500, verbose = FALSE
  )
}

gsva_statistics <- do.call(rbind, lapply(seq_len(nrow(gsva_scores)), function(i) {
  values <- gsva_scores[i, ]
  group_test <- wilcox.test(values ~ metadata$RG_ITH_group, exact = FALSE)
  association <- suppressWarnings(cor.test(values, metadata$RG_ITHscore, method = "spearman", exact = FALSE))
  data.frame(
    pathway = rownames(gsva_scores)[i],
    median_high = median(values[metadata$RG_ITH_group == "High"]),
    median_low = median(values[metadata$RG_ITH_group == "Low"]),
    mann_whitney_p = group_test$p.value,
    spearman_rho = unname(association$estimate),
    spearman_p = association$p.value
  )
}))
gsva_statistics$mann_whitney_q_bh <- p.adjust(gsva_statistics$mann_whitney_p, method = "BH")
gsva_statistics$spearman_q_bh <- p.adjust(gsva_statistics$spearman_p, method = "BH")

output_dir <- args$`output-dir`; dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(deg, file.path(output_dir, "limma_differential_expression.csv"), row.names = FALSE)
write.csv(data.frame(gene = names(ranked), moderated_t = unname(ranked)), file.path(output_dir, "limma_preranked_genes.csv"), row.names = FALSE)
write.csv(gsea, file.path(output_dir, "preranked_gsea.csv"), row.names = FALSE)
write.csv(data.frame(pathway = rownames(gsva_scores), gsva_scores, check.names = FALSE), file.path(output_dir, "gsva_scores.csv"), row.names = FALSE)
write.csv(gsva_statistics, file.path(output_dir, "gsva_group_and_score_statistics.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
