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
if (is.null(args$input) || is.null(args$`output-dir`)) {
  stop("Usage: Rscript microenvironment_statistics.R --input <sample_level_scores.csv> --output-dir <directory> [--family-map <feature_family.csv>]")
}
data <- read.csv(normalizePath(args$input, mustWork = TRUE), check.names = FALSE)
required <- c("sample_id", "RG_ITH_group", "RG_ITHscore")
if (!all(required %in% names(data))) stop("Input must contain: ", paste(required, collapse = ", "))
data$RG_ITH_group <- factor(data$RG_ITH_group, levels = c("Low", "High"))
if ("ESTIMATEScore" %in% names(data) && !"tumor_purity" %in% names(data)) {
  data$tumor_purity <- cos(0.6049872018 + 0.0001467884 * data$ESTIMATEScore)
}
features <- setdiff(names(data)[vapply(data, is.numeric, logical(1))], "RG_ITHscore")

family <- setNames(rep("microenvironment", length(features)), features)
if (!is.null(args$`family-map`)) {
  map <- read.csv(normalizePath(args$`family-map`, mustWork = TRUE), check.names = FALSE)
  if (!all(c("feature", "family") %in% names(map))) stop("Family map must contain feature and family")
  family[intersect(map$feature, features)] <- map$family[match(intersect(map$feature, features), map$feature)]
}

results <- do.call(rbind, lapply(features, function(feature) {
  values <- data[[feature]]
  group_test <- wilcox.test(values ~ data$RG_ITH_group, exact = FALSE)
  association <- suppressWarnings(cor.test(values, data$RG_ITHscore, method = "spearman", exact = FALSE))
  data.frame(
    feature = feature, family = unname(family[[feature]]),
    n = sum(is.finite(values)),
    median_high = median(values[data$RG_ITH_group == "High"], na.rm = TRUE),
    median_low = median(values[data$RG_ITH_group == "Low"], na.rm = TRUE),
    mann_whitney_p = group_test$p.value,
    spearman_rho = unname(association$estimate),
    spearman_p = association$p.value
  )
}))
results$mann_whitney_q_bh <- ave(results$mann_whitney_p, results$family, FUN = function(x) p.adjust(x, "BH"))
results$spearman_q_bh <- ave(results$spearman_p, results$family, FUN = function(x) p.adjust(x, "BH"))
output_dir <- args$`output-dir`; dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(results, file.path(output_dir, "microenvironment_group_and_score_statistics.csv"), row.names = FALSE)
