suppressPackageStartupMessages({
  library(survival)
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

read_components <- function(path) {
  frame <- read.csv(path, check.names = FALSE)
  if (!"SampleID" %in% colnames(frame)) {
    colnames(frame)[1] <- "SampleID"
  }
  required <- c("SampleID", paste0("Subclone", 1:5))
  if (!all(required %in% colnames(frame))) {
    stop(sprintf("Component table must contain: %s", paste(required, collapse = ", ")))
  }
  frame[, required]
}

read_clusters <- function(path) {
  frame <- read.csv(path, check.names = FALSE)
  required <- c("SampleID", "Cluster")
  if (!all(required %in% colnames(frame))) {
    stop("Cluster table must contain SampleID and Cluster")
  }
  frame[, required]
}

read_survival <- function(path, endpoint) {
  frame <- read.csv(path, check.names = FALSE)
  required <- c("SampleID", "time", "status")
  if (!all(required %in% colnames(frame))) {
    stop(sprintf("%s table must contain SampleID, time and status", endpoint))
  }
  frame <- frame[, required]
  frame$time <- as.numeric(frame$time)
  frame$status <- as.integer(frame$status)
  if (anyNA(frame$time) || anyNA(frame$status)) stop(sprintf("%s table contains missing outcomes", endpoint))
  if (any(frame$time < 0) || any(!frame$status %in% c(0L, 1L))) {
    stop(sprintf("%s time/status values are invalid", endpoint))
  }
  frame$Endpoint <- endpoint
  frame
}

analyse_group <- function(data, group_column, comparison) {
  group <- droplevels(factor(data[[group_column]]))
  if (nlevels(group) != 2) stop(sprintf("%s must contain exactly two groups", group_column))
  model_data <- data.frame(time = data$time, status = data$status, group = group)
  fit <- coxph(Surv(time, status) ~ group, data = model_data, ties = "efron", x = TRUE)
  fit_summary <- summary(fit)
  logrank <- survdiff(Surv(time, status) ~ group, data = model_data, rho = 0)
  ph <- cox.zph(fit, transform = "km")
  counts <- table(group)
  events <- tapply(model_data$status, group, sum)
  data.frame(
    endpoint = unique(data$Endpoint),
    comparison = comparison,
    reference_group = levels(group)[1],
    contrast_group = levels(group)[2],
    reference_n = as.integer(counts[[1]]),
    contrast_n = as.integer(counts[[2]]),
    reference_events = as.integer(events[[1]]),
    contrast_events = as.integer(events[[2]]),
    logrank_p = pchisq(logrank$chisq, df = length(logrank$n) - 1, lower.tail = FALSE),
    hazard_ratio = unname(fit_summary$conf.int[1, "exp(coef)"]),
    ci_lower_95 = unname(fit_summary$conf.int[1, "lower .95"]),
    ci_upper_95 = unname(fit_summary$conf.int[1, "upper .95"]),
    cox_p = unname(fit_summary$coefficients[1, "Pr(>|z|)"]),
    schoenfeld_p = unname(ph$table["GLOBAL", "p"]),
    stringsAsFactors = FALSE
  )
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(args$components) || is.null(args$clusters) || is.null(args$`output-dir`)) {
  stop(paste(
    "Usage: Rscript ecm_ir_survival.R",
    "--components <nmf_k5_sample_proportions.csv>",
    "--clusters <ecm_ir_cluster_assignments.csv>",
    "--output-dir <directory>",
    "[--os <os.csv>] [--dfs <dfs.csv>]"
  ))
}
if (is.null(args$os) && is.null(args$dfs)) stop("At least one of --os or --dfs is required")

output_dir <- args$`output-dir`
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
components <- read_components(normalizePath(args$components, mustWork = TRUE))
clusters <- read_clusters(normalizePath(args$clusters, mustWork = TRUE))
analysis_base <- merge(components, clusters, by = "SampleID", all = FALSE)

survival_inputs <- list()
if (!is.null(args$os)) survival_inputs$OS <- read_survival(normalizePath(args$os, mustWork = TRUE), "OS")
if (!is.null(args$dfs)) survival_inputs$DFS <- read_survival(normalizePath(args$dfs, mustWork = TRUE), "DFS")

all_statistics <- list()
all_analysis_rows <- list()
for (endpoint in names(survival_inputs)) {
  endpoint_data <- merge(analysis_base, survival_inputs[[endpoint]], by = "SampleID", all = FALSE)
  endpoint_data$Subclone4_group <- factor(
    ifelse(endpoint_data$Subclone4 >= median(endpoint_data$Subclone4), "High", "Low"),
    levels = c("Low", "High")
  )
  endpoint_data$Subclone5_group <- factor(
    ifelse(endpoint_data$Subclone5 >= median(endpoint_data$Subclone5), "High", "Low"),
    levels = c("Low", "High")
  )
  endpoint_data$Cluster <- factor(endpoint_data$Cluster, levels = sort(unique(endpoint_data$Cluster)))
  all_statistics[[paste(endpoint, "ECM", sep = "_")]] <- analyse_group(
    endpoint_data,
    "Subclone4_group",
    "ECM component: high versus low"
  )
  all_statistics[[paste(endpoint, "IR", sep = "_")]] <- analyse_group(
    endpoint_data,
    "Subclone5_group",
    "IR component: high versus low"
  )
  all_statistics[[paste(endpoint, "Cluster", sep = "_")]] <- analyse_group(
    endpoint_data,
    "Cluster",
    "ECM/IR cluster: cluster 2 versus cluster 1"
  )
  all_analysis_rows[[endpoint]] <- endpoint_data
}

statistics <- do.call(rbind, all_statistics)
analysis_rows <- do.call(rbind, all_analysis_rows)
rownames(statistics) <- NULL
rownames(analysis_rows) <- NULL
write.csv(statistics, file.path(output_dir, "ecm_ir_survival_statistics.csv"), row.names = FALSE)
write.csv(analysis_rows, file.path(output_dir, "ecm_ir_survival_analysis_table.csv"), row.names = FALSE)

metadata <- list(
  component_grouping = "Median abundance; values equal to the median are assigned High",
  cluster_comparison = "Cluster 2 versus cluster 1",
  logrank_test = "Two-sided log-rank test",
  cox_model = "Univariable Cox proportional-hazards model with Efron ties",
  confidence_interval = 0.95,
  proportional_hazards_test = "Global Schoenfeld residual test with KM time transform",
  endpoints = names(survival_inputs)
)
writeLines(
  toJSON(metadata, pretty = TRUE, auto_unbox = TRUE),
  file.path(output_dir, "ecm_ir_survival_metadata.json")
)
