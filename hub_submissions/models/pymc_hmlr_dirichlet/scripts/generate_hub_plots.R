#!/usr/bin/env Rscript
# Generate summary plots using variant nowcast hub's plotting functions

# Load required libraries
library(here)

# Source the hub's plotting functions
hub_path <- "/Users/trobacker/GitHub/variant-nowcast-hub/"
source(file.path(hub_path, "src", "plot_summary_graphs.R"))

# Set parameters
# model_output_file should be relative to hub_path/model-output/
model_output_file <- "YourTeam-PyMC-HMLR-Dirichlet/2025-11-12-YourTeam-PyMC-HMLR-Dirichlet.parquet"

# S3 date: Most recent Monday on or before nowcast date
# 2025-11-12 is Wednesday, so most recent Monday is 2025-11-10
s3_data_date <- "2025-11-10"

# Output to Desktop (must end with /)
save_path <- "/Users/trobacker/Desktop/"

# Check if model output file exists
full_model_path <- file.path(hub_path, "model-output", model_output_file)
if (!file.exists(full_model_path)) {
  stop(paste("Model output file not found:", full_model_path))
}

cat("Generating summary plots...\n")
cat("  Model output:", model_output_file, "\n")
cat("  S3 data date:", s3_data_date, "\n")
cat("  Save path:", save_path, "\n\n")

# Generate plots
plot_summary_graphs(
  model_output_file = model_output_file,
  s3_data_date = s3_data_date,
  hub_path = hub_path,
  save_path = save_path
)

cat("\n✓ Summary plots generated successfully!\n")
