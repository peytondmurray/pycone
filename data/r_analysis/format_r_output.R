# find main directory
original_dir <- getwd()

# get list of output folders to open
output_folders <- list.dirs(recursive = FALSE)

outputfoldernames <- glue("./output__duration{duration} onset{onset}")
output_folders <- output_folders[startsWith(output_folders, outputfoldernames)]

# Function to extract the offset value from the CSV filename
extract_offset <- function(filename) {
  offset_pattern <- "offset(\\d+)"
  offset_value <- str_extract(filename, offset_pattern)
  return(str_extract(offset_value, "\\d+")) # Extract just the number
}

sites <- colnames(exp_dT)[3:ncol(exp_dT)]
r_site_offset <- data.frame()

for (s in sites) {
  site1 <- s

  # Initialize an empty list to store each site1_df data frame
  dfs_list <- list()

  # Loop through the folders
  for (folder in output_folders) {
    # Identify the CSV file with the given pattern
    csv_file <- list.files(folder, pattern = "^r_value__duration.*offset.*\\.csv$", full.names = TRUE)

    # If there's a matching CSV file, process it
    if (length(csv_file) > 0) {
      csv_data <- read_csv(csv_file[1], show_col_types = FALSE) # Read the CSV (assuming only one match per folder)

      # Extract the row where the value is "site1"
      site1_row <- csv_data %>% filter(sites == site1)


      # Continue if the filtered row isn't empty
      if (nrow(site1_row) > 0) {
        site1_matrix <- as.matrix(site1_row[, -which(names(site1_row) == "sites")]) # Exclude the sites column
        transposed_matrix <- t(site1_matrix) # Transpose the matrix
        site1_df <- as.data.frame(transposed_matrix) # Convert back to dataframe

        offset_value <- extract_offset(csv_file[1])

        # Rename the column to the offset value
        colnames(site1_df) <- offset_value

        # Append the dataframe to the dfs_list
        dfs_list[offset_value] <- site1_df
      }
    }
  }

  # Binding the dataframes in the list column-wise
  #  if there is a different number of r values, fill with NA
  max_len <- max(sapply(dfs_list, length))

  df <- do.call(rbind, lapply(dfs_list, function(x) {
    c(x, rep(NA, max_len - length(x)))
  }))


  df <- as.data.frame(df)
  df_sorted <- df[order(as.numeric(rownames(df))), ]

  colnames(df_sorted) <- paste0("r", 1:ncol(df))
  df_sorted$site <- s
  df_sorted$offset <- 0:(nrow(df) - 1)


  r_site_offset <- rbind(r_site_offset, df_sorted)
}

fname <- glue("r_duration{duration}.csv")
write.csv(r_site_offset, fname, row.names = FALSE)
