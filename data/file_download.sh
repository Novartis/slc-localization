#!/bin/bash

# Set the base URL for the files
BASE_URL="https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/630/S-BIAD1630/Files/"

# Input TSV file
INPUT_FILE="filelist_sample_HATAG.tsv" # Replace with your actual file name

# Starting row (optional)
START_ROW=0 # Default to start from the beginning

# Ending row (optional)
END_ROW=0 # Default to download to the end of the file

# Check if start and end rows are provided as arguments
if [ $# -gt 0 ]; then
  START_ROW=$1
  if [ $# -gt 1 ]; then
    END_ROW=$2
  fi
fi

# Get the total number of lines in the input file
TOTAL_LINES=$(wc -l < "$INPUT_FILE")

# If END_ROW is greater than the total number of lines, set it to the maximum
if [ "$END_ROW" -gt "$TOTAL_LINES" ]; then
  END_ROW="$TOTAL_LINES"
fi

# Calculate the number of rows to download
ROWS_TO_DOWNLOAD=$((END_ROW - START_ROW))

# Skip to the starting row
if [ "$START_ROW" -gt 0 ]; then
  head -n "$((START_ROW + 1))" "$INPUT_FILE" | tail -n 1 > temp_start_row.txt
  START_ROW=$((START_ROW + 1))
fi

# Loop through the specified number of rows
count=0
while IFS= read -r line && [ $count -lt $ROWS_TO_DOWNLOAD ]; do
  # Extract the filename from the first column
  filename=$(echo "$line" | cut -f 1)

  # Construct the full URL
  full_url="$BASE_URL/$filename"

  # Extract the directory name from the filename
  dirname=$(dirname "$filename")

  # Create the directory if it doesn't exist
  mkdir -p "$dirname"

  # Skip if the file already exists
  if [ -f "$filename" ]; then
    echo -e "\n[INFO] $filename already exists, skipping."
    continue
  fi

  # Download the file using curl with retries and error handling
  echo -ne "[${count}/${ROWS_TO_DOWNLOAD}] Downloading: $full_url to $filename\r"
  curl --fail --retry 3 --retry-delay 2 -L -o "$filename" "$full_url"
  if [ $? -ne 0 ]; then
    echo -e "\n[WARNING] Failed to download $full_url"
    continue
  fi

  # Check if the file is a valid image
  filetype=$(file --mime-type -b "$filename")
  if [[ ! $filetype =~ ^image/ ]]; then
    echo -e "\n[WARNING] $filename is not a valid image (type: $filetype), deleting."
    rm -f "$filename"
    continue
  fi

  ((count++)) # Increment the counter only for valid images

done < "$INPUT_FILE"

echo -e "\nDownload complete. Downloaded $count valid images."

rm -f temp_start_row.txt
