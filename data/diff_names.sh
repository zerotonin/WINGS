#!/bin/bash

# Assuming the files are in the current directory

find . -type f -name "*_[0-9][0-9][0-9].csv" | while IFS= read -r file; do
  base_name="${file%_*.csv}"
  base_name="${base_name##*/}" # Remove the path, only keep the filename
  echo "$base_name"
done | sort -u
