#!/bin/bash

# Set the path to the directory containing the input files
#input_dir="../execs/"

#for ((i=1; i<=21; i++)); do
 # if [ $i -ne 2 ]; then
  #  input_files+=("${input_dir}input${i}.txt" "${input_dir}output${i}_adj_sorted.txt")
  #fi
#done

input_files=("input1_sorted.mtx")

# Output file for results
output_file="results.txt"

# Loop through each input file
for file in "${input_files[@]}"; do
    # Execute row_per_block
    echo "Executing row_per_block for $file" >> "$output_file"
    ./row_per_block "$file" >> "$output_file"

    # Execute row_per_thread
    echo "Executing row_per_thread for $file" >> "$output_file"
    ./row_per_thread "$file" >> "$output_file"

    # Execute row_per_warp
    echo "Executing row_per_warp for $file" >> "$output_file"
    ./row_per_warp "$file" >> "$output_file"

    echo "------------------------------------" >> "$output_file"
done

echo "Execution completed. Results saved in $output_file"
