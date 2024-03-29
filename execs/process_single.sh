#!/bin/bash

input_files=("input15.txt")

# Define the alpha and wedge values
alpha_values=("0.5")
neg_samples=("3")
bloom_filters=("0" "1")
similarities=("0")
l_rates=("0.025")

for input_file in "${input_files[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        for sms in "${similarities[@]}"; do
            for negatives in "${neg_samples[@]}"; do
                for lrates in  "${l_rates[@]}"; do
                    for bloom in "${bloom_filters[@]}"; do
                        if [ "$sms" == "1" ] && [ "$alpha" != "0" ]; then
                            continue  # Skip this iteration if wedge is 1 and alpha is not 0
                        fi

                        # Extract the input file number
                        input_number="${input_file#input}"
                        input_number="${input_number%.txt}"

                        # Generate the output file name
                        output_file="output${input_number}_pos1_alpha${alpha}_neg${negatives}_lr${lrates}_sm${sms}_bloom${bloom}.txt"

                        # Run the command with the specified parameters
                        ./gosh.out --input-graph "$input_file" --output-embedding "$output_file" --directed 0 --epochs 1000 --sampling-algorithm 0 --dimension 1 --learning-rate "$lrates" --negative-samples "$negatives" --alpha "$alpha" --wedge "$sms" --bf "$bloom"
                    done
                done
            done
        done
    done
done
