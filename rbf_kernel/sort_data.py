import argparse

def sort_dataset(input_file_path, output_file_path):
    # Initialize a list to hold the sorted data
    sorted_data = []

    # Read the file and process the data
    with open(input_file_path, 'r') as file:
        # Read the first line for the number of datapoints
        num_datapoints = int(file.readline().strip().split()[0])

        # Process each line of data
        for _ in range(num_datapoints):
            line = file.readline().strip()
            if line:
                # Split the line into label-score pairs
                pairs = line.split()
                # Convert to a list of tuples (label, score)
                tuples = [tuple(map(float, pair.split(':'))) for pair in pairs]
                # Sort the tuples by the label (first element)
                sorted_tuples = sorted(tuples, key=lambda x: x[0])
                # Format sorted tuples back to the string format
                sorted_line = ' '.join(f"{int(label)}:{score:.6f}" for label, score in sorted_tuples)
                sorted_data.append(sorted_line)

    # Write the sorted dataset to a new file
    with open(output_file_path, 'w') as output_file:
        output_file.write(f"{num_datapoints} 30938\n")  # Hardcoded '30938'
        for sorted_line in sorted_data:
            output_file.write(sorted_line + '\n')

    print(f"Sorted dataset saved to {output_file_path}.")

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Sort labels for each datapoint in a dataset.")
    parser.add_argument('input_file', type=str, help='Path to the input dataset file')
    parser.add_argument('output_file', type=str, help='Path to the output file where sorted dataset will be saved')

    # Get the arguments
    args = parser.parse_args()

    # Call the sort function with the provided arguments
    sort_dataset(args.input_file, args.output_file)

