import numpy as np
import argparse

# Function to parse each line of the dataset
def parse_line(line):
    # Split the line into labels and features
    label_part, feature_part = line.strip().split(' ', 1)

    # Parse labels (comma-separated)
    labels = [int(label) for label in label_part.split(',')]

    # Parse features (sparse format like 0:0.084556)
    features = {}
    for feature in feature_part.split():
        if ':' in feature:
            index, value = feature.split(':')
            features[int(index)] = float(value)

    return labels, features

# Function to apply polynomial kernel
def apply_polynomial_kernel(features, degree=2, coef0=1):
    # Convert the sparse dictionary to a dense feature vector
    max_index = max(features.keys())  # Find the maximum feature index
    feature_vector = np.zeros(max_index + 1)

    # Populate the dense feature vector
    for idx, value in features.items():
        feature_vector[idx] = value

    # Apply the polynomial kernel transformation to each feature
    new_features = {}
    for idx, value in features.items():
        transformed_value = (value + coef0) ** degree  # Polynomial transformation
        if transformed_value != 0:
            new_features[idx] = transformed_value

    return new_features


# Function to reconstruct the data in the same format
def format_output(labels, features):
    # Convert labels back to comma-separated string
    label_str = ','.join(map(str, labels))

    # Convert features back to sparse format like "0:0.084556"
    feature_str = ' '.join(f"{idx}:{value:.6f}" for idx, value in sorted(features.items()))

    return f"{label_str} {feature_str}"

# Main function to process the dataset
def process_dataset(input_file, output_file, degree, coef0):
    dataset = []

    with open(input_file, 'r') as file:
        # Read the first line containing the dataset metadata
        first_line = file.readline().strip()
        num_data_points, num_features, num_labels = map(int, first_line.split())

        # Initialize a list to store the updated dataset
        dataset.append(first_line)  # Keep the first line in the same format

        # Process the remaining lines
        for line in file:
            if line.strip():  # Skip empty lines
                labels, features = parse_line(line)

                # Apply the polynomial kernel transformation
                new_features = apply_polynomial_kernel(features, degree, coef0)

                # Format the output and add to the dataset
                formatted_line = format_output(labels, new_features)
                dataset.append(formatted_line)

    # Save the processed dataset to a new file
    with open(output_file, 'w') as file:
        for line in dataset:
            file.write(line + '\n')

    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Apply polynomial kernel to a sparse dataset.")
    parser.add_argument('input_file', type=str, help="Input file containing the dataset")
    parser.add_argument('output_file', type=str, help="Output file to save the transformed dataset")
    parser.add_argument('--degree', type=int, default=2, help="Degree of the polynomial kernel")
    parser.add_argument('--coef0', type=float, default=1.0, help="Coefficient term for the polynomial kernel")

    # Parse arguments
    args = parser.parse_args()

    # Process the dataset with given arguments
    process_dataset(args.input_file, args.output_file, args.degree, args.coef0)
