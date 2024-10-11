import numpy as np
from sklearn.kernel_approximation import Nystroem
import pandas as pd
import argparse


# Step 1: Load the dataset from the txt file
def load_xc_dataset(file_path):
    with open(file_path, 'r') as f:
        # Read the first line which contains the number of datapoints, feature dimensions, and number of labels
        first_line = f.readline().strip()
        num_datapoints, feature_dim, num_labels = map(int, first_line.split())

        data = []
        labels = []

        # Read the rest of the lines for datapoints
        for line in f:
            parts = line.strip().split(' ')
            labels.append(parts[0])  # Get labels (comma-separated string)

            # Sparse format for features (index:value)
            sparse_features = parts[1:]
            dense_features = np.zeros(feature_dim)

            for feat in sparse_features:
                idx, value = map(float, feat.split(':'))
                dense_features[int(idx)] = value  # Convert sparse to dense

            data.append(dense_features)

    return np.array(data), labels, num_datapoints, feature_dim, num_labels


# Step 2: Save the dataset in the original format after applying the polynomial kernel
def save_xc_dataset(file_path, X_poly, labels, num_labels):
    num_datapoints = len(labels)

    with open(file_path, 'w') as f:
        # Write the first line (metadata)
        f.write(f"{num_datapoints} {X_poly.shape[1]} {num_labels}\n")

        # Write each datapoint
        for i in range(num_datapoints):
            # Write labels
            f.write(labels[i])

            # Get non-zero features in the transformed dataset
            sparse_features = []
            for idx, value in enumerate(X_poly[i]):
                if value != 0:  # Only include non-zero features
                    sparse_features.append(f"{idx}:{value:.6f}")

            # Write features in sparse format
            f.write(' ' + ' '.join(sparse_features) + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply polynomial kernel to a sparse dataset.")
    parser.add_argument('input_file', type=str, help="Input file containing the dataset")
    parser.add_argument('output_file', type=str, help="Output file to save the transformed dataset")
    parser.add_argument('train_file', type=str, help="trainfile to fitthe transformed dataset")
    parser.add_argument('--degree', type=int, default=2, help="Degree of the polynomial kernel")
    parser.add_argument('--n_comp', type=int, default=101938, help="n_component term for the polynomial kernel")

    args = parser.parse_args()

    # Step 3: Load dataset
    X, labels, num_datapoints, feature_dim, num_labels = load_xc_dataset(args.input_file)
    Xd, labelsd, num_datapointsd, feature_dimd, num_labelsd = load_xc_dataset(args.train_file)

    # Step 4: Use Nystroem to approximate the polynomial kernel
    nystroem = Nystroem(kernel='poly', degree=args.degree, n_components=args.n_comp)  # Adjust n_components as needed
    nystroem.fit(Xd)
    X_poly = nystroem.transform(X)

    # Step 5: Save the transformed dataset back to txt in the original format
    save_xc_dataset(args.output_file, X_poly, labels, num_labels)

    print(f"Transformed dataset saved as {args.output_file}")
