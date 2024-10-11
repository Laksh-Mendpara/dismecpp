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


# Step 2: Save the dataset in the original format after applying the RBF kernel
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

    parser = argparse.ArgumentParser(description="Apply RBF kernel to a sparse dataset.")
    parser.add_argument('input_train_file', type=str, help="Input file containing the train dataset")
    parser.add_argument('input_test_file', type=str, help="Input file containing the test dataset")
    parser.add_argument('output_train_file', type=str, help="Output file to save the train transformed dataset")
    parser.add_argument('output_test_file', type=str, help="Output file to save the test transformed dataset")
    parser.add_argument('--gamma', type=float, default=1.0, help="Gamma parameter for the RBF kernel")
    parser.add_argument('--n_comp', type=int, default=101938, help="Number of components for the Nystroem approximation")

    args = parser.parse_args()

    # Step 3: Load dataset
    X_train, labels_train, num_datapoints_train, feature_dim_train, num_labels_train = load_xc_dataset(args.input_train_file)
    print('train dataset loaded')
    
    X_test, labels_test, num_datapoints_test, feature_dim_test, num_labels_test = load_xc_dataset(args.input_test_file)
    print('test dataset loaded')

    assert(num_labels_test == num_labels_train)
    assert(feature_dim_test == feature_dim_train)

    # Step 4: Use Nystroem to approximate the RBF kernel
    nystroem = Nystroem(kernel='rbf', gamma=args.gamma, n_components=args.n_comp)
    nystroem.fit(X_train)
    print('fitted on train_data')

    X_train_transformed = nystroem.transform(X_train)
    print('transformed train data')
    
    X_test_transformed = nystroem.transform(X_test)
    print('transformed test data')

    save_xc_dataset(args.output_train_file, X_train_transformed, labels_train, num_labels_train)
    print(f"Train transformed dataset saved as {args.output_train_file}")

    save_xc_dataset(args.output_test_file, X_test_transformed, labels_test, num_labels_test)
    print(f"Tset transformed dataset saved as {args.output_test_file}")
