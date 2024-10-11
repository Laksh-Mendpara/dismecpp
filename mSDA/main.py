import numpy as np
import argparse
from gensim.corpora import Dictionary
from linear_msda import mSDA
from collections import Counter


# Step 1: Load the dataset from the txt file in EuREX-4k format
def load_xc_dataset(file_path):
    print(f"Loading dataset from {file_path}")
    
    with open(file_path, 'r') as f:
        # Read the first line which contains the number of datapoints, feature dimensions, and number of labels
        first_line = f.readline().strip()
        num_datapoints, feature_dim, num_labels = map(int, first_line.split())
        
        print(f"File Metadata - Num Datapoints: {num_datapoints}, Feature Dim: {feature_dim}, Num Labels: {num_labels}")
        
        data = []
        labels = []

        # Track feature frequencies
        feature_counter = Counter()

        # Read the rest of the lines for datapoints
        for idx, line in enumerate(f):
            parts = line.strip().split(' ')
            labels.append(parts[0])  # Get labels (comma-separated string)

            # Sparse format for features (index:value)
            sparse_features = parts[1:]
            bow_features = [(int(feat.split(':')[0]), float(feat.split(':')[1])) for feat in sparse_features]

            # Count feature frequencies
            for feat in bow_features:
                feature_counter[feat[0]] += 1

            data.append(bow_features)

            # if idx < 5:
            #     print(f"Sample datapoint {idx}: Labels: {labels[-1]}, Features: {bow_features}")


    print(f"Finished loading {num_datapoints} datapoints from {file_path}")
    print(f"Total unique features: {len(feature_counter)}")
    
    return data, labels, num_datapoints, feature_dim, num_labels, feature_counter


# Step 2: Save the transformed dataset
def save_xc_dataset(file_path, representations, labels, num_labels):
    num_datapoints = len(labels)
    print(f"Saving dataset to {file_path}")

    with open(file_path, 'w') as f:
        # Write the first line (metadata)
        f.write(f"{num_datapoints} {len(representations[0])} {num_labels}\n")
        print(f"Dataset metadata saved: {num_datapoints} datapoints, {len(representations[0])} feature dimensions, {num_labels} labels")

        # Write each datapoint
        for i in range(num_datapoints):
            # Write labels
            f.write(labels[i])

            # Get non-zero features in the transformed dataset
            sparse_features = []
            for idx, value in representations[i]:
                if value != 0:
                    sparse_features.append(f"{idx}:{value:.6f}")  # Fixed: write value instead of index

            # Write features in sparse format
            f.write(' ' + ' '.join(sparse_features) + '\n')

            # Debug: print the first few representations
            # if i < 5:
            #     print(f"Sample transformed datapoint {i}: Labels: {labels[i]}, Features: {sparse_features}")

    print(f"Finished saving dataset to {file_path}")


# Step 3: Train mSDA and transform data
def apply_msda(train_data, test_data, feature_dim, dimensions, feature_counter):
    print(f"Applying mSDA with {dimensions} dimensions")

    # Create a list of prototype column indices (for the mSDA model)
    # Select the top 'dimensions' most frequent features as prototypes
    prototype_ids = [feat[0] for feat in feature_counter.most_common(dimensions)]
    print(f"Selected top {dimensions} frequent features as prototypes: {prototype_ids[:10]} ... (truncated)")

    # Initialize mSDA model
    print("Initializing mSDA")
    msda = mSDA(noise=0.6, num_layers=4, input_dimensionality=feature_dim, output_dimensionality=dimensions, prototype_ids=prototype_ids)

    # Train the model using the training data
    print("Training mSDA model")
    msda.train(train_data, chunksize=10000)
    print("Finished training mSDA model")

    # Transform both training and test data (convert generator to list)
    print("Transforming training and test data")
    train_representations = list(msda[train_data])
    test_representations = list(msda[test_data])
    print("Data transformation complete")

    # Debug: print the first few transformed data points
    # for i in range(min(5, len(train_representations))):
    #     print(f"Sample transformed training data {i}: {train_representations[i]}")

    return train_representations, test_representations


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply mSDA to a sparse dataset in EuREX-4k format.")
    parser.add_argument('input_train_file', type=str, help="Input file containing the train dataset")
    parser.add_argument('input_test_file', type=str, help="Input file containing the test dataset")
    parser.add_argument('output_train_file', type=str, help="Output file to save the train transformed dataset")
    parser.add_argument('output_test_file', type=str, help="Output file to save the test transformed dataset")
    parser.add_argument('--dim', type=int, default=1000, help="Number of dimensions for mSDA")

    args = parser.parse_args()

    # Step 4: Load dataset and compute feature frequencies
    train_data, labels_train, num_datapoints_train, feature_dim_train, num_labels_train, feature_counter_train = load_xc_dataset(args.input_train_file)
    print('Train dataset loaded')

    test_data, labels_test, num_datapoints_test, feature_dim_test, num_labels_test, feature_counter_test = load_xc_dataset(args.input_test_file)
    print('Test dataset loaded')

    assert num_labels_test == num_labels_train
    assert feature_dim_test == feature_dim_train

    # Combine feature counters from both train and test for global frequency
    feature_counter_train.update(feature_counter_test)
    print("Updated feature frequencies from test dataset")

    # Step 5: Apply mSDA using the top frequent features
    print(f"Applying mSDA with {args.dim} dimensions")
    train_representations, test_representations = apply_msda(train_data, test_data, feature_dim_train, args.dim, feature_counter_train)
    # print(type(train_representations))
    # print (train_representations[0].shape)


    # Step 6: Save transformed datasets
    save_xc_dataset(args.output_train_file, train_representations, labels_train, num_labels_train)
    print(f"Train transformed dataset saved as {args.output_train_file}")

    save_xc_dataset(args.output_test_file, test_representations, labels_test, num_labels_test)
    print(f"Test transformed dataset saved as {args.output_test_file}")
