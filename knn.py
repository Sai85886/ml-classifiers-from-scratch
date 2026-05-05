import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

class kNN:
    df = None
    features_test = None
    features_train = None
    labels_test = None
    labels_train = None

    # Load and shuffle dataset
    def init_dataset(self, dataset):
        df = pd.read_csv(dataset)
        shuffled_df = sklearn.utils.shuffle(df)
        self.df = shuffled_df

    # Split shuffled dataset into training, test
    def split_test_train(self, test_size):
        features = self.df.iloc[:, :-1]
        labels = self.df.iloc[:, -1]
        features_train, features_test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, test_size=test_size, stratify=labels)
        self.features_train = features_train
        self.features_test = features_test
        self.labels_train = labels_train
        self.labels_test = labels_test

    # Normalize feature data
    def normalize_feature_data(self):
        min_values = self.features_train.min()
        max_values = self.features_train.max()
        features_train_normalized = (self.features_train - min_values) / (max_values - min_values)
        features_test_normalized = (self.features_test - min_values) / (max_values - min_values)
        self.features_train = features_train_normalized
        self.features_test = features_test_normalized

    # Compute euclidean distances and get k nearest neighbors and their labels, return majority label among neighbors
    def predict_using_knn(self, test_instance, k):
        # compute euclidean distances
        distances = []
        for i in range(len(self.features_train)):
            distance = np.sqrt(np.sum((test_instance - self.features_train.iloc[i]) ** 2))
            distances.append((i, distance))

        # sort distances and get k nearest neighbors
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:k]
        
        # get k nearest neighbors' labels
        labels = []
        unique_labels = set()
        for neighbor in neighbors:
            label = self.labels_train.iloc[neighbor[0]]
            labels.append(label)
            unique_labels.add(label)
        
        # return majority label among neighbors
        label_counts = {}
        for i in labels:
            if i in label_counts:
                label_counts[i] += 1
            else:
                label_counts[i] = 1
    
        majority_label = -1
        majority_label_count = 0
        for label, label_count in label_counts.items():
            if majority_label_count < label_count:
                majority_label = label
                majority_label_count = label_count

        return majority_label

    # Determine accuracy of kNN predictions on given features and labels for a given k value
    def determine_accuracy(self, features, labels, k):
        correct_predictions = 0
        for i in range(len(features)):
            test_instance = features.iloc[i]
            true_label = labels.iloc[i]
            predicted_label = self.predict_using_knn(test_instance, k)
            if predicted_label == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(features)
        return accuracy
    

knn = kNN()

# train_avg_accuracies = [] # per k
# train_std_deviations = [] # per k

# test_avg_accuracies = []  # per k
# test_std_deviations = []  # per k

# for k in range(1, 52, 2):
#     train_accuracies = [] # per iteration
#     test_accuracies = []  # per iteration
    
#     for iter in range(1, 20): 
#         # Load wdbc dataset
#         knn.init_dataset('datasets/wdbc.csv')

#         # Split dataset into training, test
#         knn.split_test_train(0.2)

#         # Normalize features
#         knn.normalize_feature_data()
        
#         # Evaluate accuracy on training set
#         train_accuracy = knn.determine_accuracy(knn.features_train, knn.labels_train, k)
#         train_accuracies.append(train_accuracy)

#         # Evaluate accuracy on test set
#         test_accuracy = knn.determine_accuracy(knn.features_test, knn.labels_test, k)
#         test_accuracies.append(test_accuracy)
    
#     # Compute average accuracy and standard deviation for training set for a given k value
#     train_avg_accuracy = np.mean(train_accuracies)
#     train_std_deviation = np.std(train_accuracies, ddof=1)
#     train_avg_accuracies.append((k, train_avg_accuracy))
#     train_std_deviations.append((k, train_std_deviation))

#     print(f'Average accuracy on training set with k = {k} is {train_avg_accuracy}')
#     print(f'Standard deviation of training accuracies with k = {k} is {train_std_deviation}')
    
#     # Compute average accuracy and standard deviation for test set for a given k value
#     test_avg_accuracy = np.mean(test_accuracies)
#     test_std_deviation = np.std(test_accuracies, ddof=1)
#     test_avg_accuracies.append((k, test_avg_accuracy))
#     test_std_deviations.append((k, test_std_deviation))

#     print(f'Average accuracy on test set with k = {k} is {test_avg_accuracy}')
#     print(f'Standard deviation of test accuracies with k = {k} is {test_std_deviation}')

# # 1.1 Construct graph for training accuracies vs k values with error bars
# x_values = [x[0] for x in train_avg_accuracies]
# y_values = [x[1] for x in train_avg_accuracies]
# yerr_values = [x[1] for x in train_std_deviations]
# plt.plot(x_values, y_values, linestyle='-', marker='o')
# plt.errorbar(x_values, y_values, yerr=yerr_values, fmt='o')
# plt.xlabel('k values')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy vs k values with error bars')
# plt.savefig("figures/1.training_accuracy_vs_k.png", dpi=300, bbox_inches="tight")
# plt.show()

# # 1.2 Construct graph for test accuracies vs k values with error bars
# x_values = [x[0] for x in test_avg_accuracies]
# y_values = [x[1] for x in test_avg_accuracies]
# yerr_values = [x[1] for x in test_std_deviations]
# plt.plot(x_values, y_values, linestyle='-', marker='o')
# plt.errorbar(x_values, y_values, yerr=yerr_values, fmt='o')
# plt.xlabel('k values')
# plt.ylabel('Accuracy')
# plt.title('Test Accuracy vs k values with error bars')
# plt.savefig("figures/1.test_accuracy_vs_k.png", dpi=300, bbox_inches="tight")
# plt.show()

# # 1.6 With out feature normalization, construct graph for test accuracies vs k values with error bars
# test_avg_accuracies = []  # per k
# test_std_deviations = []  # per k

# for k in range(1, 52, 2):
#     test_accuracies = []  # per iteration
    
#     for iter in range(1, 20): 
#         # Load wdbc dataset
#         knn.init_dataset('datasets/wdbc.csv')

#         # Split dataset into training, test
#         knn.split_test_train(0.2)

#         # Evaluate accuracy on test set
#         test_accuracy = knn.determine_accuracy(knn.features_test, knn.labels_test, k)
#         test_accuracies.append(test_accuracy)
    
#     # Compute average accuracy and standard deviation for test set for a given k value
#     test_avg_accuracy = np.mean(test_accuracies)
#     test_std_deviation = np.std(test_accuracies, ddof=1)
#     test_avg_accuracies.append((k, test_avg_accuracy))
#     test_std_deviations.append((k, test_std_deviation))

#     print(f'Average accuracy on test set with k = {k} is {test_avg_accuracy}')
#     print(f'Standard deviation of test accuracies with k = {k} is {test_std_deviation}')

# # Construct graph for test accuracies vs k values with error bars
# x_values = [x[0] for x in test_avg_accuracies]
# y_values = [x[1] for x in test_avg_accuracies]
# yerr_values = [x[1] for x in test_std_deviations]
# plt.plot(x_values, y_values, linestyle='-', marker='o')
# plt.errorbar(x_values, y_values, yerr=yerr_values, fmt='o')
# plt.xlabel('k values')
# plt.ylabel('Accuracy')
# plt.title('Test Accuracy vs k values with error bars')
# plt.savefig("figures/1.test_accuracy_vs_k_without_feature_normalization.png", dpi=300, bbox_inches="tight")
# plt.show()
