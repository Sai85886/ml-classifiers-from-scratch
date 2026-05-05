import math
from naive_bayes_utils import load_training_set, load_test_set
import matplotlib.pyplot as plt

def compute_word_counts(documents):
    total_words = 0
    word_counts = {}

    for document in documents:
        for word in document:
            total_words += 1
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts, total_words


def find_class_probabilities(test_doc, prior_probabilities, word_class_counts, total_words, len_vocab, alpha = 0):
    probabilities = {}

    for class_label, prior_probability in prior_probabilities.items():
        probability = prior_probability

        for word in test_doc:
            probability = probability * (word_class_counts.get(class_label, {}).get(word, 0) + alpha) / (total_words.get(class_label, 0) + alpha * len_vocab)
        probabilities[class_label] = probability

    return probabilities


def find_log_class_probabilities(test_doc, prior_probabilities, word_class_counts, total_words, len_vocab, alpha = 0):
    log_probabilities = {}

    for class_label, prior_probability in prior_probabilities.items():
        log_probability = math.log(prior_probability)

        for word in test_doc:
            log_probability += math.log(word_class_counts.get(class_label, {}).get(word, 0) + alpha) - math.log(total_words.get(class_label, 0) + alpha * len_vocab)
        log_probabilities[class_label] = log_probability

    return log_probabilities


def evaluate_metrics(labels_true, labels_pred):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(labels_true)):
        if labels_true[i] == 'pos' and labels_pred[i] == 'pos':
            true_positives += 1
        elif labels_true[i] == 'neg' and labels_pred[i] == 'neg':
            true_negatives += 1
        elif labels_true[i] == 'pos' and labels_pred[i] == 'neg':
            false_negatives += 1
        elif labels_true[i] == 'neg' and labels_pred[i] == 'pos':
            false_positives += 1
            
    
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    confusion_matrix = {
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
    return accuracy, precision, recall, confusion_matrix


def run_experiment(percentage_positive_instances_train, percentage_negative_instances_train, percentage_positive_instances_test, percentage_negative_instances_test, alpha, log = True):
    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    len_vocab = len(vocab)

    # Compute prior class probabilities
    pos_prior_probabilities = len(pos_train) / (len(pos_train) + len(neg_train))
    neg_prior_probabilities = len(neg_train) / (len(pos_train) + len(neg_train))
    
    # Compute word probabilities
    pos_word_counts, pos_total_words = compute_word_counts(pos_train)
    neg_word_counts, neg_total_words = compute_word_counts(neg_train)

    prior_probabilities = {
        'pos': pos_prior_probabilities,
        'neg': neg_prior_probabilities
    }

    word_class_counts = {
        'pos': pos_word_counts,
        'neg': neg_word_counts
    }

    total_words = {
        'pos': pos_total_words,
        'neg': neg_total_words
    }

    labels_true = []
    labels_pred = []

    for doc in pos_test:
        if log:
            probabilities = find_log_class_probabilities(doc, prior_probabilities, word_class_counts, total_words, len_vocab, alpha)
        else:
            probabilities = find_class_probabilities(doc, prior_probabilities, word_class_counts, total_words, len_vocab, alpha)
        labels_true.append('pos')
        labels_pred.append(max(probabilities, key=probabilities.get))

    for doc in neg_test:
        if log:
            probabilities = find_log_class_probabilities(doc, prior_probabilities, word_class_counts, total_words, len_vocab, alpha)
        else:
            probabilities = find_class_probabilities(doc, prior_probabilities, word_class_counts, total_words, len_vocab, alpha)
        labels_true.append('neg')
        labels_pred.append(max(probabilities, key=probabilities.get))

    accuracy, precision, recall, confusion_matrix = evaluate_metrics(labels_true, labels_pred)

    return accuracy, precision, recall, confusion_matrix


# if __name__ == '__main__':
#     ## Question 1
#     accuracy, precision, recall, confusion_matrix = run_experiment(0.2, 0.2, 0.2, 0.2, 0, False)
#     print(f"Question 1: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Confusion Matrix: {confusion_matrix}")


#     ## Question 2
#     accuracy, precision, recall, confusion_matrix = run_experiment(0.2, 0.2, 0.2, 0.2, 1)
#     print(f"Question 2: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Confusion Matrix: {confusion_matrix}")


#     alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
#     accuracies = []
#     for alpha in alphas:
#         accuracy, precision, recall, confusion_matrix = run_experiment(0.2, 0.2, 0.2, 0.2, alpha)
#         accuracies.append(accuracy)
#         print(f"Alpha: {alpha}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Confusion Matrix: {confusion_matrix}")

#     plt.plot(alphas, accuracies)
#     plt.xscale('log')
#     plt.xlabel('Alpha')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy vs Alpha')
#     plt.savefig('HW2_2.accuracy_vs_alpha.png')

#     best_alpha = alphas[accuracies.index(max(accuracies))]
#     print(f"Best alpha: {best_alpha}")

#     ## Question 3
#     accuracy, precision, recall, confusion_matrix = run_experiment(1.0, 1.0, 1.0, 1.0, best_alpha)
#     print(f"Question 3: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Confusion Matrix: {confusion_matrix}")

#     ## Question 4
#     accuracy, precision, recall, confusion_matrix = run_experiment(0.3, 0.3, 1.0, 1.0, best_alpha)
#     print(f"Question 4: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Confusion Matrix: {confusion_matrix}")

#     ## Question 6
#     accuracy, precision, recall, confusion_matrix = run_experiment(0.1, 0.5, 1.0, 1.0, best_alpha)
#     print(f"Question 6: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Confusion Matrix: {confusion_matrix}")
