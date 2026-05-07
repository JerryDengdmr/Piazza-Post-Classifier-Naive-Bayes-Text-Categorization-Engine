// classifier.cpp
// Machine Learning Classifier using Naive Bayes

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>
#include <iomanip>
#include <cmath>
#include "csvstream.hpp"

using namespace std;

// EFFECTS: Represents a single post with label and content
struct Post {
    string label;
    string content;
};

// EFFECTS: Parses command-line arguments and validates them
// MODIFIES: train_file, test_file, train_only_mode
// Returns true if arguments are valid, false otherwise
bool parse_arguments(int argc, char* argv[], 
                     string& train_file, string& test_file, bool& train_only_mode) {
    if (argc != 2 && argc != 3) {
        cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
        return false;
    }
    
    train_file = argv[1];
    
    if (argc == 2) {
        train_only_mode = true;
        test_file = "";
    } else {
        train_only_mode = false;
        test_file = argv[2];
    }
    
    return true;
}

// EFFECTS: Reads CSV file and extracts posts
// MODIFIES: posts
// Returns true if successful, false on error
bool read_csv_file(const string& filename, vector<Post>& posts) {
    try {
        csvstream csvin(filename);
        map<string, string> row;
        
        while (csvin >> row) {
            Post post;
            post.label = row["tag"];
            post.content = row["content"];
            posts.push_back(post);
        }
        
        return true;
    } catch (const csvstream_exception& e) {
        cout << "Error opening file: " << filename << endl;
        return false;
    }
}

// EFFECTS: Extracts unique words from content string
// Returns set of unique whitespace-delimited words
set<string> extract_unique_words(const string& content) {
    istringstream source(content);
    set<string> words;
    string word;
    
    while (source >> word) {
        words.insert(word);
    }
    
    return words;
}

// EFFECTS: Trains classifier on training posts
// MODIFIES: label_counts, word_label_counts, vocabulary, word_counts_all_labels
void train(const vector<Post>& training_posts,
           map<string, int>& label_counts,
           map<string, map<string, int>>& word_label_counts,
           set<string>& vocabulary,
           map<string, int>& word_counts_all_labels) {
    
    for (const Post& post : training_posts) {
        // Count posts per label
        label_counts[post.label]++;
        
        // Extract unique words from this post
        set<string> words = extract_unique_words(post.content);
        
        // Add words to vocabulary and count occurrences
        for (const string& word : words) {
            vocabulary.insert(word);
            word_label_counts[post.label][word]++;
            word_counts_all_labels[word]++;
        }
    }
}

// EFFECTS: Computes log-prior probabilities for each label
// Returns map of label to log-prior probability
map<string, double> compute_log_priors(const map<string, int>& label_counts, 
                                        int total_posts) {
    map<string, double> log_priors;
    
    for (const auto& pair : label_counts) {
        const string& label = pair.first;
        int count = pair.second;
        log_priors[label] = log(static_cast<double>(count) / total_posts);
    }
    
    return log_priors;
}

// EFFECTS: Computes log-likelihood probabilities for each word-label pair
// Returns nested map of label to word to log-likelihood probability
map<string, map<string, double>> compute_log_likelihoods(
    const map<string, map<string, int>>& word_label_counts,
    const map<string, int>& label_counts) {
    
    map<string, map<string, double>> log_likelihoods;
    
    for (const auto& label_pair : word_label_counts) {
        const string& label = label_pair.first;
        const map<string, int>& word_counts = label_pair.second;
        int posts_with_label = label_counts.at(label);
        
        for (const auto& word_pair : word_counts) {
            const string& word = word_pair.first;
            int count = word_pair.second;
            log_likelihoods[label][word] = 
                log(static_cast<double>(count) / posts_with_label);
        }
    }
    
    return log_likelihoods;
}

// EFFECTS: Prints all training data posts
void print_training_data(const vector<Post>& posts) {
    cout << "training data:" << endl;
    for (const Post& post : posts) {
        cout << "  label = " << post.label << ", content = " << post.content << endl;
    }
}

// EFFECTS: Prints training summary statistics
void print_training_summary(int total_posts, int vocab_size) {
    cout << "trained on " << total_posts << " examples" << endl;
    cout << "vocabulary size = " << vocab_size << endl;
    cout << endl;
}

// EFFECTS: Prints class statistics with log-priors
void print_classes(const map<string, int>& label_counts, 
                   const map<string, double>& log_priors, 
                   int total_posts) {
    cout << "classes:" << endl;
    for (const auto& pair : label_counts) {
        const string& label = pair.first;
        int count = pair.second;
        double log_prior = log_priors.at(label);
        cout << "  " << label << ", " << count << " examples, log-prior = " 
             << log_prior << endl;
    }
}

// EFFECTS: Prints classifier parameters (word-label counts and log-likelihoods)
void print_classifier_parameters(const map<string, map<string, int>>& word_label_counts,
                                 const map<string, map<string, double>>& log_likelihoods) {
    cout << "classifier parameters:" << endl;
    for (const auto& label_pair : word_label_counts) {
        const string& label = label_pair.first;
        const map<string, int>& word_counts = label_pair.second;
        
        for (const auto& word_pair : word_counts) {
            const string& word = word_pair.first;
            int count = word_pair.second;
            double log_likelihood = log_likelihoods.at(label).at(word);
            cout << "  " << label << ":" << word << ", count = " << count 
                 << ", log-likelihood = " << log_likelihood << endl;
        }
    }
    cout << endl;
}

// EFFECTS: Predicts label for a test post using Naive Bayes
// MODIFIES: best_score
// Returns predicted label
string predict_label(const Post& test_post,
                     const map<string, double>& log_priors,
                     const map<string, map<string, double>>& log_likelihoods,
                     const set<string>& vocabulary,
                     const map<string, int>& word_counts_all_labels,
                     int total_training_posts,
                     double& best_score) {
    
    string best_label;
    best_score = -numeric_limits<double>::infinity();
    
    // Extract unique words from test post
    set<string> test_words = extract_unique_words(test_post.content);
    
    // Try each label and compute score
    for (const auto& label_pair : log_priors) {
        const string& label = label_pair.first;
        double score = log_priors.at(label);
        
        // Add log-likelihood for each word in test post
        for (const string& word : test_words) {
            // Case 1: Word appears with this label in training
            if (log_likelihoods.count(label) && 
                log_likelihoods.at(label).count(word)) {
                score += log_likelihoods.at(label).at(word);
            }
            // Case 2: Word in vocabulary but not with this label
            else if (vocabulary.count(word)) {
                score += log(static_cast<double>(word_counts_all_labels.at(word)) / 
                            total_training_posts);
            }
            // Case 3: Word not in vocabulary
            else {
                score += log(1.0 / total_training_posts);
            }
        }
        
        // Update best label if this score is higher
        if (score > best_score) {
            best_score = score;
            best_label = label;
        }
    }
    
    return best_label;
}

// EFFECTS: Prints test results with predictions
void print_test_results(const vector<Post>& test_posts, 
                        const vector<string>& predictions,
                        const vector<double>& scores) {
    cout << endl << "test data:" << endl;
    for (size_t i = 0; i < test_posts.size(); ++i) {
        cout << "  correct = " << test_posts[i].label 
             << ", predicted = " << predictions[i] 
             << ", log-probability score = " << scores[i] << endl;
        cout << "  content = " << test_posts[i].content << endl;
        cout << endl;
    }
}

// EFFECTS: Prints performance statistics
void print_performance(int correct, int total) {
    cout << "performance: " << correct << " / " << total 
         << " posts predicted correctly" << endl;
}

int main(int argc, char* argv[]) {
    cout.precision(3);
    
    string train_file, test_file;
    bool train_only_mode;
    
    if (!parse_arguments(argc, argv, train_file, test_file, train_only_mode)) {
        return 1;
    }
    
    // Read training data
    vector<Post> training_posts;
    if (!read_csv_file(train_file, training_posts)) {
        return 1;
    }
    
    // Read test data if in train-test mode
    vector<Post> test_posts;
    if (!train_only_mode) {
        if (!read_csv_file(test_file, test_posts)) {
            return 1;
        }
    }
    
    // Train the classifier
    map<string, int> label_counts;
    map<string, map<string, int>> word_label_counts;
    set<string> vocabulary;
    map<string, int> word_counts_all_labels;
    
    train(training_posts, label_counts, word_label_counts, 
          vocabulary, word_counts_all_labels);
    
    int total_posts = static_cast<int>(training_posts.size());
    
    // Compute log-probabilities
    map<string, double> log_priors = compute_log_priors(label_counts, total_posts);
    map<string, map<string, double>> log_likelihoods = 
        compute_log_likelihoods(word_label_counts, label_counts);
    
    // Output based on mode
    if (train_only_mode) {
        print_training_data(training_posts);
        print_training_summary(total_posts, static_cast<int>(vocabulary.size()));
        print_classes(label_counts, log_priors, total_posts);
        print_classifier_parameters(word_label_counts, log_likelihoods);
    } else {
        // Train-test mode
        cout << "trained on " << total_posts << " examples" << endl;
        
        // Make predictions for test posts
        vector<string> predictions;
        vector<double> scores;
        int correct_predictions = 0;
        
        for (const Post& test_post : test_posts) {
            double score;
            string predicted_label = predict_label(test_post, log_priors, 
                                                   log_likelihoods, vocabulary,
                                                   word_counts_all_labels, 
                                                   total_posts, score);
            predictions.push_back(predicted_label);
            scores.push_back(score);
            
            if (predicted_label == test_post.label) {
                correct_predictions++;
            }
        }
        
        // Print results
        print_test_results(test_posts, predictions, scores);
        print_performance(correct_predictions, static_cast<int>(test_posts.size()));
    }
    
    return 0;
}
