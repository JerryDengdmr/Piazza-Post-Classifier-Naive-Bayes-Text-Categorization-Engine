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

// Represents a single post, used to train classifier
struct Post {
    string label;
    string content;
};

 // Represents a Naive-Bayes classifer, trained with Post
struct Classifier {
    map<string, int> label_counts;
    map<string, map<string, int>> word_label_counts;
    set<string> vocabulary;
    map<string, int> word_counts_all_labels;
    map<string, double> log_priors;
    map<string, map<string, double>> log_likelihoods;
};

// REQUIRES: argc & argv command line arguments
// MODIFIES: NONE
// EFFECTS: Sets file name and mode and returnes false if invalid argument
bool parse_arguments(int argc, char* argv[]) {
    if (argc != 2 && argc != 3) {
        cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
        return false;
    }
    return true;
}

string get_train_file(char* argv[]) {
    return argv[1];
}

string get_test_file(int argc, char* argv[]) {
    if (argc == 3) {
        return argv[2];
    }
    return "";
}

bool is_train_only_mode(int argc) {
    return argc == 2;
}

// REQUIRES: Appropriate CSV file with 'tag' and 'content' columns
// MODIFIES: post
// EFFECTS: Read CSV file into vector
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

// REQUIRES: Valid content variable
// MODIFIES: NONE
// EFFECTS: Retures a set of words from content
set<string> extract_unique_words(const string& content) {
    istringstream source(content);
    set<string> words;
    string word;
    
    while (source >> word) {
        words.insert(word);
    }
    
    return words;
}

// REQUIRES: Valid vector training_posts
// MODIFIES: Classifier c
// EFFECTS: Trains the classifier
void train(const vector<Post>& training_posts, Classifier& c) {
    for (const Post& post : training_posts) {
        c.label_counts[post.label]++;
        
        set<string> words = extract_unique_words(post.content);
        
        for (const string& word : words) {
            c.vocabulary.insert(word);
            c.word_label_counts[post.label][word]++;
            c.word_counts_all_labels[word]++;
        }
    }
}

// REQUIRES: Valid c.label_counts with total_posts > 0
// MODIFIES: c.log_priors
// EFFECTS: Computes the log_prior for each label
void compute_log_priors(Classifier& c, int total_posts) {
    for (const auto& pair : c.label_counts) {
        const string& label = pair.first;
        int count = pair.second;
        c.log_priors[label] = 
            log(static_cast<double>(count) / total_posts);
    }
}

// REQUIRES: Valid c.word_label_counts and c.label_counts
// MODIFIES: c.log_likelihoods
// EFFECTS: Computes the log likelihood of each word
void compute_log_likelihoods(Classifier& c) {
    for (const auto& label_pair : c.word_label_counts) {
        const string& label = label_pair.first;
        const map<string, int>& word_counts = label_pair.second;
        int posts_with_label = c.label_counts.at(label);
        
        for (const auto& word_pair : word_counts) {
            const string& word = word_pair.first;
            int count = word_pair.second;
            c.log_likelihoods[label][word] = 
                log(static_cast<double>(count) / posts_with_label);
        }
    }
}

// REQUIRES: Valid posts vector containing Post variables
// MODIFIES: NONE
// EFFECTS: Prints posts with label and content
void print_training_data(const vector<Post>& posts) {
    cout << "training data:" << endl;
    for (const Post& post : posts) {
        cout << "  label = " << post.label
             << ", content = " << post.content << endl;
    }
}

// REQUIRES: 
// MODIFIES: NONE
// EFFECTS: Prints out number of total_posts and vocab_size
void print_training_summary(int total_posts, int vocab_size) {
    cout << "trained on " << total_posts << " examples" << endl;
    cout << "vocabulary size = " << vocab_size << endl;
    cout << endl;
}

// REQUIRES: Valid c.label_counts and c.log_priors variables
// MODIFIES: NONE
// EFFECTS: Prints out each class as well as log prior
void print_classes(const Classifier& c, int total_posts) {
    cout << "classes:" << endl;
    for (const auto& pair : c.label_counts) {
        const string& label = pair.first;
        int count = pair.second;
        double log_prior = c.log_priors.at(label);
        cout << "  " << label << ", " << count
             << " examples, log-prior = " << log_prior << endl;
    }
}

// REQUIRES: Valid c.word_label_counts and c.log_likelihoods
// MODIFIES: NONE
// EFFECTS: Prints the log lilihood and counts for eahc label
void print_classifier_parameters(const Classifier& c) {
    cout << "classifier parameters:" << endl;
    for (const auto& label_pair : c.word_label_counts) {
        const string& label = label_pair.first;
        const map<string, int>& word_counts = label_pair.second;
        
        for (const auto& word_pair : word_counts) {
            const string& word = word_pair.first;
            int count = word_pair.second;
            double log_likelihood = 
                c.log_likelihoods.at(label).at(word);
            cout << "  " << label << ":" << word << ", count = "
                 << count << ", log-likelihood = "
                 << log_likelihood << endl;
        }
    }
    cout << endl;
}

// REQUIRES: Valid classifier c (trained), with post contains valid variables
// MODIFIES: prediction & score
// EFFECTS: Computes log probability and assign score
void predict_label(const Post& post,
                   const Classifier& c,
                   string& prediction,
                   double& score) {
    int total_training_posts = 0;
    for (const auto& label_pair : c.label_counts) {
        total_training_posts += label_pair.second;
    }
    
    score = -1e15;
    
    for (const auto& label_pair : c.label_counts) {
        const string& label = label_pair.first;
        double log_prob = c.log_priors.at(label);
        
        set<string> unique_words = extract_unique_words(post.content);
        
        for (const string& word : unique_words) {
            auto label_it = c.log_likelihoods.find(label);
            
            if (label_it != c.log_likelihoods.end()) {
                auto word_it = label_it->second.find(word);
                
                if (word_it != label_it->second.end()) {
                    log_prob += word_it->second;
                } else {
                    auto all_it = c.word_counts_all_labels.find(word);
                    
                    if (all_it != c.word_counts_all_labels.end()) {
                        log_prob += log(
                            static_cast<double>(all_it->second)
                            / total_training_posts);
                    } else {
                        log_prob += log(1.0 / total_training_posts);
                    }
                }
            } else {
                auto all_it = c.word_counts_all_labels.find(word);
                
                if (all_it != c.word_counts_all_labels.end()) {
                    log_prob += log(
                        static_cast<double>(all_it->second)
                        / total_training_posts);
                } else {
                    log_prob += log(1.0 / total_training_posts);
                }
            }
        }
        
        if (log_prob > score) {
            score = log_prob;
            prediction = label;
        }
    }
}

// REQUIRES: posts, predictions, and scores have same size.
// MODIFIES: NONE
// EFFECTS: Prints out test results
void print_test_results(const vector<Post>& posts,
                        const vector<string>& predictions,
                        const vector<double>& scores) {
    cout << "test data:" << endl;
    
    for (size_t i = 0; i < posts.size(); ++i) {
        cout << "  correct = " << posts[i].label
             << ", predicted = " << predictions[i]
             << ", log-probability score = " << scores[i] << endl;
        cout << "  content = " << posts[i].content << endl;
        cout << endl;
    }
}

// REQUIRES:
// MODIFIES: NONE
// EFFECTS: Prints number of correct predictions
void print_performance(int correct, int total) {
    cout << "performance: " << correct << " / " << total
         << " posts predicted correctly" << endl;
}

int main(int argc, char* argv[]) {
    cout.precision(3);
    
    if (!parse_arguments(argc, argv)) {
        return 1;
    }
    
    string train_file = get_train_file(argv);
    string test_file = get_test_file(argc, argv);
    bool train_only_mode = is_train_only_mode(argc);
    
    vector<Post> training_posts;
    if (!read_csv_file(train_file, training_posts)) {
        return 1;
    }
    
    vector<Post> test_posts;
    if (!train_only_mode) {
        if (!read_csv_file(test_file, test_posts)) {
            return 1;
        }
    }
    
    Classifier c;
    train(training_posts, c);
    
    int total_posts = static_cast<int>(training_posts.size());
    
    compute_log_priors(c, total_posts);
    compute_log_likelihoods(c);
    
    if (train_only_mode) {
        print_training_data(training_posts);
        print_training_summary(total_posts,
                              static_cast<int>(c.vocabulary.size()));
        print_classes(c, total_posts);
        print_classifier_parameters(c);
    } else {
        cout << "trained on " << total_posts << " examples"
             << endl << endl;
        
        vector<string> predictions;
        vector<double> scores;
        int correct_predictions = 0;
        
        for (const Post& post : test_posts) {
            string prediction;
            double score;
            predict_label(post, c, prediction, score);
            predictions.push_back(prediction);
            scores.push_back(score);
            
            if (prediction == post.label) {
                correct_predictions++;
            }
        }
        
        print_test_results(test_posts, predictions, scores);
        print_performance(correct_predictions,
                         static_cast<int>(test_posts.size()));
    }
    
    return 0;
}
