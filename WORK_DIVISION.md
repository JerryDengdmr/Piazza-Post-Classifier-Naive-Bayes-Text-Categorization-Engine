# ML Classifier - Work Division

## What's Already Done (Your Part - ~50%)

### ✅ Completed Components:

1. **Project Setup & Basic Structure**
   - All includes and Post struct
   - Main function skeleton with precision setting

2. **Command-Line Parsing**
   - `parse_arguments()` - Handles train-only and train-test modes
   - Validates argument count and extracts file paths

3. **CSV File Reading**
   - `read_csv_file()` - Reads CSV files using csvstream library
   - Error handling for missing files

4. **Word Extraction**
   - `extract_unique_words()` - Splits content into unique words
   - Uses istringstream and set for uniqueness

5. **Training Pipeline**
   - `train()` - Builds vocabulary and counts word occurrences
   - Counts posts per label
   - Tracks word-label associations

6. **Probability Computation**
   - `compute_log_priors()` - Calculates log P(label)
   - `compute_log_likelihoods()` - Calculates log P(word|label)

7. **Train-Only Mode Output (FULLY WORKING)**
   - `print_training_data()` - Prints all training posts
   - `print_training_summary()` - Prints statistics
   - `print_classes()` - Prints label counts and log-priors
   - `print_classifier_parameters()` - Prints word-label parameters

### ✅ Testing Status:
- Train-only mode **FULLY WORKS** and passes all tests
- You can test with: `./classifier.exe train_small.csv`

---

## What Your Teammate Needs to Do (~50%)

### 🔨 Tasks Remaining:

#### 1. Implement `predict_label()` Function
**Location:** Add after `print_classifier_parameters()` function

**Function Signature:**
```cpp
string predict_label(const Post& test_post,
                     const map<string, double>& log_priors,
                     const map<string, map<string, double>>& log_likelihoods,
                     const set<string>& vocabulary,
                     const map<string, int>& word_counts_all_labels,
                     int total_training_posts,
                     double& best_score);
```

**What it should do:**
- Extract unique words from test_post.content using `extract_unique_words()`
- For each label, compute a score starting with log_prior[label]
- For each word in the test post, add to the score:
  - **Case 1:** If word appears with this label in training → add `log_likelihoods[label][word]`
  - **Case 2:** If word is in vocabulary but not with this label → add `log(word_counts_all_labels[word] / total_training_posts)`
  - **Case 3:** If word not in vocabulary → add `log(1.0 / total_training_posts)`
- Return the label with the highest score
- Set `best_score` to the highest score found

**Key points:**
- Initialize `best_score = -numeric_limits<double>::infinity()`
- Use `.count()` to check if keys exist in maps
- Use `log()` function from `<cmath>` for natural logarithm

#### 2. Implement `print_test_results()` Function
**Location:** Add after `predict_label()` function

**Function Signature:**
```cpp
void print_test_results(const vector<Post>& test_posts, 
                        const vector<string>& predictions,
                        const vector<double>& scores);
```

**What it should do:**
- Print blank line then "test data:"
- For each test post (use index i):
  - Print: `"  correct = " << test_posts[i].label << ", predicted = " << predictions[i] << ", log-probability score = " << scores[i]`
  - Print: `"  content = " << test_posts[i].content`
  - Print blank line

#### 3. Implement `print_performance()` Function
**Location:** Add after `print_test_results()` function

**Function Signature:**
```cpp
void print_performance(int correct, int total);
```

**What it should do:**
- Print: `"performance: " << correct << " / " << total << " posts predicted correctly"`

#### 4. Complete the `main()` Function - Train-Test Mode
**Location:** Replace the TODO section in the `else` block (around line 260)

**What to implement:**
```cpp
else {
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
```

---

## Testing Your Work

### Step 1: Compile
```bash
make classifier.exe
```

### Step 2: Test Train-Test Mode
```bash
./classifier.exe train_small.csv test_small.csv > test_output.txt
diff test_output.txt test_small.out.correct
```
If diff shows no output, your implementation is correct!

### Step 3: Run Full Test Suite
```bash
make test
```
All 5 tests should pass.

---

## Reference Files

- **classifier_complete.cpp** - Complete working version (for reference if stuck)
- **test_small.out.correct** - Expected output for small test
- **Design document** - `.kiro/specs/ml-classifier/design.md` has detailed algorithm explanation

---

## Tips

1. **Prediction Algorithm:** The three cases for unknown words are crucial - make sure you check them in order
2. **Precision:** The `cout.precision(3)` is already set in main, so log values will automatically show correctly
3. **Map Access:** Use `.at()` for maps when you know the key exists, `.count()` to check if key exists
4. **Testing:** Test with small dataset first before running large tests

---

## Questions?

If you get stuck:
1. Check the design document for algorithm details
2. Look at the expected output files to understand the format
3. Reference classifier_complete.cpp (but try to implement yourself first!)

Good luck! 🚀
