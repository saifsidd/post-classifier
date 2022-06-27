// Project UID db1f506d06d84ab787baf250c265e24e
#include "csvstream.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <set>
#include <map>
using namespace std;


class Classifier {
private:
    int numPosts;
    int numWords;
    set<string> uniqueWords;
    map<string, int> wordFreq;
    map<string, int> labelFreq;
    map<pair<string, string>, int> wordLabelFreq;
    vector< pair<string, string> > trainingSet;

    // Requires str is valid
    // Modifies uniqueWords, wordFreq
    // Updates uniqueWords so all unique words in str are added to uniqueWords
    void initSet(vector<string> &words);

    // Requires str is valid
    // Updates wordFreq to create a map with a word corresponding to its frequency
    void initWordFreq(vector<string> &words);

    // Requires str is valid
    // Updates labelFreq to create a map with a word corresponding to its frequency
    void initLabFreq(vector<string> &words);

    // Requires that label and word are valid strings
    // Updates wordLabelFreq to include the frequency of every word under every label
    void initWordLabFreq(string &label, string &words);

    // Requires that str is valid
    // Splits str based off white space and appends words
    vector<string> splitString(string &str);

    // Returns true if x < y
    bool less(double x, double y);

public:
    Classifier()
    : numPosts(0), numWords(0) {}
    // Requres "is" is a valid stream
    // Modifies trainingSet, numPosts, numWords, uniqueWords, wordFreq, labelFreq,
    // some other shit
    // Updates trainingSet to include a pair of a post and its respective tag
    void initTrainingSet(istream &is, bool debug);

    pair<string,double> predict(string &str, bool debug);

};


//////////////// IMPLEMENTATION OF CLASSIFIER CLASS ////////////////

pair<string,double> Classifier::predict(string &str, bool debug) {
    map<string, double> probs;
    vector<string> temp = splitString(str);
    set<string> words;
    for (const auto &w : temp) {
        words.insert(w);
    }

    string max;
    for (const auto &kv : labelFreq) {
        probs[kv.first] = log(kv.second / (static_cast<double>(numPosts)));
        for (const auto &w  : words) {
            if (wordLabelFreq[{kv.first, w}] != 0) {
                probs[kv.first] += 
                log(
                    wordLabelFreq[{kv.first, w}] 
                    / static_cast<double>(kv.second)
                );
            }
            else if (wordFreq[w] != 0) {
                probs[kv.first] += 
                log(
                    wordFreq[w] / static_cast<double>(numPosts)
                );
            }
            else {
                probs[kv.first] += 
                log(
                    1 / static_cast<double>(numPosts)
                );               
            }
        }
        max = kv.first;
    }

    for (const auto &kv : probs) {
        if (kv.second > probs[max]) {
            max = kv.first;
        }
    }

    return { max, probs[max] };

}

void Classifier::initTrainingSet(istream &is, bool debug) {
    csvstream csvin(is);
    map<string, string> row;
    if (debug) {
        cout << "training data:" << endl;
    }
    while(csvin >> row) {
        if (debug) {
            cout << "  label = " << row["tag"] << ", content = " 
            << row["content"] << endl;
        }
        vector<string> temp = splitString(row["content"]);
        vector<string> tag = splitString(row["tag"]);
        trainingSet.push_back( { row["tag"], row["content"] } );
        initSet(temp);
        initWordFreq(temp);
        initLabFreq( tag );
        initWordLabFreq(row["tag"], row["content"]);
        ++numPosts;
    }
    cout << "trained on " << numPosts << " examples" << endl;
    if (debug) {
        cout << "vocabulary size = " << uniqueWords.size() << endl << endl;
        cout << "classes:" << endl;
        for (const auto &kv : labelFreq) {
            cout << "  " << kv.first << ", " << kv.second << " examples, log-prior = " <<
            log(kv.second / (static_cast<double>(numPosts))) << endl;
        }
        cout << "classifier parameters:" << endl;
        for (const auto &kv : labelFreq) {
            for (const auto &kvWF: wordFreq) {
                if (wordLabelFreq[ {kv.first, kvWF.first} ] != 0) {
                    cout << "  " << kv.first << ":" << kvWF.first << ", count = " <<
                    wordLabelFreq[ {kv.first, kvWF.first} ] << ", log-likelihood = "
                    << log(wordLabelFreq[{kv.first, kvWF.first}]
                     / static_cast<double>(kv.second)) << endl;
                }
            }
        }
        cout << endl;
    }
}

void Classifier::initSet(vector<string> &words) {
    for (int i = 0; i < static_cast<int>(words.size()); ++i) {
        uniqueWords.insert(words[i]);
    }
    numWords = uniqueWords.size();
}

void Classifier::initWordFreq(vector<string> &words) {
    set<string> nonDupe;
    for (const auto &w : words) {
        nonDupe.insert(w);
    }
    for (const auto &w : nonDupe) {
        wordFreq[w] += 1;
    }
}

void Classifier::initLabFreq(vector<string> &words) {
    for (const auto &w : words) {
        labelFreq[w] += 1;
    }
}

void Classifier::initWordLabFreq(string &label, string &words) {
    vector<string> word = splitString(words);
    set<string> nonDupe;
    for (const auto &w : word) {
        nonDupe.insert(w);
    }
    for (const auto &w : nonDupe) {
        wordLabelFreq[ {label, w} ] += 1;
    }
}

vector<string> Classifier::splitString(string &str) {
    vector<string> words;
    istringstream source(str);
    string word;
    while (source >> word) {
        words.push_back(word);
    }
    return words;
}

bool less(double x, double y) {
    return (x / y) < 1.0;
}
//////////////// MAIN FUNCTION ////////////////

int main(int argc, const char * argv[]) {
    cout.precision(3);
    // Error checking
    if (!(argc == 3 || argc == 4)) {
        cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
        return 1;
    }
    if (argc == 4 && strcmp(argv[3], "--debug") != 0) {
        cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin.is_open()){
        cout << "Error opening file: " << argv[1] << endl;
        return 1;
    }
    ifstream finTest(argv[2]);
    if (!finTest.is_open()){
        cout << "Error opening file: " << argv[2] << endl;
        return 1;
    }

    bool debug = (argc == 4);

    Classifier data;
    data.initTrainingSet(fin, debug);



    csvstream csvTest(finTest);
    map<string, string> row;
    cout << endl << "test data:" << endl;
    int i = 0, correct = 0;
    while(csvTest >> row) {
        pair<string, double> temp = data.predict(row["content"], debug);
        cout << "  correct = " << row["tag"] << ", predicted = " << temp.first 
        << ", log-probability score = " << temp.second << endl;
        cout << "  content = " << row["content"] << endl << endl;
        if (temp.first == row["tag"]) {
            ++correct;
        }
        ++i;
    }

    cout << "performance: " << correct << " / " 
    << i << " posts predicted correctly" << endl;

    

    return 0;
}