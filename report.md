# Report

## Task 1 - Data Preparation

**_Data Cleaning, Data Transformation & Data Prep Output_**

The Data Cleaning was very specific and robust, with an output of words individually from each content file.

<ins>_Data Cleaning Choices_</ins>

- Recursively go through each file of the directories
- Process each file, starting off with removes the headers, speaker tags and lines beginning with !. Handle any special data sequences. Specifically looking through the data, there may be found texts in between angular brackets as well as unicode characters to remove
- Clean up the words, like converting underscores or ^ to spaces, or removing the numbers/ symbols in them. Also check whether the line is empty, and hence remove that line altogether

<ins>_Data Cleaning Reasoning (why)_</ins>

- The approach was made very **_strategic_** and ordered in cleaning the data; for example, only checking for empty lines at the end
- Focus was for **_strong data accuracy_**. All words preprocessed are string combinations of letters only
- All the words are uppercased so that it is **_standardized_**

<ins>_Data Transformation Choices_</ins>

The Data Transformation procesdure is more generalized, and also deals with non-existing phrases. The output consists wholly of CMU pronounciations of the text word. i.e, **_All words that were transformed consists of utterances present in the dictionary_**.

- When processing the files, we look at each word of a sentence, line by line. The precheck is to see if the word exists in the cmu_dict. If it does, we simply map the word accordingly
- If the word does not exist in the dictionary, we then try to find the best match of the word in the cmu_dict

  - The inital check is if the word is less than length of 2. If it is, we assume it is a useless word like "PR" since words "DO" and "AS" are already filtered out at this point

  - We then try to find the best split of the words between prefixes and suffixes. This was done with the main objective of acting on words like "TAPERECORDER" where one could simply get the broken words TAPE-RECORDER and pronounce them individually

    - If breaking apart of the words was successful, return the best match pronounciation as the sum of the 2 individual cmu key pronounciations

    - If both affixes do not exist, we then check whether at least one of them exist, its length is greater than 1. This helps preignore non-existing words like "H-JSHHJJ" and helps get rid of them

  - If the previous conditions did hold, we also use levenshtein distance to check the prefix and the suffix. We check both because ideally, both of them should be similar to their respective pronounciations

  - If neither affixes exist, we have a final edge case handler where we can do one final check to see if the we can find a Levenshtein mapping against the cmu dictionary. Otherwise, it is better to get rid of the word

<ins>_Data Transformation Reasoning (why)_</ins>

- The first main choice is to see if there is a direct transformation from cmu_dict possible. This is the easiest and cleanest way to ensure a base dataset

- A lot of words that were not able to be directly converted used **_combination of two words_** like "TAPERECORDER" and "STICKYFINGER". Hence, the transformation process acts on finding these best word splits - checking on affix possibilities of either both or one of them exists, which it again does recursively. It also uses Levenshtein as a final measure

- If both affixes do not exist, the code utilizies Levenshtein function for the affixes to see if any existing affixes, or even on the whole word against the CMU dictionary. This helped catch words like HAS-TUK where HAS does exist but TUK does not

- It is also carried out to be very generalized, and will work on other cleansed data sets of english words

<ins>_Data Prep Output Choices_</ins>

The Data output consists of the following files:

- `data/dev.txt`
  - 20% of the original tranformed set of word pronunciations selected at random
- `data/training.txt`
  - 80% of the original tranformed set of word pronunciations selected at random

sample text from each file looks like:

```bash
1  W AH1 N
2  B AE1 K
3  DH AE1 T
4  Y UW1
5  Y UW1
6  K AE1 N T
7  Y AE1
8  G AA1 T AH0
9  D R EH1 S T
10 Y AH1 M IY0
```

<ins>_Data Prep Output Reasoning (why)_</ins>

- The choice was to output the utterances of a single word separated per line. This was to get as many patterns of n-grams, strictly based on each individual words as samples for how the vocabulary sets are arranged

- Also, this allows for a more efficient vocabulary set size to work with. The vocabulary size set reduced from over 8000 to double digits (~67)

## Task 2 - Training N-gram Models

**_Vocabulary Construction_**

_Top 95% Words_:

The vocabulary is built from the top 95% most frequent words in the training data

- **Why**: This is done to focus on the most relevant words and to mitigate the effect of very rare words, which might not be helpful for the model's generalization. Words that fall outside the top 95% are treated as unknown `<UNK>` to handle the presence of rare or unseen words during evaluation

_Set Data Structure_: A set is used for the vocabulary to allow for fast membership checking

- **Why**: It is important for the word normalization process

**_Processing Utterances_**

_Adding Boundary Symbols_:

Boundary symbols (`<s>` for start and `</s> `for end) are added to utterances to denote the beginning and the end

- **Why**: It is crucial for n-gram models to understand context at the edges of sentences.

_Normalization_: Words not in the constructed vocabulary are replaced with <UNK

- **Why**: This step is essential for smoothing and dealing with unseen words during model evaluation

**_Training n-gram Models_**

_Tuple for n-gram Representation_: Tuples are used to represent n-grams

- **Why**: This is because they are hashable and can be used as keys in a dictionary, which is required for counting their frequencies

**_Smoothing_**

_Laplace Smoothing_: When smoothing is applied, the model is adjusted to account for unseen n-grams, distributing some probability mass to them

- **Why**: This adjustment is critical for handling words or contexts not encountered during training

_Efficient Smoothing_: Instead of modifying the model during the Laplace smoothing process (which could be computationally expensive), the code chooses to apply smoothing dynamically during the perplexity calculation

- **Why**: This approach is computationally more efficient as it avoids the need to store adjusted probabilities for all possible n-grams, which can be prohibitively large, especially for trigram models

**_Perplexity Calculation_**

_Logarithmic Computation_: Logarithms are used in the perplexity calculation to prevent underflow

- **Why**: It can occur when multiplying many probabilities together, as probabilities are often very small numbers
  Overall Design Choices

_Efficiency_: The code is designed to be efficient both in terms of memory and computation

- **Why**: By avoiding unnecessary storage of smoothed probabilities and using sets for vocabulary, the program runs faster and uses less memory

_Scalability_: The approach scales relatively well with large datasets

- **Why**: The choice to apply smoothing on the fly during perplexity calculation, rather than storing a potentially massive smoothed model, allows the program to scale to larger vocabularies and longer texts

## Task 3 - Evaluating N-gram Models

| Model   | Smoothing  | Training set PPL | Dev set PPL |
| ------- | ---------- | ---------------- | ----------- |
| unigram | -          | 34.7103          | 34.7206     |
| bigram  | unsmoothed | 6.2627           | 6.2581      |
| bigram  | Laplace    | 6.2664           | 6.2630      |
| trigram | unsmoothed | 3.0979           | 3.1022      |
| trigram | Laplace    | 3.1360           | 3.1419      |
