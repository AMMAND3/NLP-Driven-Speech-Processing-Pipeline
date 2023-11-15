# This program should loop over the files listed in the input file directory,
# assign perplexity with each language model,
# and produce the output as described in the assignment description.

# Import necessary libraries
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
import math


# Function to add boundary symbols and normalize data:- add begin-of-utterance and end-of-utterance symbols
# For unigram, we add nothing. For bigram/trigram we add one/two "<s>" at the beginning and one "</s>" at the end.
# Function to add begin-of-utterance and end-of-utterance symbols
def process_data(utterances, n, vocabulary):
    start_symbol = "<s>"
    end_symbol = "</s>"
    processed_data = []

    for utterance in utterances:
        split = re.split(r'\s+', utterance)
        # Add the start symbols
        data_point = [start_symbol] * (n - 1) + split + [end_symbol]
        # Replace words not in vocabulary with <UNK>
        data_point = [word if word in vocabulary else '<UNK>' for word in data_point]
        processed_data.append(data_point)

    return processed_data

# Function to train n-gram model
def train_ngram_model(data, n):
    model = defaultdict(Counter)
    for utterance in data:
        for i in range(len(utterance) - n + 1):
            context = tuple(utterance[i:i + n - 1])
            token = utterance[i + n - 1]
            model[context][token] += 1
    return model

# Function to apply Laplace smoothing
def apply_laplace_smoothing(model, vocabulary, vocabulary_size, n):
    smoothed_model = defaultdict(Counter)
    for context in model:
        total_count = sum(model[context].values())
        denominator = total_count + vocabulary_size
        for token in vocabulary:
            smoothed_model[context][token] = (model[context].get(token, 0) + 1) / denominator
    return smoothed_model


def calculate_perplexity(model, data, n, vocabulary_size, smoothing=False):
    log_perplexity = 0.0
    N = 0  # Total number of words
    
    for utterance in data:
        N += len(utterance)  # Increment N for the total number of words in this utterance
        for i in range(len(utterance) - n + 1):
            ngram = tuple(utterance[i:i + n])
            prefix = ngram[:-1]
            token = ngram[-1]
            
            if prefix in model and token in model[prefix]:
                    probability = model[prefix][token] / sum(model[prefix].values())
            else:
                # Handle unknown n-grams (not found in the model)
                probability = 1 / vocabulary_size  # Uniform distribution over the vocabulary
                
            # Use log probabilities to avoid underflow
            log_probability = math.log(probability)
            log_perplexity -= log_probability
    
    # Calculate the perplexity (note: should be outside the loop)
    perplexity = math.exp(log_perplexity / N)
    return perplexity


# Function to read data from a file
def read_data(file_path):
    with open(file_path, 'r') as file:
        # Read each line, strip whitespace, and keep the line intact
        data = [line.strip() for line in file.readlines()]
    return data


if __name__ == "__main__":
    
    # Get the current working directory
    parent = Path.cwd()

    model_type = sys.argv[1]
    training_data_path = sys.argv[2]
    ppl_data_path = sys.argv[3]
    smoothing = False
    if len(sys.argv) == 5:
        smoothing_arg = sys.argv[4]
        smoothing = True
        
    # Check for invalid argument combination
    if model_type == 'unigram' and smoothing:
        print("Laplace smoothing is not valid for unigram models.")
        sys.exit(1)

    # Define the fixed vocabulary (this would typically come from another source or be built from the training data)
    FIXED_VOCABULARY = {'<s>', '</s>', '<UNK>'}  # Add all known words here

    # Read and preprocess training data
    training_data = read_data(training_data_path)
    # For simplicity, let's define the vocabulary as the set of all words occurring at least once in the training data
    full_vocabulary = set(word for line in training_data for word in line.strip().split(' '))
    FIXED_VOCABULARY.update(full_vocabulary)  # Update the FIXED_VOCABULARY with words from the training set
    print(len(FIXED_VOCABULARY))

    ppl_data = read_data(ppl_data_path)

    # Define the n for the n-gram model
    n = {'unigram': 1, 'bigram': 2, 'trigram': 3}[model_type]


    # Process training and perplexity data
    training_data_with_symbols = process_data(training_data, n, FIXED_VOCABULARY)
    ppl_data_with_symbols = process_data(ppl_data, n, FIXED_VOCABULARY)

    # Train the n-gram model
    ngram_model = train_ngram_model(training_data_with_symbols, n)
    
    # Calculate the total number of n-grams in the model
    total_ngrams = sum(len(words) for context, words in ngram_model.items())
    print(f"The total number of {model_type}s in the model (without laplace) is: {total_ngrams}")
    
    # Apply Laplace smoothing if specified
    if smoothing:
        ngram_model = apply_laplace_smoothing(ngram_model, FIXED_VOCABULARY, len(FIXED_VOCABULARY), n)

    # Calculate and output perplexity
    perplexity = calculate_perplexity(ngram_model, ppl_data_with_symbols, n, len(FIXED_VOCABULARY), smoothing)
    print(f"Perplexity: {perplexity}")
    
 