# This program should loop over the files listed in the input file directory,
# assign perplexity with each language model,
# and produce the output as described in the assignment description.

# Import necessary libraries
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
import math

# Here the fixed vocabulary is first built
def build_vocabulary(utterances):
    
    vocabulary = {}
    vocab_list = set()
    
    # Count word occurrences in the training data
    for utterance in utterances:
        split = re.split(r'\s+', utterance)
        for utteri in split:
            if utteri in vocabulary:
                vocabulary[utteri] += 1
            else:
                vocabulary[utteri] = 1

    # Sort the vocabulary by word frequency
    sorted_vocab = {k: v for k, v in sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)}

    # Calculate the number of words to select (top 95%)
    total_words = len(sorted_vocab)
    num_words_to_select = int(0.95 * total_words)

    # Select the top 95% of words
    selected_words = list(sorted_vocab.keys())[:num_words_to_select]

    # Create a set from the selected words
    for word in selected_words:
        vocab_list.add(word)
        
    vocab_list.add('<UNK>')
    vocab_list.add('<s>')
    vocab_list.add('</s>')
    
    return vocab_list



# Function to add begin-of-utterance and end-of-utterance symbols
# For bigram/trigram add appropirate "<s>" at the beginning and end
def add_boundary_symbols_and_normalize(utterances, vocab_list):
    symboled_data = []
    counter = 0  # Counter for unknown words
    start_symbols = ["<s>"] * (n - 1)
    end_symbols = ["</s>"] if n > 1 else []
    # Process each utterance
    for utterance in utterances:
        # Split the utterance into words and replace non-vocabulary words with <UNK>
        words = re.split(r'\s+', utterance)
        normalized_words = [word if word in vocab_list else '<UNK>' for word in words]
        counter += normalized_words.count('<UNK>')  # Update the counter for <UNK>
        
        # Construct the data point with boundary symbols for n-gram models
        if n == 1:  # For unigram, we don't add boundary symbols
            symboled_data.extend([[word] for word in normalized_words])
        else:  # For bigram and trigram, we add start and end symbols
            data_point = start_symbols + normalized_words + end_symbols
            symboled_data.append(data_point)

    return symboled_data, counter

# Function to normalize and train n-gram model
def train(data, n):

    # Train the n-gram model on the normalized data
    model = defaultdict(Counter)
    for utterance in data:
        for i in range(len(utterance) - n + 1):
            ngram = tuple(utterance[i:i + n])
            prefix = ngram[:-1]
            token = ngram[-1]
            model[prefix][token] += 1
    return model

# Function to apply Laplace smoothing
def apply_laplace_smoothing(model, vocabulary, vocabulary_size, n):
    smoothed_model = defaultdict(Counter)
    for prefix in model.keys():
        total_count = sum(model[prefix].values())
        denominator = total_count + vocabulary_size
        for token in vocabulary:
            smoothed_model[prefix][token] = (model[prefix][token] + 1) / denominator
    return smoothed_model

# Function to calculate perplexity
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
        
    # Read training and perplexity data
    training_data = read_data(training_data_path)
    ppl_data = read_data(ppl_data_path)
    
    # Define the n for the n-gram model
    n = {'unigram': 1, 'bigram': 2, 'trigram': 3}[model_type]
    
    # First create the fixed vocabulary
    vocab_list = build_vocabulary(training_data)
    print('Vocabulary size : ' + str(len(vocab_list)))
    
    # Add boundary symbols to the data
    training_data_with_symbols, _ = add_boundary_symbols_and_normalize(training_data, vocab_list)

    # Normalize training data and train n-gram model
    ngram_model = train(training_data_with_symbols, n)
    
    # If laplace smoothing then apply laplace
    if smoothing:
        ngram_model = apply_laplace_smoothing(ngram_model, vocab_list, len(vocab_list), n)

    # Apply symbols to ppl data as well
    ppl_data_with_symbols, ppl_counter = add_boundary_symbols_and_normalize(ppl_data, vocab_list)
    # How many unknown data for ppl
    print('Unknowns found : ' + str(ppl_counter))

    # Output the perplexity
    ppl = calculate_perplexity(ngram_model, ppl_data_with_symbols, n, len(vocab_list), smoothing)
    print(f"Perplexity: {ppl}")
    