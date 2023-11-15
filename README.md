# Intro to NLP - Assignment 3

## Team

| Student name    | CCID    |
| --------------- | ------- |
| Amman Das       | amman1  |
| Hitarth Kothari | hitarth |

## TODOs

In this file you **must**:

- [*] Fill out the team table above. Please note that CCID is **different** from your student number.
- [*] Fill out the table in the [Evaluation](#evaluation) section.
- [*] Acknowledge all resources consulted (discussions, texts, urls, etc.) while working on an assignment. Non-detailed oral discussion with others is permitted as long as any such discussion is summarized and acknowledged by all parties.
- [*] Provide clear installation and execution instructions that TAs must follow to execute your code.

## Execution

_The arguments include three positional arguments in this order:_

1. one positional argument for model type (unigram/bigram/trigram)
2. one argument for the path to the training data
3. one argument for the path to the data for which perplexity will be computed
4. one optional argument for smoothing (--laplace) could be present.

For example, to execute using the main Python script when training on the training.txt and calculating perplexity for dev.txt, with laplace smoothing:

```bash
python3 src/main.py bigram data/training.txt data/dev.txt --laplace
```

## Data

- [*] We have provided the training and dev sets in the [data](data) directory.

## Evaluation

| Model   | Smoothing  | Training set PPL | Dev set PPL |
| ------- | ---------- | ---------------- | ----------- |
| unigram | -          | 34.7103          | 34.7206     |
| bigram  | unsmoothed | 6.2627           | 6.2581      |
| bigram  | Laplace    | 6.2664           | 6.2630      |
| trigram | unsmoothed | 3.0979           | 3.1022      |
| trigram | Laplace    | 3.1360           | 3.1419      |

**Grad student extension**  
|bigram (KenLM) | Kneser-Ney | | |
|trigram (KenLM) | Kneser-Ney | | |

## Resources

### Chatgpt

- _It was consulted to help seperate our vocabulary into 95:5 batch_
  - "How to appropriately split an initial vocabulary dataset into 95:5, where the rest 5 can be set as `<UNK>` values"
- _It was consulted on helping create the ngrams model_
  - "What python data structure can be used to create ngram models? Give an example"
- _It also helped in finding bugs for specific parts of the code; example for perplexity_
  - "Why is my code for perplexity giving me an abnormal zero-division error"

### Reference Links:

_The following are the reference links that were used during the development process:_

1. For removing file extension from filenames in Python: [StackOverflow](https://stackoverflow.com/questions/2212643/python-recursive-folder-read)
2. For checking if a string starts with certain characters: [StackOverflow](https://stackoverflow.com/questions/8802860/checking-whether-a-string-starts-with-xxxx)
3. For removing all non-ASCII characters: [StackOverflow](https://stackoverflow.com/questions/20889996/how-do-i-remove-all-non-ascii-characters-with-regex-and-notepad)
4. Understanding the Levenshtein distance: [DataConsulting](http://www.dataconsulting.co.uk/the-levenshtein-distance/)
5. For ranking matches using the Levenshtein distance: [StackOverflow](https://stackoverflow.com/questions/10405440/percentage-rank-of-matches-using-levenshtein-distance-matching)
6. For determining the best way to split a string in Python: [StackOverflow](https://stackoverflow.com/questions/70541496/whats-the-best-python-string-split-approach)
7. For calculating the Levenshtein distance between two strings: [CodeReview StackExchange](https://codereview.stackexchange.com/questions/217065/calculate-levenshtein-distance-between-two-strings-in-python)
8. Figuring out how to apply <UNK> to a fixed dataset [Web Stanford](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
9. Creating own n-grams model without a library [Youtube](https://www.youtube.com/watch?v=GfJpzJRYa-U&ab_channel=DataScienceandAISchool%28FutureAcademy%29)
