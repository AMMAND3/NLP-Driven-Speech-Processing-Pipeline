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
