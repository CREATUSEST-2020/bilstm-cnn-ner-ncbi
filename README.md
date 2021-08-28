# BiLSTM+CharCNN with NCBI Disease Dataset implemented with Pytorch


## Model

- Character Embedding with `CNN`
- Additional Word Embeddings with Capitalization features(allCaps, upperInitial, lowercase, mixedCaps, noInfo)
- Concatenate `word embedding` and additional embeddings with`character represention`
- Put the feature above to `BiLSTM`

## Dependencies

- python>=3.5
- torch==1.4.0


## Dataset

|           | Train  | Eval  | Test  |
| --------- | ------ | ----- | ----- |
| # of Data | 5432   | 923   | 940   |

- NCBI disease corpus: a resource for disease name recognition and concept normalization([Original Dataset link](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBI_corpus.zip))
- This dataset has been converted to CoNLL format(IOB tagging) for NER using the following tool: ([Github link](https://github.com/spyysalo/standoff2conll))

## Pretrained Word Vectors

- Use [Glove embeddings](http://nlp.stanford.edu/data/glove.6B.zip) with 300 dimension
- Download and place the embedding file in /embeddings

## Usage

Training 

```bash
$ python3 main.py --train 
```

Evaluation

```bash
$ python3 main.py --train 
```

- **Evaluation prediction result** will be saved in `preds` dir

## Results

|                            | F1 (%) |
| -------------------------- | ----------- |
| Eval                       | 82.60       |
| Test                       | 79.6        |

