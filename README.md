# Spam Message Classifier

A machine-learning project (Python, text classification) that decides whether a message is **spam** or **ham** (not spam).
It covers the full workflow: labeled data → text preprocessing → TF-IDF features → classifiers (scikit-learn) →
evaluation on unseen messages → error analysis → predictions for new messages.

> ## Please read: about the dataset
> The real **SMS Spam Collection** could not be downloaded while this project was built (no internet access),
> so `data/spam_dataset.csv` is a **synthetic dataset** produced by `src/make_dataset.py`
> (~2,000 messages, ~16% spam, with slang, Hinglish, tricky genuine promotional messages and ~2% label noise).
> **Scores on it are not a measure of real-world performance.** For realistic numbers, use the real dataset
> (instructions below), which takes about a minute.

## Project structure

```
spam-classifier/
├── data/
│   └── spam_dataset.csv        Labeled messages (label, message)
├── notebooks/
│   └── spam_classifier.ipynb   Exploration, training, evaluation (with outputs)
├── src/
│   ├── make_dataset.py         Generates the synthetic dataset
│   ├── preprocess.py           Text cleaning + dataset loading
│   ├── train_model.py          Trains, tunes, evaluates, saves model, writes report + charts
│   ├── predict.py              Classifies a new message (CLI + importable function)
│   ├── run_tests.py            Runs the test-case table, records results in evaluation.md
│   └── app.py                  Simple web interface (Flask)
├── models/
│   ├── spam_model.pkl          Saved model
│   └── vectorizer.pkl          Saved TF-IDF vectorizer
├── results/
│   ├── evaluation.md           Metrics, error analysis, before/after, test cases
│   └── charts/                 Confusion matrix, class distribution, top spam words, ...
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

python src/train_model.py        # trains, evaluates, saves models/ and results/
python src/run_tests.py          # runs the test cases and records them in results/evaluation.md
python src/predict.py "You won a free prize! Click now to claim."
python src/predict.py            # interactive mode
python src/app.py                # web interface at http://127.0.0.1:5000
```

Open `notebooks/spam_classifier.ipynb` in Jupyter or Google Colab for the step-by-step version
(upload the `src/` and `data/` folders next to it on Colab).

## Using the real SMS Spam Collection

1. Download the dataset from UCI ("SMS Spam Collection") or Kaggle ("SMS Spam Collection Dataset").
2. Put the file in `data/` (UCI's tab-separated `SMSSpamCollection`, or Kaggle's `spam.csv`).
3. Run `python src/train_model.py --data data/SMSSpamCollection` (or `data/spam.csv`).

`load_dataset()` detects both layouts. Expect a few more errors than on the synthetic data, and re-check the
`KEEP_WORDS` list in `preprocess.py` against the new data.

## Method

| Step | What is done |
|---|---|
| Cleaning | lowercase, remove punctuation and numbers, remove stop words **except** a small keep-list (*you, your, now, call, ...*) |
| Features | `TfidfVectorizer`, unigrams + bigrams, `max_features=5000`, **fitted on training data only** |
| Split | 80/20 `train_test_split`, `stratify=y`, `random_state=42` |
| Models | Multinomial Naive Bayes, Logistic Regression (balanced), Linear SVM; hyper-parameters tuned with 5-fold CV on the training set |
| Metrics | accuracy, precision, recall, F1 (spam class), confusion matrix; not accuracy alone, since spam is only ~16% of the data |

## Results (synthetic dataset, test set of 396 messages)

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|---|
| Multinomial Naive Bayes (alpha=0.1) | 0.972 | 1.000 | 0.825 | 0.904 |
| Logistic Regression (C=1) | 0.972 | 1.000 | 0.825 | 0.904 |
| Linear SVM (C=0.1) | 0.972 | 1.000 | 0.825 | 0.904 |

The three models tie. All 11 missed spam messages read like ordinary chat: the dataset contains ~2% flipped labels
on purpose, which caps recall. Full details are in `results/evaluation.md`.

**Improvement that mattered.** With the default cleaning and Naive Bayes smoothing, the guidance's own example
*"You won a free prize! Click now to claim."* only scored a 0.39 spam probability and would have been missed. Two
changes fixed it (probability 0.98): keeping the spam-signal stop words (*you, your, now, ...*) and tuning the
smoothing parameter by cross-validation. This is the "over-cleaning" pitfall from the guidance. The step-by-step
comparison is in `results/evaluation.md` under "Improving the model".

## Test cases

`python src/run_tests.py` runs 14 cases (obvious spam, normal messages, empty / whitespace / symbol-only input,
very short text, Hinglish, genuine promotional messages, disguised spam) and records them in
`results/evaluation.md`. Currently 14/14 pass. These are sanity checks, not a benchmark; the genuinely tricky
categories are where a real dataset will show more failures.

## Limitations

* Synthetic training data (see the note at the top); real messages are messier.
* Only English and light Hinglish are represented.
* Bag-of-words models cannot understand context, and spammers change wording over time, so retrain regularly.
* Privacy: do not train on real private messages without permission.

## Skills practiced

Text preprocessing and NLP basics · TF-IDF feature extraction · classification with scikit-learn ·
model evaluation (precision, recall, F1, confusion matrix) · error analysis and documentation.
