# Gibberish Detector

A markov chain based gibberish detector with fast heuristic checks.
Trained on 7B tri-grams from wikipedia.


## Usage

At inference time, the detector will return a boolean value indicating whether the input text is gibberish or not by calling `is_text(str)`.

```python
gb = GibberishDetector()
gb.is_text("String to evaluate here")
```

Before using markov probabilities, the detector use a number of
heuristics to determine if the text is gibberish or not. If the text fail any of the heuristics, the detector will retrurn false. Then
the detector will compute the markov probabilities of the text and return true if the transition probabilities are above a certain threshold.

## Data

Train data: https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011?resource=download