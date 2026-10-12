# Song identification from hummed and whistled audio

Can a model tell which of 8 songs someone is humming or whistling from a 10-second recording?

This is hard: the same song sounds completely different depending on the person, their vocal range and whether they hum or whistle, and there are no instruments or lyrics to rely on. The dataset has 800 clips across 8 classes, and the test set contains **completely unseen speakers**. A random classifier scores 12.5%.

## Approach

I compared two families of feature pipelines, each followed by normalisation and a classifier:

**Audio features**
- A1: MFCCs
- A2: MFCCs + delta and delta-delta coefficients
- A3: A2 + chroma, spectral centroid, rolloff, bandwidth, zero crossing rate and tempo

**Image features from mel spectrograms**
- B1: LBP texture features
- B2: LBP + mel band statistics
- B3: LBP + HOG + mel band statistics

**Combined:** C joins A3 with B2.

**Classifiers:** logistic regression, KNN, SVM (RBF kernel), Naive Bayes and a soft-voting ensemble (logistic regression + KNN + SVM), all tuned with GridSearchCV and 5-fold cross-validation.

## Results

- Each audio feature addition improved validation accuracy: A1 0.200, A2 0.263, A3 0.312.
- The richest image pipeline (B3, 8,699 features) overfitted. Regularisation, PCA and correlation filtering didn't fix this, so it wasn't selected.
- **The final pipeline reached 33.8% test accuracy on unseen speakers, against a 12.5% random baseline.**
- "Happy" was the easiest song to recognise, while "Feeling" and "Married" were the hardest.

## What I'd do next

- Model hums and whistles separately, since their spectrograms differ a lot.
- Take multiple overlapping windows per clip to effectively enlarge the training set.
- Select PCA components by cross-validation.
- Try transfer learning with a pretrained audio network.

## Tools

Python · librosa · scikit-learn · scikit-image · NumPy · pandas · Matplotlib

*Completed as part of the MSc Artificial Intelligence at Queen Mary University of London. The audio dataset (MLEnd Hums and Whistles) is not included in this repository.*
