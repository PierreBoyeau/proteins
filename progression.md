

# Main results

- SVM (TFIDF) : .8 ROC AUC
- Word2Vec + Boosting : 0.85 ROC AUC (l=4 and taking overlaps)
- LSTM pipeline (Trained on SwissProt on clan prediction, then
transfer learning on allergens data):  0.89 ROC AUC
(`logs_transfer_group_shuffle` with weights.11)

    For now the best model is a 1 layer Bidirectional LSTM with kernel size 3

- LSTM pipeline V2 : Without transfer learning ==> not good! .66 ROC AUC

I think that one of the issue is the convolution choices OR the RNN choice (1 layer only)

# SPP
Looks like tfidf is a big plus : with overall features ==> 0.87 to 0.91