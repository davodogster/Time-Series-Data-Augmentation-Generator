# Time-Series-Data-Augmentation-Generator
Generate augmented time series batches for sequence to sequence deep learning when training an LSTM/RNN.

This repo is for anyone training a sequence to seqeunce model that wants to use time series data augmentation on the fly
It makes use of the awesome [tsaug library](https://tsaug.readthedocs.io/en/stable/).

It expects data loaded into memory in this format:
X_train: 3D array (N_sequences, sequence length, n_features)
y_train: 3D array (N_sequences, sequence length, n_classes)

# Usage

```
import seq2seq_ts_generator as tsaug_gen

training_gen = tsaug_gen.tsaug_generator(X_train, y_train, batch_size= 128)
n = next(training_gen)
print(n)
hist = model.fit_generator(training_gen, epochs=100, 
                                       verbose = 2, use_multiprocessing=True,
                           steps_per_epoch=16) 
```

It could easily be adapted for sequence to classification problems - just don't return augmented y - you don't need it.

Any Suggestions for improvements - e.g using tf.data to speed it up, then let me know.

Enjoy!
