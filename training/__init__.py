"""Training utilities for the fake news detection system.

Kept as a package so that ``training/train.py`` can do
``from training.data_loader import load_dataset`` regardless of the directory
it is launched from.

Nothing heavy is imported here on purpose - importing this package must not
trigger a full training run or pull in Flask.
"""