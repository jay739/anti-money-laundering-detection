# Contributing to anti-money-laundering-detection

Thanks for considering a contribution. Issues, discussions and PRs are all
welcome.

## Getting set up

Python 3.10 or newer and Java 17 (Spark 4 requires it). `pip install -r requirements.txt`. Download the IBM AML dataset from Kaggle into `data/`, then see the README for the train and evaluate commands.

## Running the checks

`pip install -r requirements-test.txt` then `pytest -q tests`. The tests start a local Spark session, so Java 17 must be on the PATH. CI runs them on every PR.

## Ground rules

- Open an issue or discussion before large changes, so effort isn't wasted.
- Keep PRs focused: one change per PR.
- New behavior needs a test where the repo has a test suite.
- CI must pass before merge.

## Licensing

By contributing you agree your contributions are licensed under the repo's
MIT license.
