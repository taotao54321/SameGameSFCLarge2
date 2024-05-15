# SNES SameGame (J) solver for "SameGame, Hard"

## Requirement

* x64 CPU with BMI2 instructions
* Some RAM (16 GB or above)

I tested only on Linux.

## Usage

`dedup_board` binary generates the unique legal boards in the game.

```sh
cargo --example=dedup_board --profile=release-lto -- > dedup.out 2> dedup.log
```

And, `beam_search_all` binary searches the maximum score for a given board set.
I recommend to specify a initial maximum score for pruning (the known best score is 8987).
The last argument is a beam width (capacity). A large value will mean a large memory consumption. Please test from a small value (e.g. 1000000).

```sh
cat dedup.out | cargo --example=beam_search_all --profile=release-lto -- --prune-score-max=8700 1000000
```
