# Results

Store generated seed-level metrics and aggregate tables here. The manuscript
build script derives its cited tables and figures from these reports and writes
them to `papers/preprint/artifacts/` and `papers/preprint/figures/`.

The completed controlled-experiment summary is maintained in
[`../../RESULTS.md`](../../RESULTS.md). It reports the three-seed BPB tables,
paired controls, and the limits of the current evidence.

Use `experiments/evaluate_ilm.py` to write one held-out BPB report per
checkpoint. Then run `experiments/aggregate_results.py` over evaluation and
generation reports to create CSV summaries, bootstrap intervals, and PDF
figures in an `aggregate/` subdirectory.
