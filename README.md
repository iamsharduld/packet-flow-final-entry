# Application Inference from Packet Flows: final entry

This rebuilds our final submission to the
[Application Inference from Packet Flows](https://github.com/FG-AI4H/application-inference-packet-flows-public)
challenge from saved model predictions. It scored **97.52** on the Final Test (macro F1: Task 1 0.982, Task 2 0.968).

## What it can and can't do

- It can rebuild the submission exactly. You can also change the last step (for example the Task 2 safety
  policies) and see which labels change.
- It can't tell you the score of a changed version, because the test labels are not public.
- It can't retrain the models or classify new flows. It includes the models' saved outputs, not the training
  code.

## Run it

```bash
git clone https://github.com/FG-AI4H/application-inference-packet-flows-public
uv run make_submission.py --dataset application-inference-packet-flows-public/dataset
```

This writes `out/submission.json` and checks that it matches the file we submitted. Without uv, run
`pip install numpy scipy`, then `python make_submission.py` with the same argument. Tests: `uv run pytest`.

## How it works

The models were trained beforehand. `predictions/` holds their probabilities, and `make_submission.py` does the last step.

- **Count rule.** Pick the labels with the highest total log-probability, subject to the number of flows per
  class matching the published test counts.
- **Task 1 (full flow).** Average five models, multiply by an earlier Task 2 blend's probabilities for the same
  flow, then apply the count rule.
- **Task 2 (first five packets).** Blend ten models with weights fitted by EM to minimise out-of-fold log loss,
  then apply the count rule with two safety policies:
  1. Flows that a reference blend labels facebook or instagram keep that label. These classes have the fewest
     training examples and are worth the most points per flow.
  2. The other changes form loops that keep the class counts. Loops are accepted in order of log-probability
     gain while their worst-case cost fits within the points Task 2 had left to win, `50 * (1 - 0.95164)`, where
     0.95164 is the reference labels' Task 2 macro F1 on the Final Test.

## Disclosure

- This entry uses the published test class counts in the dataset's `metadata.json`. The organizers allowed this
  for this edition, provided it is disclosed.
- This entry uses no test labels. Task 2 predictions use only Task 2 data.
- The Task 2 risk budget uses one leaderboard number: the reference labels' Final Test macro F1.

## Credits

The saved predictions come from TabPFN v2 and TabPFN-2.5 (Prior Labs), TabICL, Mitra (AutoGluon), TabDPT,
RealMLP and TabM (pytabkit), LightGBM, XGBoost, CatBoost and scikit-learn. Built with TabPFN.

## License

Code: MIT. The files in `predictions/` include outputs of TabPFN-2.5, whose licence allows those outputs to be
used only for non-commercial purposes.
