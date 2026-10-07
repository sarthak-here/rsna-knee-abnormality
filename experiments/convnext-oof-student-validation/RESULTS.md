# Result

The fixed-epoch paired OOF-teacher target audit passed.

- Control ensemble gold macro AUC: `0.8294126`
- Corrected-target ensemble gold macro AUC: `0.8475575`
- Delta: `+0.0181449`
- Positive seeds: `5 / 5`
- Paired-bootstrap `p_improve`: `0.993`
- Paired-bootstrap 95% interval: `[0.0033146, 0.0339168]`
- Fit population in both arms: 2,610 identical non-gold studies
- Organizer-labelled studies used for fitting: 0
- Submission artifacts: none

The candidate passed all four gates, proving that the true fold-held-out public
ConvNeXt teacher improves a controlled RadImageNet student relative to Steven v4
targets alone. Its absolute private score remains below the stronger incumbent
diagnostic and it has not been compared with the locked `0.944` leaderboard
ensemble. Therefore this result authorizes only another private experiment.
