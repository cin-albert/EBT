# Evaluate model on OOD dataset

## Objective
All of thinking scaling experiments so far are evaluated on in-distribution dataset (RedPajama-V2), while most capabilities claims in the paper were measured on OOD data. Therefore, to fairly verify the claims, we have to test our reproduced model on the OOD data as well

## Assets
- **Model:** Continue using the checkpoints we trained on 16M documents of RedPajama-v2
- **Data:** OOD datasets used in the paper, build them if they are not existed yet

## Tests
- RQ1: (Self-verification) Does picking the lowest-energy candidate (best-of-N) lower perplexity, as the paper reports? Or, does self-verification help? --> This is Q2
- RQ2: (Think longer) Do more inference steps than trained improve the prediction, as the paper reports, or are they only good at the number of steps they were trained on?