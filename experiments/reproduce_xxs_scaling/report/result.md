# EBT vs Transformer++ — xxs scaling run (numerical results)

## Final validation and throughput

| model | step | valid loss | valid ppl | steps/s | tokens/s | peak mem (MiB) | ETA 105k (h) |
|---|---|---|---|---|---|---|---|
| Transformer++ | 104999 | 3.6964 | 40.30 | 16.595 | 135949 | 9263 | 1.76 |
| EBT | 104999 | 3.8373 | 46.40 | 3.347 | 27417 | 38445 | 8.71 |

## Validation curve

| step | Transformer++ loss | Transformer++ ppl | EBT loss | EBT ppl |
|---|---|---|---|---|
| 14999 | 4.2930 | 73.19 | 4.8040 | 122.00 |
| 29999 | 4.0458 | 57.16 | 4.2550 | 70.46 |
| 44999 | 3.9273 | 50.77 | 4.1022 | 60.47 |
| 59999 | 3.8406 | 46.56 | 3.9987 | 54.53 |
| 74999 | 3.7727 | 43.50 | 3.9218 | 50.49 |
| 89999 | 3.7220 | 41.35 | 3.8650 | 47.70 |
| 104999 | 3.6964 | 40.30 | 3.8373 | 46.40 |
