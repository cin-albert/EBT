# Energy Landscape with Momentum

## Idea
Current energy optimization using normal Gradient Descent to find optimal of the energy landscape. What if we add the momentum acceleration to this gradient descent?

## Plan
- Using the final checkpoint trained on 128M document
- Training-free: apply momentum at inference time, no need to retrain the model
- Extension 1: Since energy optimization run only on 2-3 steps, they may not be sufficient to accumulate momentum, so we should try to increase the number of steps (e.g. 5, 8, 10) to see if momentum on increasing the number of steps help. Of course, we still have the caveat: thinking longer doesn't help in our reproduction.
- Extension 2: Retrain EBT with Momentum Gradient descent. If training-free doesn't work, we may need to try to train the model with more steps and momentum and see if momentum is adapted better


## Baselines
We should compare this momentum-based variants with previous arms: T++, EBT with normal GD (both paper and ours results)