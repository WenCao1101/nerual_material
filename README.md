### Scripts

- `diffus_mitsuba.py`  
  Optimizes **diffuse BRDF** parameters by minimizing the discrepancy to a ground-truth image via inverse rendering.

- `nerual_netwrk_inverse.py`  
  Parameterizes the BRDF with a small neural network (MLP) and uses the same inverse-rendering pipeline to recover its weights.
