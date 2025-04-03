# ennr
Helper files for running experiments on measure equivariance of neural networks using diffeomorphisms 


# Files we need to make
`/model_wrappers` (files associated with pretrained models)
- `load.py` (tell it a model name (str), and it'll give you a wrapped version of that model
- `AbstractClass.py` (this contains a lot of helper functions for evlulating models)
  - Functions
    - `eval`: Gets end to end
    - `get_activation`: Gets activations of model
    - `steering_g_naive(AB: diffeo_param, i, j)`: Does activation steering w/ diffeo ONLY
- `/models` (File per model, i.e. `ResNet50.py`, `ViT.py`, etc.)

`/diffeos`
- `diffeo.py`
- `inverse.py`
- `generation.py`
```
class Diffeo
  def __init__(parameters for ensemble):
    self._generate_diffeos()
  def _generate_diffeos(...): # imports from `generation.py`
  def find_inverse(...): # imports from `inverse.py`
  def save():

  @static_method
  def load()
```

`/representationLearning`
- ...

`/metrics` (visualization / performance related functions)
- `similiarity_metrics.py` (a bunch of metrics to compare two tensors)
- `for_activation.py` (visulaizations / metrics for activation to activation comparisons)
- `for_endtoend.py` (visulaizations / metrics for end to end comparisons)


# Front-End Workflow
## Inspecting Invariance
1. Pick a bunch of models using `model_wrappers/load.py`
2. Pick an ensemble of diffeos using `/diffeos/diffeo.py`
3. Run metric using `/metrics`

## Learning Equivariance
1. Pick a model using `model_wrappers/load.py`
2. Pick an ensemble of diffeos using `/diffeos/diffeo.py`
3. Learn to activation steer using `/representationLearning` & model pikced in (1)
4. Run metric using `/metrics`

## Noise vs Mislabeling
I need to create my own data structure for noise $\epsilon$ and misslabeling $\eta$



