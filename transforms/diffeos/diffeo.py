import torch
import torch as t
from torch.nn.functional import grid_sample
from dataclasses import dataclass, field

from generation import band_limited_sparse_transform_amplitude, get_diffeo_grid_sample, get_id_grid
from inverse import compose_diffeo_from_left

@dataclass
class DiffeoConfig:
    """Configuration for diffeomorphism parameters"""
    resolution: int = 224
    x_range: list[int] = field(default_factory=lambda: [0, 3])
    y_range: list[int] = field(default_factory=lambda: [0, 3])
    num_nonzero_params: int = 3
    strength: list[float] = field(default_factory=lambda: [0.1])
    num_diffeo_per_strength: int = 10


def read_diffeo_from_path(path: str, device = t.device("cpu")):
   diffeo_param_dict = t.load(path, weights_only=False, map_location = device)
   diffeo_param = diffeo_param_dict['AB']
   inv_diffeo_param = diffeo_param_dict['inv_AB']
   diffeo_strenghts = diffeo_param_dict['diffeo_config']['strength']
   num_of_diffeo_per_strength = diffeo_param_dict['diffeo_config']['num_diffeo_per_strength']
   return diffeo_param, inv_diffeo_param, diffeo_strenghts, num_of_diffeo_per_strength

class DiffeoGrids(torch.Tensor):
  """ a class to store diffeo grids, mostly for readability  """

class DiffeoContainer:
  '''
  Container class for managing diffeomorphism transformations and their compositions.

  Args:
      x_res: X-dimension resolution
      y_res: Y-dimension resolution
      diffeo_config: DiffeoConfig that can be used to generate diffeos
      diffeo_params: list of parameters to generate diffeomorphism tensors (optional)
      diffeos: list of diffeomorphism tensors (optional)
      device: PyTorch device for computation

  Properties:
      x_res: X resolution
      y_res: Y resolution
      device: Current device
      length: Total number of diffeomorphisms

  Methods contain capability for:
      - Grid sampling with batched inputs
      - Grid generation from config or parameters
      - Up/down sampling to new resolutions
      - Computing inverse transformations
      - Save & load
      - to(device)
'''
  def __init__(self, diffeo_config: DiffeoConfig = None, x_res: int = 224, y_res: int = 224, diffeo_params = None, diffeos = None, device = t.device('cpu')):
    self._x_res = x_res
    self._y_res = y_res
    self.diffeo_params = diffeo_params
    if diffeos is None: self.diffeos = []
    if diffeos is not None: self.diffeos = diffeos
    self.resampled = {} # dict for resampled grid which are container objects
    self.inverse = None # container for inverse grid
    self._find_inverse_loss = []
    self._device = device
    self.to(device)

    self.initialize_diffeo_config(diffeo_config)

  @property
  def x_res(self): return self._x_res
  @property
  def y_res(self): return self._y_res 
  @property
  def device(self): return self._device
  @property
  def length(self):
    length = 0
    for diffeo in self.diffeos:
      length += len(diffeo)
    return length
  
  def initialize_diffeo_config(diffeo_config: DiffeoConfig):
    if diffeo_config is None: pass

    pass
  
  @staticmethod
  def _generate_diffeo(diffeo_config):
    pass
  
  def __getitem__(self, index):
    if isinstance(index, int): return self.diffeos[index]
    return self.diffeos[index[0]][index[1:]]
  
  def __len__(self):
    return len(self.diffeos)
  
  def __str__(self):
    return f"{type(self).__name__}(x_res={self.x_res}, y_res={self.y_res}, with {self.length} diffeos)"
  
  def __call__(self, input, mode = 'bilinear', align_corners = True, in_inference = False):
    if in_inference == False:
      # not during inference we have freedom with shape 
      if len(input.shape) == 4:
      # image have the same batch size as number of diffeo of a particular strength
      # input shape: batch, channel, x, y
      # resulting shape: strength, batch, channel, x, y
        return t.stack([grid_sample(input, diffeos, mode = mode, align_corners = align_corners) for diffeos in self.diffeos], dim = 0)
      if len(input.shape) == 5:
      # loops through diffeo and then images
      # intput shape: img, batch, channel, x, y
      # resulting shape: img, strength, batch, channel, x, y
        return t.stack([t.stack([grid_sample(image, diffeos, mode = mode, align_corners = align_corners) for diffeos in self.diffeos], dim = 0) for image in input], dim = 0)
    
    if in_inference == True:
      # all strength of diffeo will be reshaped in the batch dimension
      # input/output shape: batch, channel, x, y
      diffeos = t.cat(self.diffeos, dim = 0)
      output = grid_sample(input, diffeos, mode = mode, align_corners=align_corners)
      return output
  
  def to(self, device):
    self._device = device
    for index, diffeo in enumerate(self.diffeos): 
      self.diffeos[index] = self.diffeos[index].to(device)
    for children in self.resampled.values(): children.to(device)
    if type(self.inverse) == type(DiffeoContainer): self.inverse.to(device)
  
  def get_id_grid(self, x_res = None, y_res = None):
    if x_res == None: x_res = self.x_res
    if y_res == None: y_res = self.y_res
    id_grid = get_id_grid(x_res, y_res).to(self.device)
    return id_grid    

  def up_down_sample(self, new_x_res, new_y_res, mode = 'bilinear', align_corners = False):
    id_grid = self.get_id_grid(x_res = new_x_res, y_res = new_y_res).to(self.device)
    new_diffeo = []
    for diffeos in self.diffeos:
      new_diffeo.append(compose_diffeo_from_left(id_grid.repeat(len(diffeos), 1, 1, 1), diffeos, mode = mode, align_corners = align_corners))
    self.resampled[f'{new_x_res},{new_y_res}']= DiffeoContainer(new_x_res,new_y_res,diffeos = new_diffeo)
    return self.resampled[f'{new_x_res},{new_y_res}']
  
  def res_resample(self, new_x_res, new_y_res, mode = 'bilinear', align_corners = False, **kwargs):
    pass
  
  def get_inverse_param(self):
    pass

  def get_inverse_grid(self, base_learning_rate = 1000, epochs = 10000, learning_rate_scaling = 1, mode = 'bilinear', align_corners = True):
    inverse = []
    for diffeo in self.diffeos:
      lr = base_learning_rate * (1 + learning_rate_scaling * len(diffeo))
      inv_grid, lost_hist, epoch_num = find_inv_grid(diffeo, learning_rate = lr, mode = mode, align_corners = align_corners, epochs = epochs)
      inverse.append(inv_grid)
      self._find_inverse_loss.append({'loss': lost_hist, 'stopping_epoch': epoch_num, 'lr': lr})
    self.inverse = DiffeoContainer(self.x_res, self.y_res, diffeos=inverse, device = self.device)
    return self.inverse



class SparseDiffeoContainer(DiffeoContainer):
  '''
    Container for sparse diffeomorphism transformations, inheriting from DiffeoContainer.
    Args:
      x_res: X-dimension resolution
      y_res: Y-dimension resolution
      A, B: lists of transformation coefficients (optional)
      diffeos: list of diffeomorphisms (optional)
      rng: Random number generator
      seed: Random seed when rng is None
      device: PyTorch device
    Methods handle:
    - Sparse coefficient generation with amplitude control
    - Grid creation from A,B coefficients
    - Composition of transformations at different levels
    - Memory management for grids
    - Child container tracking for compositions
    Notes:
    - Uses sparse_transform_amplitude for coefficient generation
    - Maintains transformation parameters history
    - Allows clearing of computed grids to manage memory
  '''
  def __init__(self, x_res: int, y_res: int, A = None, B = None, diffeos = None, rng = None, seed = 37, device = t.device('cpu')):
    super().__init__(x_res, y_res, diffeos, device)
    if rng == None:
      self.rng = 'default with seed=37'
      self._rng = np.random.default_rng(seed = seed)
    else: 
      self.rng = 'passed in'
      self._rng = self.rng
    self.A = A
    self.B = B
    if A == None: self.A = []
    if B == None: self.B = []
    self.diffeo_params = []
    self.children = []

  def sparse_AB_append(self, 
                       x_range, 
                       y_range, 
                       num_of_terms, 
                       diffeo_amp, 
                       num_of_diffeo, 
                       rng = None, 
                       seed = 37, 
                       alpha = None):
    
    if rng == 'self': rng = self._rng
    
    self.diffeo_params.append({'x_range': x_range, 
                               'y_range': y_range, 
                               'num_of_diffeo':num_of_diffeo, 
                               'diffeo_amp':diffeo_amp, 
                               'num_of_terms': num_of_terms, 
                               'rng':rng, 
                               'seed':seed, 
                               'alpha':alpha})
    
    A_nm, B_nm = band_limited_sparse_transform_amplitude(**self.diffeo_params[-1])
    
    self.A.append(A_nm)
    self.B.append(B_nm)
  
  def get_all_grid(self):
    for A, B in zip(self.A, self.B):
      self.diffeos.append(get_diffeo_grid_sample(self.x_res, self.y_res, A, B))
    
  def clear_all_grid(self):
    self.diffeos = []

  def get_composition(self, level = 1):
    new_container = diffeo_compose_container(self, level = level)
    if new_container in self.children: self.children.remove(new_container)
    self.children.append(new_container)
    return self.children[-1]


