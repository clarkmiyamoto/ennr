import torch
import torch as t
from torch.nn.functional import grid_sample
from dataclasses import dataclass, field

from generation import band_limited_sparse_transform_amplitude, get_diffeo_grid_sample, get_id_grid
from inverse import compose_diffeo_from_left

@dataclass
class DiffeoConfig:
    """
    Configuration for sampling diffeomorphism parameters
      resolution: grid resolution for diffeo
      x/y_range: the range of x/y frequency we sample from
      num_nonzero_params: the sparsity of the parameter matrix
      diffeo_amp: a list of amplitude we want to sample from
      num_diffeo_per_amp: how many diffeo to sample from a particular amplitude
      seed: numpy dirichlet distribution sampling seed
      alpha: controls the dirichlet distribution 'tilt'
    """
    resolution: int = 224
    x_range: list[int] = field(default_factory=lambda: [0, 3])
    y_range: list[int] = field(default_factory=lambda: [0, 3])
    num_nonzero_params: int = 3
    diffeo_amps: list[float] = field(default_factory=lambda: [0.1])
    num_diffeo_per_amp: int = 10
    seed: int = 37
    alpha: list[float] = None


def read_diffeo_from_path(path: str, device = t.device("cpu")):
   diffeo_param_dict = t.load(path, weights_only=False, map_location = device)
   diffeo_param = diffeo_param_dict['AB']
   inv_diffeo_param = diffeo_param_dict['inv_AB']
   diffeo_strenghts = diffeo_param_dict['diffeo_config']['diffeo_amp']
   num_of_diffeo_per_amp = diffeo_param_dict['diffeo_config']['num_diffeo_per_amp']
   return diffeo_param, inv_diffeo_param, diffeo_strenghts, num_of_diffeo_per_amp

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

    self._initialize_diffeo_config(diffeo_config)

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
  
  def _initialize_diffeo_config(self, diffeo_config: DiffeoConfig):
    if diffeo_config is None: pass
    arguments = diffeo_config.__dict__.copy()
    res = arguments.pop('resolution')
    self.x_res = res
    self.y_res = res
    params = []
    strength_list = arguments.pop('diffeo_amps')
    for strength in strength_list:
      A, B = band_limited_sparse_transform_amplitude(diffeo_amp=strength, **arguments)
      params.append(torch.cat([A,B], dim = -1))
    self.diffeo_params = torch.stack(params, dim = 0)
    self.diffeos = self.get_grid_from_param(self.x_res, self.y_res, self.diffeo_params)
    pass
  
  @staticmethod
  def get_grid_from_param(x_res, y_res, params):
    '''assume param have shape [strength, counts, x, y, 2]'''
    diffeos = []
    for param in params:
      A = param[..., 0]
      B = param[..., 1]
      diffeos.append(get_diffeo_grid_sample(x_res, y_res, A, B))
    return diffeos

  
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
  

  def up_down_sample(self, new_x_res, new_y_res, mode = 'bilinear', align_corners = False):
    id_grid = get_id_grid(x_res = new_x_res, y_res = new_y_res).to(self.device)
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