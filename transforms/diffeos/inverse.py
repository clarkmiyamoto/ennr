import torch as t
import torch.nn as nn 
import torch.nn.functional as F

from tqdm import tqdm

from generation import get_diffeo_grid_sample, get_id_grid

def compose_diffeo_from_left(diffeo_l: t.tensor, diffeo_r: t.tensor, mode = 'bilinear', align_corners = True):
  '''
   Composes two diffeomorphisms by first applying r then l (l interpolates r).
   
   Args:
       diffeo_l: Left diffeomorphism tensor of shape (n, x_res, y_res, 2)
       diffeo_r: Right diffeomorphism tensor of shape (n, x_res, y_res, 2)
       mode: Interpolation mode for grid_sample
       align_corners: Grid sample alignment parameter
       
   Returns:
       product: Composed diffeomorphism tensor of shape (n, x_res, y_res, 2)
   '''
  if len(diffeo_l.shape) != 4 or len(diffeo_r.shape) != 4 or diffeo_l.shape[0] != diffeo_r.shape[0]:
    raise Exception(f'shape do not match, left:{diffeo_l.shape}, right:{diffeo_r.shape}')
  img = t.permute(diffeo_r, (0, 3, 1, 2))
  product = t.nn.functional.grid_sample(img, diffeo_l, mode = mode, padding_mode='border', align_corners= align_corners) # left multiplication
  product = t.permute(product, (0, 2, 3, 1))
  return product


class BiasNetwork(nn.Module):
  '''
  Neural network module that adds learnable bias to AB coefficients with frequency scaling.
  
  Args:
    AB: Input tensor of AB coefficients
    extra_freq_scaling: Factor to scale the padding size of higher frequencies
  
  Properties:
    result_magnitude: Sum of absolute values for the final result (A and B)
    bias_magnitude: Sum of absolute values for the learned bias (A and B)
    original_magnitude: Sum of absolute values for the original AB coefficients
  
  Notes:
    - Initializes bias randomly scaled by the number of elements
    - Pads the input AB tensor with zeros based on frequency scaling factor
'''
  def __init__(self, AB, extra_freq_scaling = 1):
    # grid should have the shape of a grid_sample grid, i.e. (Channel, X, Y, 2)
    super().__init__()
    _, _, x_cutoff, y_cutoff = AB.shape
    self.AB = F.pad(AB,(0,(extra_freq_scaling) * y_cutoff, 0, (extra_freq_scaling) * x_cutoff),mode = 'constant', value = 0)
    # self.bias = nn.Parameter(self.AB.abs().sum() * (t.randint_like(self.AB, 1) - 1/2)/self.AB.numel())
    # self.bias = nn.Parameter(t.rand_like(self.AB))
    self.bias = nn.Parameter(t.rand_like(self.AB)/self.AB.numel())
    self.result = AB
  def forward(self):
    self.result = self.bias + self.AB
    return self.result
  @property
  def result_magnitude(self):
    return self.result[0].detach().abs().sum(), self.result[1].detach().abs().sum()
  @property
  def bias_magnitude(self):
    return self.bias[0].detach().abs().sum(), self.bias[1].detach().abs().sum()
  @property
  def original_magnitude(self):
    return self.AB[0].detach().abs().sum(), self.AB[1].detach().abs().sum()



def find_param_inverse(AB: t.Tensor, 
                       extra_freq_scaling = 1, 
                       num_epochs = 500,
                       resolution = 224, 
                       device = t.device('cpu'),
                       disable_tqdm_log = True) -> t.Tensor:
  '''
  Finds inverse parameters for a diffeomorphism using gradient descent.
  
  Args:
    AB: Input tensor [A,B] representing diffeomorphism parameters
    extra_freq_scaling: Factor for frequency padding in parameter space
    num_epochs: Number of optimization iterations
    resolution: the resolution to learn the diffeo
    device: PyTorch device for computation
  
  Returns:
  Tuple containing:
  - Inverse parameters [A_inverse, B_inverse] as tensor
  - History of inverse loss values
  - History of parameter magnitudes
  
  Notes:
  - Uses Adagrad optimizer with MSE loss
  - Grid size given by resolution
  - Includes regularization based on parameter magnitudes
  - Small bias values (<1e-7) are zeroed every 50 epochs
  '''
  inv_loss_hist = []
  AB_mag = []

  grid = get_diffeo_grid_sample(resolution, resolution, AB[0], AB[1]).to(device)
  AB = BiasNetwork(-AB.to(device), extra_freq_scaling = extra_freq_scaling).to(device)
  loss_fn = nn.MSELoss()

  optimizer = t.optim.Adagrad(AB.parameters(), lr = 0.1)
  
  id_grid = get_id_grid(resolution, resolution, device)
  
  for epoch in tqdm(range(num_epochs), disable=disable_tqdm_log):
      optimizer.zero_grad()
      new_AB = AB()
      with t.device(device):
          inv_grid = get_diffeo_grid_sample(resolution, resolution, new_AB[0], new_AB[1]).to(device)
      un_distorted = compose_diffeo_from_left(inv_grid, grid)
      unreg_loss = loss_fn(un_distorted, id_grid.expand(len(un_distorted),-1,-1,-1))
      loss = unreg_loss * (1 + 0.1 * AB.result.abs().sum()/AB.AB.abs().sum())
      loss.backward()
      optimizer.step()
      # scheduler.step(loss)

      if (epoch) % 50 == 0:
            with t.no_grad(): 
              inv_loss_hist.append(unreg_loss.item())
              AB_mag.append(AB.result.abs().sum().item()/2)
              AB.bias[t.abs(AB.bias) < 1e-7] = 0
            # print(loss.item())
  
  return AB.result.detach() , inv_loss_hist, AB_mag
