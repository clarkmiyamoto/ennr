import torch
import torch.nn as nn
from torchvision.transforms import v2
from abc import ABC, abstractmethod

class AbstractModel(ABC, nn.Module):
    """
    Abstract base class for all model wrappers.
    
    This class defines common functionality for running neural network equivariance experiments
    One must define methods for 
        evaluation, 
        activation extraction,
        steering.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def preprocess_transform(self) -> v2:
        pass

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    @torch.no_grad()
    def eval(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward()
    
    @torch.no_grad()
    def get_activation(self):
        pass

    def _apply_steering_func(self, func):
        pass

    
