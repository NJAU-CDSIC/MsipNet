import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable
import numpy as np


class SmoothGrad(object):
    """
    SmoothGrad: Compute smoothed gradients for model interpretability.
    Adds Gaussian noise to inputs multiple times, averages the gradients
    to reduce visual noise and highlight important features.
    Can compute gradients for sequence (x) and additional input (h),
    supports batch processing, and magnitude options (1=abs, 2=squared).
    """

    def __init__(self, model, device='cpu', only_seq=False, train=False,
                 x_stddev=0.015, t_stddev=0.015, nsamples=20, magnitude=2):
        self.model = model
        self.device = device
        self.train = train
        self.only_seq = only_seq
        self.x_stddev = x_stddev
        self.t_stddev = t_stddev
        self.nsamples = nsamples
        self.magnitude = magnitude
        self.features = model

    def get_gradients(self, x, s, h):
        self.model.eval()
        self.model.zero_grad()

        x = x.to(self.device)
        s = s.to(self.device)
        h = h.to(self.device)
        x.requires_grad = True
        h.requires_grad = True

        output = self.model(x, s, h)
        output = torch.sigmoid(output)

        output.backward()

        return x.grad, h.grad

    def get_smooth_gradients(self, x, s, h):
        return self.__call__(x, s, h)

    def __call__(self, x, s, h):
        batch_size, seq_len, feature_dim = x.shape
        x_stddev = (self.x_stddev * (x.max() - x.min())).to(self.device).item()
        h_stddev = (self.t_stddev * (h.max() - h.min())).to(self.device).item()

        total_grad_x = torch.zeros(x.shape).to(self.device)
        total_grad_h = torch.zeros(h.shape).to(self.device)
        x_noise = torch.zeros(x.shape).to(self.device)
        h_noise = torch.zeros(h.shape).to(self.device)

        for i in range(self.nsamples):
            x_plus_noise = x + x_noise.zero_().normal_(0, x_stddev)
            h_plus_noise = h + h_noise.zero_().normal_(0, h_stddev)

            grad_x, grad_h = self.get_gradients(x_plus_noise, s, h_plus_noise)

            if self.magnitude == 1:
                total_grad_x += torch.abs(grad_x)
                total_grad_h += torch.abs(grad_h)
            elif self.magnitude == 2:
                total_grad_x += grad_x * grad_x
                total_grad_h += grad_h * grad_h

        total_grad_x /= self.nsamples
        total_grad_h /= self.nsamples
        return total_grad_x, total_grad_h

    def get_batch_gradients(self, X, S, H):
        assert len(X) == len(S) == len(H), "The size of input X, S, and H must be the same."

        grad_x = torch.zeros_like(X)
        grad_h = torch.zeros_like(H)

        for i in range(X.shape[0]):
            x = X[i:i + 1]
            s = S[i:i + 1]
            h = H[i:i + 1]

            grad_x[i:i + 1], grad_h[i:i + 1] = self.get_smooth_gradients(x, s, h)

        return grad_x, grad_h


def generate_saliency(model, x, s, h, smooth=False, nsamples=2, stddev=0.15, only_seq=False, train=False):
    """
    Generate saliency maps for inputs x and h.
    Uses SmoothGrad to compute gradients optionally averaged over noisy samples.
    Parameters: model, inputs (x, s, h), smoothing options (nsamples, stddev), and mode flags.
    Returns: x_grad and h_grad as saliency gradients.
    """

    saliency = SmoothGrad(model, only_seq, train)
    x_grad, h_grad = saliency.get_smooth_gradients(x, s, h, nsamples=nsamples, x_stddev=stddev, t_stddev=stddev)
    return x_grad, h_grad


class GuidedBackpropReLU(torch.autograd.Function):
    """
    Custom ReLU for Guided Backpropagation.
    Forward pass: standard ReLU (outputs positive inputs only).
    Backward pass: allows gradients to flow only where both input and upstream gradient are positive.
    Used to generate clearer saliency maps for model interpretability.
    """

    @staticmethod
    def forward(ctx, input):
        pos_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size(), dtype=input.dtype, device=input.device),
            input,
            pos_mask)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        pos_mask_1 = (input > 0).type_as(grad_output)
        pos_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size(), dtype=input.dtype, device=input.device),
            torch.addcmul(
                torch.zeros(input.size(), dtype=input.dtype, device=input.device),
                grad_output,
                pos_mask_1),
            pos_mask_2)
        return grad_input


class GuidedBackpropReLUModule(nn.Module):
    """
    Wraps the GuidedBackpropReLU autograd function as an nn.Module.
    Allows using Guided Backpropagation ReLU within standard PyTorch models.
    Forward pass applies the custom ReLU with modified backward behavior.
    Used for generating interpretable saliency maps with guided backpropagation.
    """

    def __init__(self):
        super(GuidedBackpropReLUModule, self).__init__()

    def forward(self, input):
        return GuidedBackpropReLU.apply(input)


class GuidedBackpropSmoothGrad(SmoothGrad):
    """
    Extends SmoothGrad to use Guided Backpropagation.
    Replaces all ReLU activations in the model with GuidedBackpropReLU.
    Allows computation of smoothed gradients using guided backprop for clearer saliency maps.
    Inherits SmoothGrad functionality (noise, averaging, magnitude options).
    """

    def __init__(self, model, device='cpu', only_seq=False, train=False,
                 x_stddev=0.15, t_stddev=0.15, nsamples=20, magnitude=2):
        super(GuidedBackpropSmoothGrad, self).__init__(
            model, device, only_seq, train, x_stddev, t_stddev, nsamples, magnitude)

        self.replace_relu_with_guided_relu(self.features)

    def replace_relu_with_guided_relu(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, GuidedBackpropReLUModule())
            else:
                self.replace_relu_with_guided_relu(child)