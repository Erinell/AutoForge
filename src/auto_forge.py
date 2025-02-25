#!/usr/bin/env python
"""
Script for generating 3D printed layered models from an input image.

This script uses a learned optimization with a Gumbel softmax formulation
to assign materials per layer and produce both a discretized composite that
is exported as an STL file along with swap instructions.
"""
import torch

from src.helper_functions import gumbel_softmax

def create_update_step(optimizer, loss_function, h, max_layers, material_colors, material_TDs, background):
    """
    Create a update step function using the specified loss function.

    Args:
        optimizer (torch.optim): The optimizer to use for updating parameters.
        loss_function (callable): The loss function to compute gradients.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (Tensor): Array of material colors.
        material_TDs (Tensor): Array of material transmission/opacity parameters.
        background (Tensor): Background color.

    Returns:
        callable: A function that performs a single update step.
    """
    def update_step(params, target, tau_height, tau_global, gumbel_keys):
        """
        Perform a single update step.

        Args:
            params (dict): Dictionary containing the model parameters.
            target (Tensor): The target image array.
            tau_height (float): Temperature parameter for height compositing.
            tau_global (float): Temperature parameter for material selection.
            gumbel_keys (Tensor): Random keys for sampling in each layer.

        Returns:
            tuple: A tuple containing the updated parameters, new optimizer state, and the loss value.
        """
        optimizer.zero_grad()
        loss_val = loss_function(params, target, tau_height, tau_global, gumbel_keys,
                                 h, max_layers, material_colors, material_TDs, background).requires_grad_()
        loss_val.backward()
        optimizer.step()
        return params, loss_val.item()

    return update_step


def discretize_solution_pytorch(params, tau_global, gumbel_keys, h, max_layers):
    """
    Discretize the continuous pixel height logits into integer layer counts,
    and force hard material selections.

    Args:
        params (dict): Dictionary containing the parameters 'pixel_height_logits' and 'global_logits'.
        tau_global (float): Temperature parameter for material selection.
        gumbel_keys (Tensor): Random keys for sampling in each layer.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.

    Returns:
        tuple: A tuple containing the discrete global material assignments and the discrete height image.
    """
    pixel_height_logits = params['pixel_height_logits']
    global_logits = params['global_logits']
    pixel_heights = (max_layers * h) * torch.sigmoid(pixel_height_logits)
    discrete_height_image = torch.round(pixel_heights / h).to(torch.int32)
    discrete_height_image = torch.clamp(discrete_height_image, 0, max_layers)

    def discretize_layer(logits, key):
        p = gumbel_softmax(logits, tau_global, key, hard=True)
        return torch.argmax(p)

    discrete_global = torch.stack([discretize_layer(global_logits[i], gumbel_keys[i]) for i in range(max_layers)])
    return discrete_global.cpu().numpy(), discrete_height_image.cpu().numpy()