import torch

from src.composite import composite_image_combined
from src.utils import rgb_to_lab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

def huber_loss(pred, target, delta=0.1):
    """
    Compute the Huber loss between predictions and targets.

    Parameters:
        pred (Tensor): Predicted values.
        target (Tensor): Ground-truth values.
        delta (float): Threshold at which to change between quadratic and linear loss.

    Returns:
        Tensor: The Huber loss.
    """
    return torch.nn.functional.huber_loss(pred, target, "mean", delta)

def loss_fn_perceptual(params, target, tau_height, tau_global, gumbel_keys, h, max_layers, material_colors, material_TDs, background):
    """
    Compute a perceptual loss between the composite and target images.

    Both images are normalized to [0,1], converted to CIELAB, and then the MSE is computed.
    """
    comp = composite_image_combined(params['pixel_height_logits'], params['global_logits'],
                                        tau_height, tau_global, gumbel_keys,
                                        h, max_layers, material_colors, material_TDs, background, mode="continuous")
    comp_norm = comp# / 255.0
    target_norm = target# / 255.0
    comp_lab = rgb_to_lab(comp_norm)
    target_lab = rgb_to_lab(target_norm)
    loss_lab = torch.mean((comp_lab - target_lab) ** 2)
    #jax.debug.print("hello {bar}", bar=loss_lab)
    return huber_loss(comp, target)

def loss_fn(params, target, tau_height, tau_global, gumbel_keys, h, max_layers, material_colors, material_TDs, background):
    """
    Compute the mean squared error loss between the composite and target images.
    By default, we use continuous (soft) compositing.

    Args:
        params (dict): Dictionary containing the parameters 'pixel_height_logits' and 'global_logits'.
        target (Tensor): The target image array.
        tau_height (float): Temperature parameter for height compositing.
        tau_global (float): Temperature parameter for material selection.
        gumbel_keys (Tensor): Random keys for sampling in each layer.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (Tensor): Array of material colors.
        material_TDs (Tensor): Array of material transmission/opacity parameters.
        background (Tensor): Background color.

    Returns:
        Tensor: The mean squared error loss.
    """
    comp = composite_image_combined(params['pixel_height_logits'], params['global_logits'],
                                        tau_height, tau_global, gumbel_keys,
                                        h, max_layers, material_colors, material_TDs, background, mode="continuous")
    return torch.mean((comp - target) ** 2)
