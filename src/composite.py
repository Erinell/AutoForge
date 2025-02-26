import torch

from src.helper_functions import gumbel_softmax
from src.utils import adaptive_round

def composite_pixel_combined(pixel_height_logit, global_logits, tau_height, tau_global,
                             h, max_layers, material_colors, material_TDs,
                             background, gumbel_keys, mode="continuous"):
    """
    Composite one pixel using either a continuous or discrete method,
    depending on the `mode` parameter.

    Args:
        pixel_height_logit: Raw logit for pixel height.
        global_logits: Global logits per layer for material selection.
        tau_height: Temperature parameter for height (soft printing).
        tau_global: Temperature parameter for material selection.
        h: Layer thickness.
        max_layers: Maximum number of layers.
        material_colors: Array of material colors.
        material_TDs: Array of material transmission/opacity parameters.
        background: Background color.
        gumbel_keys: Random keys for sampling in each layer.
        mode: "continuous" for soft compositing, "discrete" for hard discretization.

    Returns:
        Composite color for the pixel (scaled to [0,255]).
    """
    # Compute continuous pixel height (in physical units)
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logit).cuda()
    continuous_layers = pixel_height / h
    # Adaptive rounding: when tau_height is high, we get a soft round; when tau_height is low (<=0.01), we get hard rounding.
    adaptive_layers = adaptive_round(continuous_layers, tau_height, high_tau=0.1, low_tau=0.01, temp=0.1)
    # For the forward pass we want a crisp decision; however, we want to use gradients from the adaptive value.
    discrete_layers = torch.round(continuous_layers).cuda() + (adaptive_layers - torch.round(continuous_layers).cuda())
    discrete_layers = discrete_layers.to(torch.int32)

    # Parameters for opacity calculation.

    A = 0.178763
    k = 39.302848
    b = 0.351177

    def step_fn(carry, i):
        comp, remaining = carry
        # Process layers from top (last layer) to bottom (first layer)
        j = max_layers - 1 - i
        p_print = torch.where(j < discrete_layers, torch.tensor(1.0), torch.tensor(0.0)).cuda()
        eff_thick = p_print * h

        # For material selection, force a one-hot (hard) result when tau_global is very small.
        if mode == "discrete":
            p_i = gumbel_softmax(global_logits[j], tau_global, gumbel_keys[j], hard=True).cuda()
        elif mode == "continuous":
            p_i = torch.cond(
                tau_global < 1e-3,
                lambda: gumbel_softmax(global_logits[j], tau_global, gumbel_keys[j], hard=True).cuda(),
                lambda: gumbel_softmax(global_logits[j], tau_global, gumbel_keys[j], hard=False).cuda(),
            )
        else:
            p_i = global_logits[j]
        
        color_i = torch.matmul(p_i, material_colors).cuda()
        TD_i = torch.matmul(p_i, material_TDs).cuda() * 0.1
        
        
        opac = A * torch.log(1 + k * (eff_thick / TD_i)) + b * (eff_thick / TD_i)
        opac = opac.clone().detach()  # Convertir en tenseur
        opac = torch.clamp(opac, min=0.0, max=1.0)
        
        
        new_comp = comp + remaining * opac * color_i
        new_remaining = remaining * (1 - opac)
        return (new_comp, new_remaining), None

    init_state = (torch.zeros(3).cuda(), torch.tensor(1.0).cuda())
    
    comp, remaining = init_state
    for i in range(max_layers):
        (comp, remaining), _ = step_fn((comp, remaining), i)
    
    result = comp + remaining * background
    return result * 255.0


def composite_image_combined(pixel_height_logits, global_logits, tau_height, tau_global, gumbel_keys,
                             h, max_layers, material_colors, material_TDs, background, mode="continuous") -> torch.Tensor:
    """
    Apply composite_pixel_combined over the entire image.

    Args:
        pixel_height_logits: 2D array of pixel height logits.
        global_logits: Global logits for each layer.
        tau_height: Temperature for height compositing.
        tau_global: Temperature for material selection.
        gumbel_keys: Random keys per layer.
        h: Layer thickness.
        max_layers: Maximum number of layers.
        material_colors: Array of material colors.
        material_TDs: Array of material transmission/opacity parameters.
        background: Background color.
        mode: "continuous" or "discrete".

    Returns:
        The composite image (with values scaled to [0,255]).
    """
    # Utilisation de torch.vmap ou boucle explicite car torch.vmap est expérimental
    H, W = pixel_height_logits.shape
    output = torch.zeros((H, W, 3))
    return torch.vmap(torch.vmap(
        lambda ph_logit: composite_pixel_combined(
            ph_logit, global_logits, tau_height, tau_global, h, max_layers,
            material_colors, material_TDs, background, gumbel_keys, mode
        )
    ))(pixel_height_logits)