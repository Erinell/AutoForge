import numpy as np
from torch import stack, tensor, where, round

def hex_to_rgb(hex_str):
    """
    Convert a hex color string to a normalized RGB list.

    Args:
        hex_str (str): The hex color string (e.g., '#RRGGBB').

    Returns:
        list: A list of three floats representing the RGB values normalized to [0, 1].
    """
    hex_str = hex_str.lstrip('#')
    return [int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

def srgb_to_linear_lab(rgb):
    """
    Convert sRGB (range [0,1]) to linear RGB.

    Args:
        rgb (Tensor): A Tensor array of shape (..., 3) representing the sRGB values.

    Returns:
        Tensor: A Tensor array of shape (..., 3) representing the linear RGB values.
    """
    return where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

def linear_to_xyz(rgb_linear):
    """
    Convert linear RGB to XYZ using sRGB D65.

    Args:
        rgb_linear (Tensor): A Tensor array of shape (..., 3) representing the linear RGB values.

    Returns:
        Tensor: A Tensor array of shape (..., 3) representing the XYZ values.
    """
    R = rgb_linear[..., 0]
    G = rgb_linear[..., 1]
    B = rgb_linear[..., 2]
    X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B
    Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
    Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B
    return stack([X, Y, Z], dim=-1)

def xyz_to_lab(xyz):
    """
    Convert XYZ to CIELAB. Assumes D65 reference white.

    Args:
        xyz (Tensor): A Tensor array of shape (..., 3) representing the XYZ values.

    Returns:
        Tensor: A Tensor array of shape (..., 3) representing the CIELAB values.
    """
    xyz_ref = tensor([0.95047, 1.0, 1.08883], dtype=xyz.dtype, device=xyz.device)
    xyz = xyz / xyz_ref
    delta = 6/29
    f = where(xyz > delta**3, xyz ** (1/3), (xyz / (3 * delta**2)) + (4/29))
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return stack([L, a, b], dim=-1)

def rgb_to_lab(rgb):
    """
    Convert an sRGB image (values in [0,1]) to CIELAB.

    Args:
        rgb (Tensor): A Tensor array of shape (..., 3) representing the sRGB values.

    Returns:
        Tensor: A Tensor array of shape (..., 3) representing the CIELAB values.
    """
    rgb_linear = srgb_to_linear_lab(rgb)
    xyz = linear_to_xyz(rgb_linear)
    lab = xyz_to_lab(xyz)
    return lab

def adaptive_round(x, tau, high_tau=1.0, low_tau=0.01, temp=0.1):
    """
    Compute a soft (adaptive) rounding of x.

    When tau is high (>= high_tau) returns x (i.e. no rounding).
    When tau is low (<= low_tau) returns round(x).
    In between, linearly interpolates between x and round(x).

    Args:
        x (jnp.ndarray): The input array to be rounded.
        tau (float): The temperature parameter controlling the degree of rounding.
        high_tau (float, optional): The high threshold for tau. Defaults to 1.0.
        low_tau (float, optional): The low threshold for tau. Defaults to 0.01.
        temp (float, optional): A temperature parameter for interpolation. Defaults to 0.1.

    Returns:
        Tensor: The adaptively rounded array.
    """
    beta = np.clip((high_tau - tau) / (high_tau - low_tau), 0.0, 1.0)
    return (1 - beta) * x + beta * round(x)

def heightmap_to_viridis(height_map, max_value):
    """
    Convert height map (range [0,1]) to viridis-like colormap

    Increase red for higher value
    Peak green in middle
    Decrease blue from start

    Args:
        height_map (np.ndarray): The height map in range [0, 1]
        max_value (float): The max value to 

    Returns:
        (np.ndarray): The rgb image
    """
    
    normalized = np.clip(height_map / max_value, 0, 1) * 255
    normalized = normalized.astype(np.uint8)
    r = np.clip(normalized * 2, 0, 255)
    g = np.clip(normalized * 3 - 255, 0, 255)
    b = np.clip(255 - normalized * 2, 0, 255)
    
    return np.stack([r, g, b], axis=-1)
