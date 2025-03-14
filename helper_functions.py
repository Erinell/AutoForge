import json
import os
import uuid

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import json
import pandas as pd
import numpy as np

import struct


def generate_stl(height_map, filename, background_height, scale=1.0):
    """
    Generate a binary STL file from a height map.

    Args:
        height_map (np.ndarray): 2D array representing the height map.
        filename (str): The name of the output STL file.
        background_height (float): The height of the background in the STL model.
        scale (float, optional): Scale factor for the x and y dimensions. Defaults to 1.0.
    """
    H, W = height_map.shape
    vertices = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            # Original coordinates: x = j*scale, y = (H - 1 - i), z = height + background
            vertices[i, j, 0] = j * scale
            vertices[i, j, 1] = (H - 1 - i)  # (Consider applying scale if needed)
            vertices[i, j, 2] = height_map[i, j] + background_height

    triangles = []

    def add_triangle(v1, v2, v3):
        """
        Add a triangle to the list of triangles.

        Args:
            v1 (np.ndarray): First vertex of the triangle.
            v2 (np.ndarray): Second vertex of the triangle.
            v3 (np.ndarray): Third vertex of the triangle.
        """
        triangles.append((v1, v2, v3))

    for i in range(H - 1):
        for j in range(W - 1):
            v0 = vertices[i, j]
            v1 = vertices[i, j + 1]
            v2 = vertices[i + 1, j + 1]
            v3 = vertices[i + 1, j]
            # Reversed order so normals face upward
            add_triangle(v2, v1, v0)
            add_triangle(v3, v2, v0)

    for j in range(W - 1):
        v0 = vertices[0, j]
        v1 = vertices[0, j + 1]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v0, v1, v1b)
        add_triangle(v0, v1b, v0b)
    for j in range(W - 1):
        v0 = vertices[H - 1, j]
        v1 = vertices[H - 1, j + 1]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v1, v0, v1b)
        add_triangle(v0, v0b, v1b)
    for i in range(H - 1):
        v0 = vertices[i, 0]
        v1 = vertices[i + 1, 0]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v1, v0, v1b)
        add_triangle(v0, v0b, v1b)
    for i in range(H - 1):
        v0 = vertices[i, W - 1]
        v1 = vertices[i + 1, W - 1]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v0, v1, v1b)
        add_triangle(v0, v1b, v0b)

    v0 = np.array([0, 0, 0], dtype=np.float32)
    v1 = np.array([(W - 1) * scale, 0, 0], dtype=np.float32)
    v2 = np.array([(W - 1) * scale, (H - 1) * scale, 0], dtype=np.float32)
    v3 = np.array([0, (H - 1) * scale, 0], dtype=np.float32)
    add_triangle(v2, v1, v0)
    add_triangle(v3, v2, v0)

    num_triangles = len(triangles)

    # Write the binary STL file.
    with open(filename, 'wb') as f:
        header_str = "Binary STL generated from heightmap"
        header = header_str.encode('utf-8')
        header = header.ljust(80, b' ')
        f.write(header)
        f.write(struct.pack('<I', num_triangles))
        for tri in triangles:
            v1, v2, v3 = tri
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            if norm == 0:
                normal = np.array([0, 0, 0], dtype=np.float32)
            else:
                normal = normal / norm
            f.write(struct.pack('<12fH',
                                normal[0], normal[1], normal[2],
                                v1[0], v1[1], v1[2],
                                v2[0], v2[1], v2[2],
                                v3[0], v3[1], v3[2],
                                0))



def generate_swap_instructions(discrete_global, discrete_height_image, h, background_layers, background_height, material_names):
    """
    Generate swap instructions based on discrete material assignments.

    Args:
        discrete_global (np.ndarray): Array of discrete global material assignments.
        discrete_height_image (np.ndarray): Array representing the discrete height image.
        h (float): Layer thickness.
        background_layers (int): Number of background layers.
        background_height (float): Height of the background in mm.
        material_names (list): List of material names.

    Returns:
        list: A list of strings containing the swap instructions.
    """
    L = int(np.max(np.array(discrete_height_image)))
    instructions = []
    if L == 0:
        instructions.append("No layers printed.")
        return instructions
    instructions.append("Start with your background color")
    for i in range(0, L):
        if i == 0 or int(discrete_global[i]) != int(discrete_global[i - 1]):
            ie = i
            instructions.append(f"At layer #{ie + background_layers} ({(ie * h) + background_height:.2f}mm) swap to {material_names[int(discrete_global[i])]}")
    instructions.append("For the rest, use " + material_names[int(discrete_global[L - 1])])
    return instructions


def initialize_pixel_height_logits(target):
    """
    Initialize pixel height logits based on the luminance of the target image.

    Assumes target is a Tensor of shape (H, W, 3) in the range [0, 255].
    Uses the formula: L = 0.299*R + 0.587*G + 0.114*B.

    Args:
        target (Tensor): The target image array with shape (H, W, 3).

    Returns:
        Tensor: The initialized pixel height logits.
    """

    # Compute normalized luminance in [0,1]
    normalized_lum = (0.299 * target[..., 0] +
                      0.587 * target[..., 1] +
                      0.114 * target[..., 2]) / 255.0
    # To avoid log(0) issues, add a small epsilon.
    eps = 1e-6
    pixel_height_logits = torch.log((normalized_lum + eps) / (1 - normalized_lum + eps))
    return pixel_height_logits


def load_materials_data(csv_filename):
    """
    Load the full material data from the CSV file.

    Args:
        csv_filename (str): Path to the CSV file containing material data.

    Returns:
        list: A list of dictionaries (one per material) with keys such as
              "Brand", "Type", "Color", "Name", "TD", "Owned", and "Uuid".
    """
    df = pd.read_csv(csv_filename)
    # Use a consistent key naming. For example, convert 'TD' to 'Transmissivity' and 'Uuid' to 'uuid'
    records = df.to_dict(orient="records")
    return records


def extract_filament_swaps(disc_global, disc_height_image, background_layers):
    """
    Given the discrete global material assignment (disc_global) and the discrete height image,
    extract the list of material indices (one per swap point) and the corresponding slider
    values (which indicate at which layer the material change occurs).

    Args:
        disc_global (np.ndarray): Discrete global material assignments.
        disc_height_image (np.ndarray): Discrete height image.
        background_layers (int): Number of background layers.

    Returns:
        tuple: A tuple containing:
            - filament_indices (list): List of material indices for each swap point.
            - slider_values (list): List of layer numbers where a material change occurs.
    """
    # L is the total number of layers printed (maximum value in the height image)
    L = int(np.max(np.array(disc_height_image)))
    filament_indices = []
    slider_values = []
    prev = int(disc_global[0])
    for i in range(L):
        current = int(disc_global[i])
        # If this is the first layer or the material changes from the previous layer…
        if current != prev:
            slider = (i + background_layers)-1
            slider_values.append(slider)
            filament_indices.append(prev)
        prev = current
    return filament_indices, slider_values


def generate_project_file(project_filename, args, disc_global, disc_height_image,
                          width_mm, height_mm, stl_filename, csv_filename):
    """
    Export a project file containing the printing parameters, including:
      - Key dimensions and layer information (from your command-line args and computed outputs)
      - The filament_set: a list of filament definitions (each corresponding to a color swap)
        where the same material may be repeated if used at different swap points.
      - slider_values: a list of layer numbers (indices) where a filament swap occurs.

    The filament_set entries are built using the full material data from the CSV file.

    Args:
        project_filename (str): Path to the output project file.
        args (Namespace): Command-line arguments containing printing parameters.
        disc_global (np.ndarray): Discrete global material assignments.
        disc_height_image (np.ndarray): Discrete height image.
        width_mm (float): Width of the model in millimeters.
        height_mm (float): Height of the model in millimeters.
        stl_filename (str): Path to the STL file.
        csv_filename (str): Path to the CSV file containing material data.
    """
    # Compute the number of background layers (as in your main())
    background_layers = int(args.background_height / args.layer_height)

    # Load full material data from CSV
    material_data = load_materials_data(csv_filename)

    # Extract the swap points from the discrete solution
    filament_indices, slider_values = extract_filament_swaps(disc_global, disc_height_image, background_layers)

    # Build the filament_set list. For each swap point, we look up the corresponding material from CSV.
    # Here we map CSV columns to the project file’s expected keys.
    filament_set = []
    for idx in filament_indices:
        mat = material_data[idx]
        filament_entry = {
            "Brand": mat["Brand"],
            "Color": mat[" Color"],
            "Name": mat[" Name"],
            # Convert Owned to a boolean (in case it is read as a string)
            "Owned": str(mat[" Owned"]).strip().lower() == "true",
            "Transmissivity": float(mat[" TD"]) if not float(mat[" TD"]).is_integer() else int(mat[" TD"]),
            "Type": mat[" Type"],
            "uuid": mat[" Uuid"]
        }
        filament_set.append(filament_entry)

    # add black as the first filament with background height as the first slider value
    filament_set.insert(0, {
            "Brand": "Autoforge",
            "Color": args.background_color,
            "Name": "Background",
            "Owned": False,
            "Transmissivity": 0.1,
            "Type": "PLA",
            "uuid": str(uuid.uuid4())
    })
    # add black to slider value
    slider_values.insert(0, (args.background_height//args.layer_height)-1)

    # reverse order of filament set
    filament_set = filament_set[::-1]

    # Build the project file dictionary.
    # Many keys are filled in with default or derived values.
    project_data = {
        "base_layer_height": args.layer_height,  # you may adjust this if needed
        "blue_shift": 0,
        "border_height": args.background_height,  # here we use the background height
        "border_width": 3,
        "borderless": True,
        "bright_adjust_zero": False,
        "brightness_compensation_name": "Standard",
        "bw_tolerance": 8,
        "color_match_method": 0,
        "depth_mode": 2,
        "edit_image": False,
        "extra_gap": 2,
        "filament_set": filament_set,
        "flatten": False,
        "full_range": False,
        "green_shift": 0,
        "gs_threshold": 0,
        "height_in_mm": height_mm,
        "hsl_invert": False,
        "ignore_blue": False,
        "ignore_green": False,
        "ignore_red": False,
        "invert_blue": False,
        "invert_green": False,
        "invert_red": False,
        "inverted_color_pop": False,
        "layer_height": args.layer_height,
        "legacy_luminance": False,
        "light_intensity": -1,
        "light_temperature": 1,
        "lighting_visualizer": 0,
        "luminance_factor": 0,
        "luminance_method": 2,
        "luminance_offset": 0,
        "luminance_offset_max": 100,
        "luminance_power": 2,
        "luminance_weight": 100,
        "max_depth": args.background_height + args.layer_height * args.max_layers,
        "median": 0,
        "mesh_style_edit": True,
        "min_depth": 0.48,
        "min_detail": 0.2,
        "negative": True,
        "red_shift": 0,
        "reverse_litho": True,
        "slider_values": slider_values,
        "smoothing": 0,
        "srgb_linearize": False,
        "stl": os.path.basename(stl_filename),
        "strict_tolerance": False,
        "transparency": True,
        "version": "0.7.0",
        "width_in_mm": width_mm
    }

    # Write out the project file as JSON
    with open(project_filename, "w") as f:
        json.dump(project_data, f, indent=4)



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


def load_materials(csv_filename):
    """
    Load material data from a CSV file.

    Args:
        csv_filename (str): Path to the hueforge CSV file containing material data.

    Returns:
        tuple: A tuple containing:
            - material_colors (Tensor): Array of material colors in float64.
            - material_TDs (Tensor): Array of material transmission/opacity parameters in float64.
            - material_names (list): List of material names.
            - colors_list (list): List of color hex strings.
    """
    df = pd.read_csv(csv_filename)
    material_names = [brand + " - " + name for brand, name in zip(df["Brand"].tolist(), df[" Name"].tolist())]
    material_TDs = (df[' TD'].astype(float)).to_numpy()
    colors_list = df[' Color'].tolist()
    # Use float64 for material colors.
    material_colors = torch.tensor([hex_to_rgb(color) for color in colors_list], dtype=torch.float64)
    material_TDs = torch.tensor(material_TDs, dtype=torch.float64)
    return material_colors, material_TDs, material_names, colors_list


def gumbel_softmax(logits, temperature, key, hard=False):
    """
    Compute the Gumbel-Softmax.

    Args:
        logits (Tensor): The input logits for the Gumbel-Softmax.
        temperature (float): The temperature parameter for the Gumbel-Softmax.
        hard (bool, optional): If True, return hard one-hot encoded samples. Defaults to False.

    Returns:
        Tensor: The Gumbel-Softmax samples.
    """
    
    y = F.softmax((logits + key) / temperature, dim=-1)
    if hard:
        y_hard = torch.zeros_like(y).scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
        y = y_hard + (y - y_hard).detach()
    return y


def srgb_to_linear_lab(rgb):
    """
    Convert sRGB (range [0,1]) to linear RGB.

    Args:
        rgb (Tensor): A Tensor array of shape (..., 3) representing the sRGB values.

    Returns:
        Tensor: A Tensor array of shape (..., 3) representing the linear RGB values.
    """
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


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
    return torch.stack([X, Y, Z], dim=-1)


def xyz_to_lab(xyz):
    """
    Convert XYZ to CIELAB. Assumes D65 reference white.

    Args:
        xyz (Tensor): A Tensor array of shape (..., 3) representing the XYZ values.

    Returns:
        Tensor: A Tensor array of shape (..., 3) representing the CIELAB values.
    """
    xyz_ref = torch.tensor([0.95047, 1.0, 1.08883], dtype=xyz.dtype, device=xyz.device)
    xyz = xyz / xyz_ref
    delta = 6/29
    f = torch.where(xyz > delta**3, xyz ** (1/3), (xyz / (3 * delta**2)) + (4/29))
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


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
    return (1 - beta) * x + beta * torch.round(x)