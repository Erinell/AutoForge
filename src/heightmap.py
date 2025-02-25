import torch
from skimage.color import rgb2lab
from itertools import permutations
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np

def init_height_map(target, max_layers, h):
    """
    Initialize pixel height logits based on the luminance of the target image.
    Assumes target is a torch.Tensor of shape (H, W, 3) in the range [0, 255].
    Uses the formula: L = 0.299*R + 0.587*G + 0.114*B.
    Args:
        target (torch.Tensor): The target image tensor with shape (H, W, 3).
    Returns:
        torch.Tensor: The initialized pixel height logits.
    """
    # Compute normalized luminance in [0,1]
    normalized_lum = (0.299 * target[..., 0] +
                     0.587 * target[..., 1] +
                     0.114 * target[..., 2]) / 255.0
    
    # To avoid log(0) issues, add a small epsilon
    eps = 1e-2

    # Convert target to numpy for KMeans (sklearn doesn't work with PyTorch tensors directly)
    target_np = target.cpu().numpy().reshape(-1, 3)

    # First kmeans: cluster image pixels into max_layers colors
    kmeans = KMeans(n_clusters=max_layers, random_state=0).fit(target_np)
    labels = kmeans.labels_
    labels = labels.reshape(target.shape[0], target.shape[1])
    centroids = kmeans.cluster_centers_

    # Define a simple luminance function (Rec.601)
    def luminance(col):
        return 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]

    # Step 2: Second clustering of centroids into bands
    num_bands = 10
    band_kmeans = KMeans(n_clusters=num_bands, random_state=0).fit(centroids)
    band_labels = band_kmeans.labels_

    # Group centroids by band and sort within each band by luminance
    bands = []
    for b in range(num_bands):
        indices = np.where(band_labels == b)[0]
        if len(indices) == 0:
            continue
        lum_vals = np.array([luminance(centroids[i]) for i in indices])
        sorted_indices = indices[np.argsort(lum_vals)]
        band_avg = np.mean(lum_vals)
        bands.append((band_avg, sorted_indices))

    # Step 3: Compute a representative color for each band in Lab space
    band_reps = []
    for _, indices in bands:
        band_avg_rgb = np.mean(centroids[indices], axis=0)
        band_avg_rgb_norm = band_avg_rgb / 255.0 if band_avg_rgb.max() > 1 else band_avg_rgb
        lab = rgb2lab(np.array([[band_avg_rgb_norm]]))[0, 0, :]
        band_reps.append(lab)

    # Step 4: Identify darkest and brightest bands based on L channel
    L_values = [lab[0] for lab in band_reps]
    start_band = np.argmin(L_values)
    end_band = np.argmax(L_values)

    # Step 5: Find the best ordering for the middle bands
    all_indices = list(range(len(bands)))
    middle_indices = [i for i in all_indices if i not in (start_band, end_band)]

    min_total_distance = float('inf')
    best_order = None
    total = len(middle_indices) * len(middle_indices)
    tbar = tqdm(permutations(middle_indices), total=total, desc="Finding best ordering for color bands:")
    for perm in tbar:
        candidate = [start_band] + list(perm) + [end_band]
        total_distance = 0
        for i in range(len(candidate) - 1):
            total_distance += np.linalg.norm(band_reps[candidate[i]] - band_reps[candidate[i + 1]])
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_order = candidate
            tbar.set_description(f"Finding best ordering for color bands: Total distance = {min_total_distance:.2f}")
        if tbar.n > 500000:
            break

    new_order = []
    for band_idx in best_order:
        new_order.extend(bands[band_idx][1].tolist())

    # Remap each pixel's label
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order)}
    new_labels = np.vectorize(lambda x: mapping[x])(labels)

    # Convert to float and normalize
    new_labels = new_labels.astype(np.float32) / new_labels.max()

    # Convert to PyTorch tensor and apply inverse sigmoid
    normalized_lum = torch.from_numpy(new_labels).to(dtype=torch.float64)
    pixel_height_logits = torch.log((normalized_lum + eps) / (1 - normalized_lum + eps))

    return pixel_height_logits