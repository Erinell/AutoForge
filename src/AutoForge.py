import math
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ui_form import Ui_main

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QSize, Signal, QThread

from src.composite import composite_image_combined
from src.loss import loss_fn, loss_fn_perceptual
from src.auto_forge import create_update_step, discretize_solution_pytorch
from src.utils import heightmap_to_viridis, hex_to_rgb
from src.helper_functions import generate_project_file, generate_stl, generate_swap_instructions, initialize_pixel_height_logits, load_materials
from src.heightmap import init_height_map

class Callback:
    iteration: int = 0
    iterations: int = 0
    loss: float = 0.0
    best_loss: float = 0.0
    tau_height: float = 0.0
    highest_layer: int = 0
    comp_im : QImage = None
    disc_comp_im: QImage = None
    best_comp_im: QImage = None
    height_map_im: QImage = None

class AutoForge(QThread):
    update_signal = Signal(Callback)
    
    def __init__(self, ui):
        super().__init__()

        self.input_image: str = None
        self.csv_file: str = None
        self.output_folder: str = "outputs"
        self.iterations: int = 20000
        self.learning_rate: float = 5e-3
        self.layer_height: float = 0.04
        self.max_layers: int = 75
        self.background_height: float = 0.4
        self.background_color: str = "#8e9089"
        self.max_size: int = 512
        self.decay: float = 0.0001
        self.loss: str = "mse"
        self.save_interval_pct: float = 20

        self.target_image = None
        self.material_colors: torch.Tensor = None
        self.material_TDs: torch.Tensor = None
        self.material_names: list = None
        
        self.ui: Ui_main = ui
        self.signals = Callback()
        
    def _sanity_check(self):
        os.makedirs(self.output_folder, exist_ok=True)
        
        assert (self.background_height / self.layer_height).is_integer(), "Background height must be divisible by layer height."
        assert self.max_size > 0, "max_size must be positive."
        assert self.iterations > 0, "iterations must be positive."
        assert self.learning_rate > 0, "learning_rate must be positive."
        assert self.layer_height > 0, "layer_height must be positive."

    def load_target_image(self):
        img = cv2.imread(self.input_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img.shape
        
        if w_img >= h_img:
            self.new_w = self.max_size
            self.new_h = int(self.max_size * h_img / w_img)
        else:
            self.new_h = self.max_size
            self.new_w = int(self.max_size * w_img / h_img)
            
        scale = self.new_w / self.new_h
        self.target_image = cv2.resize(img, (self.new_w, self.new_h))
        q_img = QImage(self.target_image.data, self.new_w, self.new_h, QImage.Format.Format_RGB888)
        self.ui.targetPic.setFixedSize(QSize(min(self.ui.targetPic.width(), self.new_w), min(self.ui.targetPic.width(), self.new_h) / scale))
        self.ui.targetPic.setPixmap(QPixmap.fromImage(q_img))
        
        
        
    def load_materials(self, blacklist: list = None):
        material_colors, material_TDs, material_names, _ = load_materials(self.csv_file, blacklist)

        self.material_colors = material_colors.clone().detach()
        self.material_TDs = material_TDs.clone().detach()
        self.material_names = material_names
        
    def run(self):
        # run optimiser
        self._sanity_check()
        
        target = torch.tensor(self.target_image).cuda()
        self.background_layers = int(self.background_height // self.layer_height)
    
        self.background = torch.tensor(hex_to_rgb(self.background_color)).cuda()
        h, w = self.target_image.shape[:2]

        best_params, _ = self.run_optimizer(
            target.clone().detach(), h, w, self.max_layers, self.layer_height,
            self.material_colors.clone().detach(), self.material_TDs.clone().detach(), self.background,
            self.iterations,
            self.learning_rate,
            self.decay,
            loss_function=loss_fn if self.loss == "mse" else loss_fn_perceptual,
            visualize=False,
            output_folder=self.output_folder,
            save_interval_pct=self.save_interval_pct if self.save_interval_pct > 0 else None,
            img_width=w,
            img_height=h,
            background_height=self.background_height,
            material_names=self.material_names,
            csv_file=self.csv_file
        )
        
        self.done(best_params)
        
    def done(self, best_params):
        gumbel_keys_disc = torch.randn(self.decay, self.material_colors.shape[0])
        tau_global_disc = self.decay
        disc_global, disc_height_image = discretize_solution_pytorch(best_params, tau_global_disc, gumbel_keys_disc, self.layer_height, self.max_layers)
        disc_comp = composite_image_combined(best_params['pixel_height_logits'], best_params['global_logits'],
                                                tau_global_disc, tau_global_disc, gumbel_keys_disc,
                                                self.layer_height, self.max_layers, self.material_colors, self.material_TDs, self.background, mode="discrete")
        discrete_comp_np = np.clip(disc_comp.cpu().numpy(), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.output_folder, "discrete_comp.jpg"),
                    cv2.cvtColor(discrete_comp_np, cv2.COLOR_RGB2BGR))

        height_map_mm = (disc_height_image.astype(np.float32)) * self.layer_height
        stl_filename = os.path.join(self.output_folder, "final_model.stl")
        generate_stl(height_map_mm, stl_filename, self.background_height, scale=1.0)

        swap_instructions = generate_swap_instructions(disc_global, disc_height_image,
                                                    self.layer_height, self.background_layers, self.background_height, self.material_names)
        instructions_filename = os.path.join(self.output_folder, "swap_instructions.txt")
        with open(instructions_filename, "w") as f:
            for line in swap_instructions:
                f.write(line + "\n")

        project_filename = os.path.join(self.output_folder, "project_file.hfp")
        generate_project_file(project_filename, 
                            self.background_color,
                            self.background_height,
                            self.layer_height,
                            self.max_layers,
                            disc_global,
                            disc_height_image,
                            self.new_w, self.new_h, stl_filename, self.csv_file)
        print("Project file saved to", project_filename)
        print("All outputs saved to", self.output_folder)
        print("Happy printing!")
        
    
    def run_optimizer(self, target, H, W, max_layers, h, material_colors, material_TDs, background,
                  num_iters, learning_rate, decay_v, loss_function, visualize=False,
                  output_folder=None, save_interval_pct=None,
                  img_width=None, img_height=None, background_height=None,
                  material_names=None, csv_file=None):
        """
        Run the optimization loop to learn per-pixel heights and per-layer material assignments.

        Args:
            target (Tensor): The target image array.
            H (int): Height of the target image.
            W (int): Width of the target image.
            max_layers (int): Maximum number of layers.
            h (float): Layer thickness.
            material_colors (Tensor): Array of material colors.
            material_TDs (Tensor): Array of material transmission/opacity parameters.
            background (Tensor): Background color.
            num_iters (int): Number of optimization iterations.
            learning_rate (float): Learning rate for optimization.
            decay_v (float): Final tau value for Gumbel-Softmax.
            loss_function (callable): The loss function to compute gradients.
            visualize (bool, optional): Enable visualization during optimization. Defaults to False.

        Returns:
            tuple: A tuple containing the best parameters and the best composite image.
        """
        num_materials = material_colors.shape[0]
        
        global_logits = torch.ones((max_layers, num_materials)) * -1.0
        for i in range(max_layers):
            global_logits[i, i % num_materials] = 1.0
        global_logits += torch.rand(global_logits.shape) * 0.2 - 0.1  # Uniform [-0.1, 0.1]

        # pixel_height_logits = initialize_pixel_height_logits(target).cuda()
        pixel_height_logits = init_height_map(target,max_layers,h).cuda()
        params = {
            'global_logits': global_logits.requires_grad_(True),
            'pixel_height_logits': pixel_height_logits.requires_grad_(True)
        }
        
        optimizer = torch.optim.Adam([params['global_logits'], params['pixel_height_logits']], lr=learning_rate)
        
        update_step = create_update_step(optimizer, loss_function, h, max_layers, material_colors, material_TDs, background)

        warmup_steps = num_iters // 4
        decay_rate = -math.log(decay_v) / (num_iters - warmup_steps)

        def get_tau(i, tau_init=1.0, tau_final=decay_v, decay_rate=decay_rate):
            """
            Compute the tau value for the current iteration.

            Args:
                i (int): Current iteration.
                tau_init (float): Initial tau value.
                tau_final (float): Final tau value.
                decay_rate (float): Decay rate for tau.

            Returns:
                float: The computed tau value.
            """
            if i < warmup_steps:
                return tau_init
            else:
                return max(tau_final, tau_init * math.exp(-decay_rate * (i - warmup_steps)))

        best_params = None
        best_loss = float('inf')
        best_params_since_last_save = None
        best_loss_since_last_save = float('inf')
        # Determine the checkpoint interval (in iterations) based on percentage progress.
        checkpoint_interval = int(num_iters * save_interval_pct / 100) if save_interval_pct is not None else None

        tbar = tqdm(range(num_iters), disable=False)
        for i in tbar:
            tau_height = get_tau(i, tau_init=1.0, tau_final=decay_v, decay_rate=decay_rate)
            tau_global = get_tau(i, tau_init=1.0, tau_final=decay_v, decay_rate=decay_rate)
            gumbel_keys = torch.randn(max_layers, num_materials)
            params, loss = update_step(params, target, tau_height, tau_global, gumbel_keys)
            
            disc_comp = composite_image_combined(params['pixel_height_logits'], params['global_logits'],
                                                    decay_v, decay_v, gumbel_keys,
                                                    h, max_layers, material_colors, material_TDs, background,
                                                    mode="discrete")
            loss_val = torch.mean((disc_comp - target) ** 2)
            if loss_val < best_loss_since_last_save:
                best_loss_since_last_save = loss_val
                best_params_since_last_save = {k: v.clone().detach() for k, v in params.items()}
                
            if loss_val < best_loss or best_params is None:
                best_loss = loss_val
                best_params = {k: v.clone().detach() for k, v in params.items()}

                comp = composite_image_combined(best_params['pixel_height_logits'], best_params['global_logits'],
                                                    tau_height, tau_global, gumbel_keys,
                                                    h, max_layers, material_colors, material_TDs, background, mode="continuous")
                comp_np = np.clip(comp.cpu().detach().numpy(), 0, 255).astype(np.uint8)
                self.signals.best_comp_im = QImage(comp_np, comp_np.shape[1], comp_np.shape[0], QImage.Format.Format_RGB888)
                
                disc_comp_np = np.clip(disc_comp.cpu().detach().numpy(), 0, 255).astype(np.uint8)
                self.signals.disc_comp_im = QImage(disc_comp_np, disc_comp_np.shape[1], disc_comp_np.shape[0], QImage.Format.Format_RGB888)
            
            if i % 50 == 0:
                comp = composite_image_combined(best_params['pixel_height_logits'], best_params['global_logits'],
                                                    tau_height, tau_global, gumbel_keys,
                                                    h, max_layers, material_colors, material_TDs, background, mode="continuous")
                comp_np = np.clip(comp.cpu().numpy(), 0, 255).astype(np.uint8)
                self.signals.comp_im = QImage(comp_np, comp_np.shape[1], comp_np.shape[0], QImage.Format.Format_RGB888)
                
                
                height_map = (max_layers * h) * torch.sigmoid(best_params['pixel_height_logits']).cuda()
                height_map_np = height_map.cpu().detach().numpy()

                viridis_image = heightmap_to_viridis(height_map_np, max_layers * h)
                self.signals.height_map_im = QImage(viridis_image, viridis_image.shape[1], viridis_image.shape[0], QImage.Format.Format_RGB888)
                
                highest_layer = np.max(height_map_np)
                self.signals.highest_layer = highest_layer

            if checkpoint_interval is not None and (i + 1) % checkpoint_interval == 0 and i > 10:
                print("Saving intermediate outputs...")
                self.save_intermediate_outputs(i, best_params_since_last_save, tau_global, gumbel_keys, h, max_layers,
                                        material_colors, material_TDs, background,
                                        output_folder, W, H, background_height, material_names, csv_file)
                best_params_since_last_save = None
                best_loss_since_last_save = float('inf')
            # tbar.set_description(f"loss = {loss_val:.4f}, Best Loss = {best_loss:.4f}")
            
            self.signals.iteration = i
            self.signals.iterations = num_iters
            self.signals.loss = loss_val
            self.signals.best_loss = best_loss
            self.signals.tau_height = tau_height
            self.update_signal.emit(self.signals)
        
        best_comp = composite_image_combined(best_params['pixel_height_logits'], best_params['global_logits'],
                                                tau_height, tau_global, gumbel_keys,
                                                h, max_layers, material_colors, material_TDs, background, mode="continuous")
        return best_params, best_comp


    def save_intermediate_outputs(self, iteration, params, tau_global, gumbel_keys, h, max_layers,
                                material_colors, material_TDs, background,
                                output_folder, img_width, img_height, background_height,
                                material_names, csv_file):
        disc_comp = composite_image_combined(
            params['pixel_height_logits'], params['global_logits'],
            tau_global, tau_global, gumbel_keys,
            h, max_layers, material_colors, material_TDs, background, mode="discrete")
        discrete_comp_np = np.clip(disc_comp.cpu().numpy(), 0, 255).astype(np.uint8)
        image_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_comp.jpg")
        cv2.imwrite(image_filename, cv2.cvtColor(discrete_comp_np, cv2.COLOR_RGB2BGR))

        disc_global, disc_height_image = discretize_solution_pytorch(params, tau_global, gumbel_keys, h, max_layers)
        height_map_mm = (disc_height_image.astype(np.float32)) * h
        stl_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_model.stl")
        generate_stl(height_map_mm, stl_filename, background_height, scale=1.0)

        # Generate swap instructions.
        background_layers = int(background_height // h)
        swap_instructions = generate_swap_instructions(disc_global, disc_height_image,
                                                    h, background_layers, background_height, material_names)
        instructions_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_swap_instructions.txt")
        with open(instructions_filename, "w") as f:
            for line in swap_instructions:
                f.write(line + "\n")

        # Generate the project file.
        project_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_project.hfp")
        generate_project_file(project_filename,
                            self.background_color,
                            background_height, self.layer_height, max_layers,
                            disc_global,
                            disc_height_image,
                            img_width, img_height, stl_filename, csv_file)

        

