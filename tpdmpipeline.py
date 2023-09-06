from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from diffusers import DiffusionPipeline
from diffusers.utils import randn_tensor
from diffusers.pipeline_utils import ImagePipelineOutput


class TPDMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler, gan_generator):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler, gan_generator=gan_generator)

    def get_device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            if getattr(module, "device", None) is not None:
                return module.device

        return torch.device("cpu")
    
    @torch.no_grad()
    def __call__(self,
                 batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        truncated_step: int = 499,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        show_progress: bool = True,
        device=None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            truncated_step (`int`, *optional*, defaults to 499):
                T_Truc, step to start denoising from, this would create denoising step [truncated_step+1, truncated_step,...,0]
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if device is None:
            device = self.get_device()

        # Sample gaussian noise to begin loop
        image_shape = (
            batch_size,
            512,
        )

        image = randn_tensor(image_shape, generator=generator, device=device)
            
        # set step values
        self.scheduler.set_timesteps(timesteps=list(range(truncated_step)[::-1]))

        label = torch.zeros([1, self.gan_generator.c_dim], device=device) 
        # 1. predict noise to T_Trunc with GAN
        image = self.gan_generator(image, label)
        pbar = self.progress_bar(self.scheduler.timesteps) if show_progress else self.scheduler.timesteps
        for t in pbar:
            # 2. predict noise model_output
            model_output = self.unet(image, t).sample

            # 3. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu()        

        if output_type == "pil":
            image = self.numpy_to_pil(image.permute(0, 2, 3, 1).numpy())
        elif output_type == "numpy":
            image = image.permute(0, 2, 3, 1).numpy()
        elif output_type == "torch":
            pass

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
