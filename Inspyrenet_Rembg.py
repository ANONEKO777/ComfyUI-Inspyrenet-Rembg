from PIL import Image
import torch
import numpy as np
from transparent_background import Remover
from tqdm import tqdm


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class InspyrenetRembg:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "torchscript_jit": (["default", "on"],)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, torchscript_jit):
        if (torchscript_jit == "default"):
            remover = Remover()
        else:
            remover = Remover(jit=True)

        processed_images = []
        processed_masks = []

        for img in tqdm(image, "Inspyrenet Rembg"):
            orig_image = tensor2pil(img)

            rgba_result = remover.process(orig_image, type='rgba')
            mask_pil = rgba_result.split()[3]
            # Create a new transparent background image and paste the original image with the mask
            # I found that this step is necessary when exporting videos using VideoHelper, otherwise the exported video will still have a background
            new_im = Image.new("RGBA", orig_image.size, (0,0,0,0))
            new_im.paste(orig_image, mask=mask_pil)
            new_im_tensor = pil2tensor(new_im)
            mask_tensor = pil2tensor(mask_pil)

            processed_images.append(new_im_tensor)
            processed_masks.append(mask_tensor)

        img_stack = torch.cat(processed_images, dim=0)
        mask_stack = torch.cat(processed_masks, dim=0)

        return (img_stack, mask_stack)

class InspyrenetRembgAdvanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "torchscript_jit": (["default", "on"],)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, torchscript_jit, threshold):
        if (torchscript_jit == "default"):
            remover = Remover()
        else:
            remover = Remover(jit=True)

        processed_images = []
        processed_masks = []

        for img in tqdm(image, "Inspyrenet Rembg"):
            orig_image = tensor2pil(img)

            rgba_result = remover.process(orig_image, type='rgba', threshold=threshold)
            mask_pil = rgba_result.split()[3]
            # Create a new transparent background image and paste the original image with the mask
            # I found that this step is necessary when exporting videos using VideoHelper, otherwise the exported video will still have a background
            new_im = Image.new("RGBA", orig_image.size, (0,0,0,0))
            new_im.paste(orig_image, mask=mask_pil)
            new_im_tensor = pil2tensor(new_im)
            mask_tensor = pil2tensor(mask_pil)

            processed_images.append(new_im_tensor)
            processed_masks.append(mask_tensor)

        img_stack = torch.cat(processed_images, dim=0)
        mask_stack = torch.cat(processed_masks, dim=0)

        return (img_stack, mask_stack)