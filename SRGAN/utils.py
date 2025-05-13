import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)
    print(files)
    gen.eval()
    for file in files:
        print(low_res_folder + file)
        image = Image.open(low_res_folder + file)
        with torch.no_grad():
            upscaled_img = gen(
                config.test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        print(upscaled_img  * 0.5 + 0.5)
        save_image(upscaled_img * 0.5 + 0.5, f"/home/rafaelfabrichimidt/Documentos/projetos/Mestrado/PDI/artigo/Images/saved/{file}")
    gen.train()