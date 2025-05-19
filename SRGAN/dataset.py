import config
import os 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyImageFolder(Dataset):
    def __init__(self, root_dir, new_size):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.new_size = new_size
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])
        image = Image.open(os.path.join(root_and_dir, img_file))
        image = image.resize(self.new_size)
        image = image.convert("RGB")
        image = np.array(image)
        #print(image.shape)

        high_res = config.highres_transform(image=image)["image"]
        low_res = config.lowres_transform(image=image)["image"]
        return high_res, low_res
    


def test():
    dataset = MyImageFolder(root_dir="/home/rafaelfabrichimidt/Documentos/projetos/Mestrado/PDI/artigo/Images/train__")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)
    print(loader)
    for low_res, high_res in loader:
         print(low_res.shape)
         print(high_res.shape)


if __name__ == "__main__":
    test()