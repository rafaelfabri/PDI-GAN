import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2

class ReadDataSet():
    def __init__(self, path_train = '', path_test = ''):
        self.path_train = path_train
        self.path_test = path_test

    def read(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((48,48)),
            v2.RandomHorizontalFlip(p=0.5),
            #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToTensor()
        ])

        dataset = torchvision.datasets.ImageFolder(root = self.path_train, transform=transform)

        dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

        return dataloader
