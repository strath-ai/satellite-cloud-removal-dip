from torch.utils.data import Dataset
import torch
        

class SingleImageDataset(Dataset):
    """Basic Dataset Class for a Single Image
    """

    def __init__(self, image,
                 input = None,
                 input_noise_std = 0.0,
                 repetition = 1):
        super().__init__()
        self.set_image(image)
        self.set_input(input)
        self.input_noise_std = input_noise_std
        self.repetition = repetition

    def __getitem__(self, index: int):
        if self.input is None:
            return self.image
        else:
            if self.input_noise_std != 0.0:
                return self.image, self.input + self.input_noise_std*torch.randn(self.input.shape).to(self.input.device)
            else:
                return self.image, self.input
    
    def set_image(self, x):
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x)
        if len(x.shape) < 4:
            x = torch.unsqueeze(x, 0)        
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        self.image = x
        
    def set_input(self, x):
        if x is not None:
            if type(x) is not torch.Tensor:
                x = torch.Tensor(x)
            if len(x.shape) < 4:
                x = torch.unsqueeze(x, 0)        
            if x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
        self.input = x
        
    def __len__(self) -> int:
        return self.repetition