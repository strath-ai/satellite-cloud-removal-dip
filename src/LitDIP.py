import pytorch_lightning as pl
from .DeepImagePrior.SkipNetwork import *
from .Dataset import *

import torch
import torch.nn.functional as F

class LitDIP(pl.LightningModule):

    def __init__(self,
                 target_channels = 3,
                 latent_seed = 'meshgrid',
                 sigmoid_output = True,
                 lr = 2e-2,
                 regularization_noise_std = 1e-1,
                 epoch_steps = 1000,
                 logging = False
                ):
        super().__init__()
        
        self.input = None        
        self.target = None
        self.mask = None
        self.latent_seed = latent_seed
        self.sigmoid_output = sigmoid_output
        self.logging = logging
        self.epoch_steps = epoch_steps
        self.target_channels = target_channels
        self.regularization_noise_std = regularization_noise_std
        self.lr = lr

        if isinstance(self.latent_seed, list) or isinstance(self.latent_seed, tuple):
            self.input_depth = self.latent_seed[1]
        elif self.latent_seed == 'meshgrid':
            self.input_depth = 2
        else:
            self.input_depth = np.sum(self.target_channels)
        
        # Model
        self.init_model()           
       
    def image_check(self, x):
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x)
        if len(x.shape) < 4:
            x = torch.unsqueeze(x, 0)  
        if np.argmin(x.shape[1:]) == 2:
            x = x.permute(0, 3, 1, 2)
        return x.to(self.device)
    
    def init_model(self):
        # source from original DIP github repository
        self.model = SkipNetwork(in_channels = self.input_depth,
                                 out_channels = int(np.sum(self.target_channels)),
                                 hidden_dims_down = [16, 32, 64, 128, 128, 128],
                                 hidden_dims_up = [16, 32, 64, 128, 128, 128],
                                 hidden_dims_skip = [0, 0, 0, 0, 0, 0],
                                 filter_size_down = 3,
                                 filter_size_up = 5,
                                 filter_skip_size = 0,
                                 sigmoid_output = self.sigmoid_output,
                                 bias = True,
                                 padding_mode='zero',
                                 upsample_mode='nearest',
                                 downsample_mode='stride',
                                 act_fun='LeakyReLU',
                                 need1x1_up=True
                                )    
    
    def set_input(self, x):
        if type(x) is not list:
            self.input = self.image_check(x)
        else:
            x_c = []
            for x_i in x:
                x_i_c = self.image_check(x_i)
                x_c.append(x_i_c)
            self.input = torch.cat(x_c, 1).float()
            
    def set_target(self, x):
        
        # set target
        if type(x) is not list:            
            self.target_channels = x.shape[-1]
            self.target = self.image_check(x)
        else:
            x_c = []
            self.target_channels = []
            for x_i in x:                
                x_i_c = self.image_check(x_i)
                x_c.append(x_i_c)
                self.target_channels.append(x_i.shape[-1])
            self.target = torch.cat(x_c, 1)
       
        # at the same time, initialize input
        if isinstance(self.latent_seed, list) or isinstance(self.latent_seed, tuple):
            self.set_input(self.latent_seed[0]*torch.rand(self.latent_seed[1], *self.target.shape[-2:]))
            
        elif self.latent_seed == 'meshgrid':
            X,Y = torch.meshgrid(torch.arange(0, 1, 1/self.target.shape[-2]),
                                 torch.arange(0, 1, 1/self.target.shape[-1])
                                )
            meshgrid = torch.cat([X[None,:], Y[None,:]])            
            self.set_input(meshgrid)
            
        else:
            self.set_input(self.target) 
            
        # reinit model
        self.init_model()   
            
        self.train_dataset = SingleImageDataset(self.target,
                                                input = self.input,
                                                input_noise_std = self.regularization_noise_std,
                                                repetition = self.epoch_steps)
        
    def set_mask(self, x):       
        if type(x) is not list:
            flat_mask = torch.tensor(x).view(1,x.shape[-2], x.shape[-1])
            self.mask = torch.stack(self.target_channels*[flat_mask.clone()], 1)
        else:
            masks = []
            #x = np.array(x)
            for c_idx in range(len(x)):
                x_i = x[c_idx]
                flat_mask = torch.tensor(x_i).view(1, x_i.shape[0], x_i.shape[1])
                deep_mask = torch.stack(self.target_channels[c_idx]*[flat_mask.clone()], 1)
                masks.append(deep_mask)
            self.mask = torch.cat(masks, 1).bool()#.permute(0,2,3,1)
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=1,
                                                   num_workers=1,
                                                   shuffle=False
                                                  )
        return train_loader
    
    def output(self):
        out = self.forward(self.input.to(self.device)).detach()

        if type(self.target_channels) is list:
            out_list = []
            offset_ch = 0
            for ch in self.target_channels:
                out_list.append(out[0,offset_ch:offset_ch+ch,...].permute(1,2,0).cpu().numpy())
                offset_ch += ch
            return out_list
        else:
            return out[0].permute(1,2,0).cpu().numpy()

    def forward(self, input):
        return self.model.forward(input)
            
    def validation_epoch_end(self, outputs):
        if self.logging:
            avg_loss = np.mean([x['val_loss'] for x in outputs])

            # sample representations                
            tensorboard = self.logger.experiment
            a = self.output()

            tensorboard.add_image('Generated',
                                  a,
                                  global_step = self.current_epoch,
                                  dataformats='HWC')

            self.log('val_loss', torch.tensor(avg_loss))      
        
    def get_loss(self, output, target, mask = None):
        
        loss = F.mse_loss(output, target, reduction = 'none')
        
        if mask is not None:
            loss = loss[mask]
        
        return {'loss': loss.mean()}
    
    def on_pretrain_routine_end(self):
        self.input = self.input.to(self.device)
        self.target = self.target.to(self.device)

    def training_step(self, batch, batch_idx):
            
        if self.regularization_noise_std != 0.0:
            input = self.input + self.regularization_noise_std*torch.randn(self.input.shape).to(self.input.device)
        else:
            input = self.input
            
        out = self.forward(input)

        return self.get_loss(out, self.target, self.mask)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)