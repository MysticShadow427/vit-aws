import torch
from torch import nn
from vit_pytorch import ViT
from torch.utils.mobile_optimizer import optimize_for_mobile

class Optimized_ViT(nn.Module):
    def __init__(self,n_classes,optimize_before_training = False):
        super().__init__()
        self.model = ViT(
            image_size = 64,
            patch_size = 8,
            num_classes = n_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        self.obt = optimize_before_training

        self.vit = None

        if self.obt:
            self.vit = optimize_model(self.model)
        else:
            self.vit = self.model

    def forward(self,x):
        return self.vit(x)

def optimize_model(model):
    backend = "x86" 
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend

    quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8) # quantizing ViT
    scripted_quantized_model = torch.jit.script(quantized_model)  # scripting ViT 
    
    optimized_scripted_quantized_model = optimize_for_mobile(scripted_quantized_model)
    return optimized_scripted_quantized_model

