import torch
from vit_pytorch import ViT
from torch.utils.mobile_optimizer import optimize_for_mobile

class Optimized_ViT(nn.Module):
    def __init__(self,n_classes):
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
        self.scripted_jit_model = torch.jit.script(self.model) # scripting ViT 

        backend = "x86" 
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend

        self.quantized_model = torch.quantization.quantize_dynamic(self.model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        self.scripted_quantized_model = torch.jit.script(self.quantized_model) # quantizing ViT
        
        self.optimized_scripted_quantized_model = optimize_for_mobile(self.scripted_quantized_model)


    def forward(self,x):
        return self.optimized_scripted_quantized_model(x)



