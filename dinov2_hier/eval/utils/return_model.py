import torch
import timm
import os
import torch.nn as nn


from torchvision import transforms
from torchvision.models import resnet


def get_models(modelname, image_size, saved_model_path=None):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

 

    # --- our finetuned models
    if modelname.lower().replace("_reg","") in ["dinov2_vits14","dinov2_vitb14","dinov2_vitl14","dinov2_vitg14"]:
        model = get_dino_finetuned_downloaded(saved_model_path,modelname,image_size)

    elif modelname.lower() in ["dinobloom_s","dinobloom_b","dinobloom_l","dinobloom_g"]:
        modelname_dict= {"dinobloom_s":"dinov2_vits14", "dinobloom_b":"dinov2_vitb14", "dinobloom_l":"dinov2_vitl14", "dinobloom_g":"dinov2_vitg14"}
        modelname = modelname_dict[modelname]
        model = get_dino_finetuned_downloaded(saved_model_path,modelname,image_size)

    elif modelname.lower() in ["dinov2_vits14_classifier","dinov2_vitb14_classifier","dinov2_vitl14_classifier","dinov2_vitg14_classifier"]:
        model = get_dino_student_classifier(saved_model_path, modelname)

        
    else: 
        raise ValueError(f"Model {modelname} not found")

    model = model.to(device)
    model.eval()

    return model

class DINOClassifier(nn.Module):
    def __init__(self, backbone,mlp):
        super().__init__()
        self.backbone=backbone
        self.mlp=mlp

    def forward(self,x):
        x=self.backbone(x)
        return x,self.mlp(x)

class MLP(nn.Module):
    """
    (from pytorch metric learning package)
    layer_sizes[0] is the dimension of the input
    layer_sizes[-1] is the dimension of the output
    """
    def __init__(self, layer_sizes, final_relu=False, grad_rev=False):
        super().__init__()

        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        if grad_rev:
            layer_list.append(RevGrad())
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))

        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)
    

def get_uni(saved_model_path):
    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
    model.load_state_dict(torch.load(os.path.join(saved_model_path, "pytorch_model.bin"), map_location="cpu"), strict=True)
    return model

def get_dino_student_classifier(model_path, modelname,n_classes=27):
    modelname=modelname.replace("_classifier","")
    model = torch.hub.load("facebookresearch/dinov2",modelname )
    pretrained = torch.load(model_path, map_location=torch.device("cpu"))
    # make correct state dict for loading
    state_dict_backbone = {}
    state_dict_mlp = {}
    for key, value in pretrained["student"].items():
        if "dino_head" in key or "ibot_head" in key or "supervised_head_1." in key:
            pass
        elif "supervised_head_0." in key:
            new_key = key.replace("supervised_head_0.", "")
            state_dict_mlp[new_key] = value
        elif "supervised_head." in key:
            new_key = key.replace("supervised_head.", "")
            state_dict_mlp[new_key] = value
        else:
            new_key = key.replace("backbone.", "")
            state_dict_backbone[new_key] = value
    # change shape of pos_embed
    input_dims = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }
    embed_dim=input_dims[modelname]
    pos_embed = nn.Parameter(torch.zeros(1, 257,embed_dim ))
    model.pos_embed = pos_embed
    # load state dict
    model.load_state_dict(state_dict_backbone, strict=True)
    supervised_head = MLP(layer_sizes=[embed_dim, n_classes])
    supervised_head.load_state_dict(state_dict_mlp,strict=True)
    return DINOClassifier(model,supervised_head)


def get_retCCL(model_path):
    model = retccl_res50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
    pretext_model = torch.load(model_path, map_location=torch.device("cpu"))
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    return model


def get_vim_finetuned(checkpoint=None):
    from models.vim import get_vision_mamba_model

    model = get_vision_mamba_model(checkpoint=checkpoint)
    return model




# for 224
def get_dino_finetuned_downloaded(model_path, modelname,image_size):
    model = torch.hub.load("facebookresearch/dinov2", modelname)
    # load finetuned weights

    # pos_embed has wrong shape
    if model_path is not None:
        pretrained = torch.load(model_path, map_location=torch.device("cpu"))
        # make correct state dict for loading
        new_state_dict = {}
        for key, value in pretrained["teacher"].items():
            if "dino_head" in key or "ibot_head" in key:
                pass
            else:
                new_key = key.replace("backbone.", "")
                new_state_dict[new_key] = value
        # change shape of pos_embed
        input_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        num_tokens=int(1+(image_size/14)**2)
        pos_embed = nn.Parameter(torch.zeros(1, num_tokens, input_dims[modelname.replace("_reg","")]))
        model.pos_embed = pos_embed
        # load state dict
        model.load_state_dict(new_state_dict, strict=True)
    return model





def multiply_by_255(img):
    return img * 255

def get_transforms(model_name,image_size=224,model_path=None):
    # from imagenet, leave as is
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if model_name.lower() in ["ctranspath", "resnet50", "simclr_lung", "beit_fb", "resnet50_full", "uni","conch"]:
        size = 224

    elif model_name.lower().replace("_reg","") in [
        "dinov2_vits14",
        "dinov2_vits14_classifier",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
        "dinov2_finetuned",
        "dinov2_vits14_interpolated",
        "dinov2_finetuned_downloaded",
        "remedis",
        "vim_finetuned",
        "dinobloom_s",
        "dinobloom_b",
        "dinobloom_l",
        "dinobloom_g"
    ]:
        size = image_size


    else:
        raise ValueError("Model name not found")
    
    size=(size,size)
    
    transforms_list = [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]


    preprocess_transforms = transforms.Compose(transforms_list)

    return preprocess_transforms

