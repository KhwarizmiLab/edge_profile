import torchvision.models as vision_models
import torchtext.models as text_models
import torch

alexnet = "alexnet", ['alexnet']
convnext = "convnext", ['convnext_tiny', 'convnext_small', 'convnext_base']
densenet = "densenet", ['densenet121', 'densenet161', 'densenet169', 'densenet201']
efficientnet = "efficientnet", ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                                'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l']
googlenet = "googlenet", ['googlenet']
inception = "inception", ['inception_v3']
maxvit = "maxvit", ['maxvit_t']
mnasnet = "mnasnet", ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']
mobilenet = "mobilenet", ['mobilenet_v2', "mobilenet_v3_large", "mobilenet_v3_small"]
resnet = "resnet", ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
          'resnext101_32x8d','wide_resnet50_2', 'wide_resnet101_2']
shufflenet = "shufflenet", ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']
squeezenet = "squeezenet", ['squeezenet1_0', 'squeezenet1_1']
vgg = "vgg", ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']
visiontransformer = "visiontransformer", ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']
xlmr = "xlmr", ['XLMR_BASE_ENCODER', 'XLMR_LARGE_ENCODER']
roberta = "roberta", ['ROBERTA_BASE_ENCODER', 'ROBERTA_LARGE_ENCODER']

model_families = [alexnet, convnext, densenet, efficientnet, googlenet, inception, maxvit, mnasnet, mobilenet, 
                  resnet, shufflenet, squeezenet, vgg, visiontransformer, xlmr, roberta]

name_to_family = {}
for family, family_models in model_families:
    for model in family_models:
        name_to_family[model] = family

all_models = []
for i in model_families:
    all_models.extend(i[1])

quantized_models = [x for x in list(name_to_family.keys()) if hasattr(vision_models.quantization, x)]

def getModelParams(model_arch: str):
    model_params = {
        "googlenet": {
            "kwargs": {
                "aux_logits": False
            }
        },
        "alexnet": {
            "input_size": 224,
            "lr": 0.01
        },
        "resnext50_32x4d": {
            "lr": 0.01
        },
        "resnext101_32x8d": {
            "lr": 0.01
        },
        "squeezenet1_0": {
            "lr": 0.0001,
            "optim": "adam"
        },
        "squeezenet1_1": {
            "lr": 0.0001,
            "optim": "adam"
        },
        "vgg11": {
            "lr": 0.01
        }, 
        "vgg13": {
            "lr": 0.01
        }, 
        "vgg16": {
            "lr": 0.01
        }, 
        "vgg19": {
            "lr": 0.01
        },   
        "vgg11_bn": {
            "lr": 0.01
        }, 
        "vgg13_bn": {
            "lr": 0.01
        }, 
        "vgg16_bn": {
            "lr": 0.01
        }, 
        "vgg19_bn": {
            "lr": 0.01
        },
        "mnasnet0_5": {
            "lr": 0.001,
            "input_size": 224
        }, 
        "mnasnet0_75": {
            "lr": 0.001,
            "input_size": 224
        }, 
        "mnasnet1_0": {
            "lr": 0.001,
            "input_size": 224
        }, 
        "mnasnet1_3": {
            "lr": 0.001,
            "input_size": 224
        },
    }
    return model_params.get(model_arch, {})

def get_model(model_arch:str, llm=False, pretrained=False, kwargs={}):
    """If pretrained is true, does not pass kwargs"""
    if llm:
        model_arch = model_arch.upper()
    else:
        model_arch = model_arch.lower()

    if model_arch not in name_to_family:
        raise ValueError(f"Model {model_arch} not supported")
    model_params = getModelParams(model_arch)
    if "kwargs" in model_params and not pretrained:
        kwargs.update(model_params["kwargs"])
    if "num_classes" in kwargs and pretrained:
        num_classes = kwargs["num_classes"]
        kwargs.pop("num_classes")
        model = getattr(vision_models, model_arch)(pretrained=pretrained, **kwargs)
        success = fixLastLayer(model, model_arch, num_classes)
        if not success:
            print("Cannot reset number of classes on pretrained model, will default to 1000.")
        return model, None
    
    print(f"Passing {kwargs} args to torch to construct {model_arch}")
    if llm:
        model_class = getattr(text_models, model_arch)
        return model_class.get_model(load_weights=pretrained, **kwargs), model_class.transform()
    return getattr(vision_models, model_arch)(pretrained=pretrained, **kwargs), None


def get_quantized_model(model_arch: str, kwargs={}):
    """
    Returns quantized version of the model from
    torchvision.models.quantization.
    If model is not supported, returns None.
    """
    model_arch = model_arch.lower()
    if hasattr(vision_models.quantization, model_arch):
        model_params = getModelParams(model_arch)
        if "kwargs" in model_params:
            kwargs.update(model_params["kwargs"])
        print(f"Passing {kwargs} args to torch to construct {model_arch}")
        return getattr(vision_models.quantization, model_arch)(**kwargs)
    
    print(f"Warning, model architecture {model_arch} is not supported for quanitzation, returning None from get_quantized_model().")
    return None


def fixLastLayer(model: torch.nn.Module, architecture: str, num_classes: int, finetune: bool = False) -> bool:
    # implements finetuning changing the last layer see here
    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    print(f"Setting number of classes for {architecture} to {num_classes} ...")
    if finetune:
        for param in model.parameters():
            param.requires_grad = False

    supported_models = all_models   #["mnasnet1_3"]
    if architecture not in supported_models:
        return False
    family = name_to_family[architecture]
    if architecture == "alexnet":
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
        return True
    if architecture in ['resnet18', 'resnet34']:
        model.fc = torch.nn.Linear(512, num_classes)
        return True
    if architecture in ['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d','wide_resnet50_2', 'wide_resnet101_2']:
        model.fc = torch.nn.Linear(2048, num_classes)
        return True
    if family == "vgg":
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
        return True
    if family == "squeezenet":
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
        return True
    if family == "densenet":
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
        return True
    if architecture == "googlenet":
        model.fc = torch.nn.Linear(1024, num_classes)
        return True
    if architecture == 'mobilenet_v2':
        model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
        return True
    if architecture in ["mobilenet_v3_large", "mobilenet_v3_small"]:
        in_features = model.classifier[0].out_features
        model.classifier[3] = torch.nn.Linear(in_features, num_classes)
        return True
    if family == "mnasnet":
        model.classifier[1] = torch.nn.Linear(1280, num_classes)
        return True
    if family == "shufflenet":
        model.fc = torch.nn.Linear(model._stage_out_channels[-1], num_classes)
        return True
    return False
