from .backbones import resnet18, resnet50, VGG
from .seq_modules import BiLSTM
from .pred_modules import CTC, Attention


_backbone_factory = {"resnet18": resnet18, "resnet50": resnet50, "vgg": VGG}


seq_factory = {"bilstm": BiLSTM}


pred_factory = {"ctc": CTC, "attn": Attention}


def get_module(backbone_name, seq_name, pred_name):
    print(f"backbone_name: {backbone_name}")
    print(f"backbone_fac: {_backbone_factory}")
    if backbone_name not in _backbone_factory:
        raise ValueError(f"Backbone {backbone_name} not found")
    backbone = _backbone_factory[backbone_name]

    if seq_name not in seq_factory:
        seq_module = None
    else:
        seq_module = seq_factory[seq_name]

    if pred_name not in pred_factory:
        raise ValueError(f"Prediction {pred_name} not found")
    pred_module = pred_factory[pred_name]

    return backbone, seq_module, pred_module
