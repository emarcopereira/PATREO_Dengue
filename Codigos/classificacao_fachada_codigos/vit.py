from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import ViTFeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def get_timm_vit(num_classes=3):
    model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, in_chans=3, num_classes=num_classes)
    return model
