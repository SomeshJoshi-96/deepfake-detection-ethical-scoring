
import torch.nn as nn
from transformers import ViTModel
import torchvision.models as models


model_name_or_path = 'google/vit-base-patch16-224'
    
class VITClassifier(nn.Module):
    def __init__(self, model_name , num_classes, freeze_layers = False):
        super(VITClassifier, self).__init__()
        self.model_name = model_name
        self.vit = ViTModel.from_pretrained(model_name)

        if freeze_layers:
            for params in self.vit.parameters():
                params.requires_grad = False
                #freezing the vit parameters as training vit is too time consuming

            # Unfreeze the last 2 layers for fine-tuning
            for layer in self.vit.encoder.layer[-4:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        self.classifier  = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
        
    def forward(self, x):
        output_vit = self.vit(x)
        output_logits = self.classifier(output_vit.last_hidden_state[:, 0])
        return output_logits
    

def create_resnet18_classifier(num_classes, freeze_pretrained=True, hidden_dim=256, dropout_prob=0.4):
    model = models.resnet18(weights='IMAGENET1K_V1')
    if freeze_pretrained:
        print("Freezing ResNet convolutional base layers.")
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_dim),       
        nn.BatchNorm1d(hidden_dim),            
        nn.ReLU(inplace=True),                 
        nn.Dropout(p=dropout_prob),            
        nn.Linear(hidden_dim, hidden_dim // 2), 
        nn.BatchNorm1d(hidden_dim // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_prob / 2),      

        nn.Linear(hidden_dim //2, num_classes)   
    )
    
    if freeze_pretrained:
        print("Ensuring new MLP head parameters are trainable.")
        for param in model.fc.parameters():
            param.requires_grad = True

    return model
    
