import torch
import torch.nn as nn
# Architecture 

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels=256, bottleneck_channels=64, out_channels=256):
        super(BottleneckBlock, self).__init__()
        
        # 1. Compression : de 256 à 64 canaux (Conv 1x1)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        # 2. Traitement : reste à 64 canaux (Conv 3x3)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        # 3. Restauration : de 64 à 256 canaux (Conv 1x1 simple !)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # On garde une copie de l'entrée pour la connexion résiduelle (ResNet)
        identity = x 
        
        # Étape 1
        out = self.relu(self.bn1(self.conv1(x)))
        # Étape 2
        out = self.relu(self.bn2(self.conv2(out)))
        # Étape 3 (Restauration)
        out = self.bn3(self.conv3(out))
        
        # On ajoute l'entrée du bloc (skip connection) avant le dernier ReLU
        out += identity
        out = self.relu(out)
        
        return out

# Test rapide : une image de taille 56x56 avec 256 canaux
input_tensor = torch.randn(1, 256, 56, 56)
model = BottleneckBlock()
output_tensor = model(input_tensor)

print("Taille d'entrée :", input_tensor.shape)  # torch.Size([1, 256, 56, 56])
print("Taille de sortie :", output_tensor.shape) # torch.Size([1, 256, 56, 56])
