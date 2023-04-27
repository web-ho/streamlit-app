import os
import torch

from main import create_model



def load_model(model_path):
    model = create_model(num_classes=25)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    check = torch.load(model_path)
    model.load_state_dict(check['model_state_dict'])
    optimizer.load_state_dict(check['optimizer_state_dict'])
    epoch = check['epoch']
    loss = check['best_loss']

    return model


# Applying post-training quantization to reduce model size and increase inference speed
# one can also try pruning, it is complex to apply. Both have upsides and downsides.
# choose which suits your need.
model = load_model('weights\\ResNet_21.pt')

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8)
# Save the quantized model
torch.save(quantized_model, 'weights\\Quantized_ResNet_21.pt')

