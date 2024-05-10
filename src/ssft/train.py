import torch
import timm
from lightning.pytorch import Trainer
import data

model = timm.create_model('resnet18')

# Get Dataset Eval and Train
dh = data.DataHandler('BCN')

# Get optimizer
optimizer = timm.optim.create_optimizer_v2(
        model, opt="lookahead_AdamW", lr=lr, weight_decay=0.01
    )

# Get Loss
train_loss_fn = timm.loss.BinaryCrossEntropy(
        target_threshold=bce_target_thresh, smoothing=smoothing
    )
validate_loss_fn = torch.nn.CrossEntropyLoss()

# Create Trainer
trainer = Trainer(devices=1, accelerator="mps")

# Train
trainer.fit(model, dh)
