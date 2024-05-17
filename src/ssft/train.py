import torch
import timm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import data

import os
import time
from timm.optim import AdamP
from timm.loss import BinaryCrossEntropy


def baseline_training(model_name: str, dataset_name: str, num_epochs: int):
    # Get training starttime for
    t = time.strftime("%Y%m%d-%H%M%S")
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else
                           'cpu')

    # Get model
    model = timm.create_model(model_name, num_classes=2, pretrained=False).to(device)

    # Get Dataset Eval and Train
    dh_train = data.DataHandler(os.path.join(dataset_name, 'train/'), batch_size=16)
    dh_val = data.DataHandler(os.path.join(dataset_name, 'val/'), batch_size=16)

    # Get Loss
    train_loss_fn = BinaryCrossEntropy(
            #target_threshold=bce_target_thresh, smoothing=smoothing
        )
    #validate_loss_fn = torch.nn.CrossEntropyLoss()

    # with
    optimizer = AdamP(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        print('epoch:', epoch + 1)
        # Train Loop
        avg_loss = 0
        for i, (inputs, targets, _) in enumerate(dh_train):
            print('batch', i)
            outputs = model(inputs)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = (avg_loss + loss) / (i + 1)
            break

        print('Mean train_loss', avg_loss.item())
        # Evaluation Loop
        avg_val_loss = 0
        avg_val_acc = 0
        print("Evaluating...")
        for i, (inputs, targets, _) in enumerate(dh_val):
            outputs = model(inputs)
            val_loss = train_loss_fn(outputs, targets)
            avg_val_loss = (avg_val_loss + val_loss) / (i + 1)
            mask = torch.eq(torch.nn.functional.one_hot(outputs.to(torch.int64), 1), targets)
            acc = torch.sum(mask) / len(mask)
            avg_val_acc = (avg_val_acc + acc) / (i + 1)

        print('val_loss', avg_val_loss.item())
        print('val_acc', avg_val_acc.item())

    torch.save(model.state_dict(), f=f'custom_weights{t}.pt')
    exit()

    timm.create_model(model_name=model_name,
                      checkpoint_path=custom_weights.pt,
                      num_classes=2,
                      #in_chans=2,
                      )

    # Create callbacks
    callbacks = [ModelCheckpoint(os.path.join('models', model_name, t),
                                 monitor='val_loss',
                                 filename='{epoch}-{val_loss:.2f}',
                                 save_weights_only=True,
                                 every_n_epochs=1),
                 EarlyStopping(monitor='val_loss', mode='min')]

    # Create Trainer
    trainer = Trainer(devices=1, accelerator="mps", callbacks=callbacks)

    # Train
    trainer.fit(model, dh_train, dh_val)


if __name__ == "__main__":
    models = ['resnet18']
    for model_name in models:
        baseline_training(model_name, 'isic1920_fil_split', num_epochs=1)
