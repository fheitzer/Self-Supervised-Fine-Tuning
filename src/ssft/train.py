import torch
import timm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import data
from networks import LitResnet, Ensemble

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
    model = timm.create_model(model_name, num_classes=2, pretrained=False).to('cuda').to(torch.float32)

    # Get Dataset Eval and Train
    dh_train = data.DataHandler(os.path.join(dataset_name, 'train/'), batch_size=32, device='cuda')
    dh_val = data.DataHandler(os.path.join(dataset_name, 'val/'), batch_size=32, device='cuda')

    # Get Loss
    train_loss_fn = torch.nn.CrossEntropyLoss(
            #target_threshold=bce_target_thresh, smoothing=smoothing
        )
    #validate_loss_fn = torch.nn.CrossEntropyLoss()
    # output_fn = torch.nn.functional.softmax(dim=2)
    # with
    optimizer = AdamP(model.parameters(), lr=0.01)
    print("Training...")
    for epoch in range(num_epochs):
        print('epoch:', epoch + 1)
        # Train Loop
        avg_loss = 0
        for i, (inputs, targets, _) in enumerate(dh_train):
            # print('batch', i)
            logits = model(inputs)#.to('cuda')
            # print("logits train", logits)
            predictions = torch.nn.functional.softmax(logits, dim=1)
            targets = torch.nn.functional.one_hot(targets.to(torch.int64).to("cuda"), num_classes=2).to(torch.float32)
            # print(predictions)
            # print(targets)
            loss = train_loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = (avg_loss + loss) / (i + 1)

        print('Mean train_loss', avg_loss.item())
        # Evaluation Loop
        avg_val_loss = 0
        avg_val_acc = 0
        print("Evaluating...")
        for i, (inputs, targets, _) in enumerate(dh_val):
            # Get Preds
            logits = model(inputs)#.to('cuda')
            # print("logits", logits)
            predictions = torch.nn.functional.softmax(logits, dim=1)
            # print("outputs", predictions)
            targets = torch.nn.functional.one_hot(targets.to(torch.int64).to("cuda"), num_classes=2).to(torch.float32)
            # print(predictions.size())
            # print(targets.size())
            val_loss = train_loss_fn(predictions, targets)
            avg_val_loss = (avg_val_loss + val_loss) / (i + 1)
            predictions = torch.round(predictions)
            #print("outputs", predictions)
            #print("targets", targets)
            # TODO: Make this work
            mask = torch.all(torch.eq(predictions, targets),  dim=1)
            #print("mask", mask)
            acc = torch.sum(mask) / len(mask)
            #print("acc", acc)
            avg_val_acc = (avg_val_acc + acc) / (i + 1)

        print('val_loss', avg_val_loss.item())
        print('val_acc', avg_val_acc.item())

    torch.save(model.state_dict(), f=f'custom_weights{t}.pt')
    exit()

    ######################################################################

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


def train_model(out_path, model_name: str='resnet18', dataset_name: str='isic1920_fil_split'):
    # Get Dataset Eval and Train
    dh_train = data.DataHandler(os.path.join(dataset_name, 'train/'), batch_size=32, device='cuda')
    dh_val = data.DataHandler(os.path.join(dataset_name, 'val/'), batch_size=32, device='cuda')

    #my_sampler_tr = ImbalancedDatasetSampler(...)  # da geht auch ein anderer sampler, ich nutz meistens irgendeinen selbst geschriebenen
    model = LitResnet(model_name, )
    callbacks = [ModelCheckpoint(os.path.join('models', model_name, t),
                                 monitor='val_loss',
                                 filename='{epoch}-{val_loss:.2f}',
                                 save_weights_only=True,
                                 every_n_epochs=1),
                 EarlyStopping(monitor='val_loss', mode='min')]

    trainer = Trainer(devices=1, accelerator="cuda", callbacks=callbacks)
    trainer.fit(model, dh_train, dh_val)

    torch.save(model, os.path.join(out_path,name)+".pt")  # out_path und name sind natürlich parmeter von train_model()
    return model


if __name__ == "__main__":
    models = ['resnet18']
    for model_name in models:
        baseline_training(model_name, 'isic1920_fil_split', num_epochs=1)
