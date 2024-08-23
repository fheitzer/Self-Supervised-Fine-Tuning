#### TRAIJNING

def baseline_training(model_name: str, dataset_name: str, num_epochs: int):
    # Get training starttime for
    t = time.strftime("%Y%m%d-%H%M%S")
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else
                           'cpu')

    # Get model
    model = timm.create_model(model_name, num_classes=2, pretrained=False).to(device).to(torch.float32)

    # Get Dataset Eval and Train
    dh_train = data.DataHandler(os.path.join(dataset_name, 'train/'), batch_size=32, device=device)
    dh_val = data.DataHandler(os.path.join(dataset_name, 'val/'), batch_size=32, device=device)

    # Get Loss
    train_loss_fn = torch.nn.CrossEntropyLoss(
            #target_threshold=bce_target_thresh, smoothing=smoothing
        )
    #validate_loss_fn = torch.nn.CrossEntropyLoss()
    # output_fn = torch.nn.functional.softmax(dim=2)
    # with
    optimizer = AdamP(model.parameters(), lr=0.0001)
    print("Training...")
    for epoch in range(num_epochs):
        print('epoch:', epoch + 1)
        # Train Loop
        avg_loss = 0
        for i, (inputs, targets, _) in enumerate(dh_train):
            # print('batch', i)
            logits = model(inputs)#.to('cuda')
            # print("logits train", logits)
            #predictions = torch.nn.functional.softmax(logits, dim=1)
            targets = torch.nn.functional.one_hot(targets.to(torch.int64).to(device), num_classes=2).to(torch.float32)
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
            targets = torch.nn.functional.one_hot(targets.to(torch.int64).to(device), num_classes=2).to(torch.float32)
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


######## DATAHANDLER ITERATOR


    def __iter__(self):
        assert self.data_dir, "Data Directory needs to be specified!"
        for i, data in enumerate(self.dataloader, 0):
            inputs, label, path = data
            inputs, label = inputs.to(self.device), label.to(self.device)
            yield inputs, label, path
        #####################################
        # if not self.paths:
        #     self.read_dir()
        # if self.shuffle:
        #     random.shuffle(self.paths)
        #
        # for batch in self.paths:
        #     images = list()
        #     for image_path in batch:
        #         # Load PIL Image
        #         image = PIL.Image.open(image_path)
        #         # Transform pil image to torch tensor and apply transformations
        #         image = self.transform(image)
        #         # Send tensor to device (GPU)
        #         image = image.to(self.device)
        #         images.append(image)
        #     yield images

    # def batch(self, n):
    #     assert self.batch, "Already batched!"
    #     # Flatten
    #     self.paths = [item for sublist in self.paths for item in sublist]
    #     # Batch
    #     self.paths = [self.paths[i: i+n] for i in range(0, len(self.paths), n)]
    #     # Set batched to true
    #     self.batched = True
    #
    # def prefetch(self, n):
    #     pass
    #
    # def cache(self, n):
    #     pass
    #
    # def shuffle(self):

    #     self.shuffle = True
    #
    # def set_data_paths(self, data_dir):
    #     self.data_dir = data_dir
    #
    # def read_dir(self):
    #     self.paths = [[os.path.join(self.data_dir, file)] for file in os.listdir(self.data_dir) if x.endswith('.png')]
    ####################################################