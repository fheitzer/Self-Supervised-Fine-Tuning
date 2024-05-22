erstmal imports (ist natürlich nicht alles, aber damit du unten siehst wo genau meine Funktionen her kommen:
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import Dataset, DataLoader


normalerweise schreib ich mir nen "helper", um Default-Werte der timm library zu überschrieben (zB num-classes/nc):

def create_model(model = "resnet18", pretrained=False, global_pool="catavgmax",nc=2):
    model = timm.create_model(model, num_classes=nc, pretrained=pretrained, global_pool=global_pool)
    return model


Dann schreib ich mir ne eigene NN klasse, die in der init auf den helper zugreift:

class LitResnet(LightningModule):
    def __init__(self, lr=0.05, model = "resnet18", pretrained=False, global_pool="catavgmax",nc=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model,pretrained,global_pool,nc)
        self.criterion = nn.CrossEntropyLoss()
        self.nc = nc

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
...


Und schlussendlich schreib ich mir ne "train_model"-funktion, die alle Parameter etc. bekommt, aber die ist primär deshalb so komplex, weil da halt auch alle pfade für daten ergebnisse loggs, etc. definiert werden, hier mal ne minimal-version:

def train_model(...):
    ds_tr = CustomImageDataset(...)
    my_sampler_tr = ImbalancedDatasetSampler(...)  # da geht auch ein anderer sampler, ich nutz meistens irgendeinen selbst geschriebenen
    model = LitResnet(...)
    trainer = Trainer(...)
    trainer.fit(model, dls_tr)
    torch.save(model, os.path.join(out_path,name)+".pt")  # out_path und name sind natürlich parmeter von train_model()
    return model


Die Funktion kann ich dann einfach callen und bekomm mein fertiges model und als checkpoint gespeichert ist es dann auch direkt.
