import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import cv2

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split # Corrigido: `StratifiedShuffleSplit` para `train_test_split`
# Corrigido: Não precisa importar de transformers aqui, era um bloco duplicado.
# from transformers import ViTFeatureExtractor, ViTForImageClassification
# from transformers import TrainingArguments, Trainer
# import numpy as np
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Subset, DataLoader
# import torch

import torchvision
# Corrigido: `InterpolationMode` não é usado diretamente
# from torchvision.transforms.functional import InterpolationMode
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

from data.dataset import ISICDataset

# Corrigido: O `evaluate` ou `datasets` load_metric não é usado neste script, pode remover se não for usar.
# from evaluate import load as load_metric # Ou from datasets import load_metric


import os, sys
os.chdir(sys.path[0])

# Removido bloco duplicado de feature_extractor e VITPreprocess se não for usar ViT no EffNet
# (Apenas para garantir que o script de EffNet não tenha dependências desnecessárias)
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# class VITPreprocess(ImageOnlyTransform):
#     def __init__(self, feature_extractor, always_apply: bool = True, p: float = 1.0):
#         super().__init__(always_apply, p)
#         self.feature_extractor = feature_extractor
#     def apply(self, img, **params):
#         return self.feature_extractor(img, data_format="channels_last")['pixel_values'][0]

train_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=600),
        # Corrigido: ShiftScaleRotate para Affine
        A.Affine(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=90, p=0.75
        ),
        A.RandomCrop(height=380, width=380),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2()
    ]
)

valid_transform = A.Compose(
    [
        # Corrigido: O valid_transform também precisa do mesmo processamento de imagem que o train_transform
        # para redimensionamento e recorte central, se o modelo espera um tamanho fixo.
        # Caso contrário, o modelo pode receber imagens de tamanhos variados.
        # Assumindo que o modelo espera 380x380 após o SmallestMaxSize e RandomCrop.
        A.SmallestMaxSize(max_size=380), # Reduz para um tamanho base
        A.CenterCrop(height=380, width=380), # Recorte central para validação
        ToTensorV2()
    ]
)

# Caminhos do dataset ajustados
full_dataset = ISICDataset("../dataset/ISIC_2019_Training_GroundTruth.csv",
                    "../ISIC_2019_Training_Input",
                    transform=train_transform,
                    val_transform=valid_transform # Corrigido: Adicionado val_transform
                )
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size

SPLIT_SEED = 42
BATCH_SIZE = 8 # Artigo menciona BATCH_SIZE=8 para EfficientNet-B6
EPOCHS = 20
LR = 1e-4

train_indices, valid_indices, _, _ = train_test_split(
                                range(len(full_dataset)),
                                full_dataset.labels,
                                stratify=full_dataset.labels,
                                test_size=valid_size,
                                random_state=42)

train_dataset = Subset(full_dataset, train_indices)
valid_dataset = Subset(full_dataset, valid_indices)

# AUMENTANDO num_workers E ADICIONANDO pin_memory=True
# Usando metade dos núcleos da CPU para num_workers como um bom ponto de partida.
num_cpu_workers = os.cpu_count() // 2
if num_cpu_workers == 0: # Garante que haja pelo menos 1 worker se a CPU tiver poucos núcleos
    num_cpu_workers = 1

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_cpu_workers, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=num_cpu_workers, pin_memory=True)


class ISICModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Corrigido: Usar 'weights' em vez de 'pretrained' para torchvision models
        # E especificar os pesos para evitar UserWarning
        # O artigo usou EfficientNet-B6, que corresponde a EfficientNet_B6_Weights.IMAGENET1K_V1
        # Para os pesos mais atualizados, usaria EfficientNet_B6_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b6(weights=torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1)
        # O artigo usa EfficientNet-B6, que tem uma estrutura ligeiramente diferente do exemplo comum de ResNet.
        # A última camada linear do EfficientNet é `_fc` no classifier.
        # Vamos redefinir o `_fc` para corresponder ao número de classes.
        # EfficientNet tem um `classifier` que contém um Dropout e um Linear.
        # A última camada Linear é `self.model.classifier[1]`.
        self.in_features = self.model.classifier[1].in_features # Obter in_features da última camada Linear original

        # A sua abordagem de remover a última camada e adicionar um novo classificador
        # também está correta e é comum, mas o in_features precisa ser do penúltimo bloco de features,
        # que é a saída do `self.model.features`. Para EfficientNet-B6, a saída do features é 2304.
        # Então, sua lógica original para `ISICModel` está OK, apenas confirmando o `in_features`.
        block_list = list(self.model.children())
        block_list = block_list[:-1] # Remove a última camada do EfficientNet (classifier)
        self.model = nn.Sequential(*block_list) # Agora self.model termina nas features

        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, self.in_features), # in_features é 2304 para EfficientNet-B6
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.in_features, len(full_dataset.classes_names))
        )

    def forward(self, x):
        embeddings = self.model(x)
        embeddings = embeddings.view(-1,self.in_features) # Achata o tensor para passar pelo classificador linear
        pred = self.classifier(embeddings)
        return pred

class ISICPL(pl.LightningModule):

    def __init__(self, isic_model):
        super(ISICPL, self).__init__()
        self.model = isic_model
        num_classes_ = len(full_dataset.classes_names) # Definir aqui para usar nas métricas
        self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes_, average="weighted", top_k=1)
        self.metric_precision = torchmetrics.Precision(task = "multiclass", num_classes=num_classes_, average="weighted", top_k=1)
        self.metric_recall = torchmetrics.Recall(task = "multiclass", num_classes=num_classes_, average="weighted", top_k=1)
        self.metric_f1 = torchmetrics.F1Score(task = "multiclass", num_classes=num_classes_, average="weighted", top_k=1)
        self.metric_bacc = torchmetrics.Recall(task = "multiclass", num_classes=num_classes_, average="macro", top_k=1) # BMA
        self.criterion = nn.CrossEntropyLoss()
        self.lr = LR

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam( self.model.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             epochs=EPOCHS,
                                                             steps_per_epoch=len(train_dataloader),
                                                             max_lr=LR)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_index):
        # Para PyTorch Lightning, o batch do DataLoader já pode ser uma tupla (inputs, labels)
        # Se seu ISICDataset retorna (imagem, label), as linhas abaixo estão corretas.
        images = batch[0].float()
        labels = batch[1].long()
        output = self.model(images)
        loss = self.criterion(output, labels)
        score = self.metric(output.argmax(1), labels)
        precision = self.metric_precision(output.argmax(1), labels)
        recall = self.metric_recall(output.argmax(1), labels)
        f1 = self.metric_f1(output.argmax(1), labels)
        bacc = self.metric_bacc(output.argmax(1), labels) # Para BMA
        logs = {"train_loss": loss, "train_acc" : score, "train_prec": precision, "train_rec": recall, "train_f1": f1, "train_bacc": bacc}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar = True, logger= True
        )
        return loss

    def validation_step(self, batch, batch_index):
        images = batch[0].float()
        labels = batch[1].long()
        output = self.model(images)
        loss = self.criterion(output, labels)
        score = self.metric(output.argmax(1), labels)
        precision = self.metric_precision(output.argmax(1), labels)
        recall = self.metric_recall(output.argmax(1), labels)
        f1 = self.metric_f1(output.argmax(1), labels)
        bacc = self.metric_bacc(output.argmax(1), labels) # Para BMA
        # Corrigido: logs para incluir bacc para validação
        logs = {"valid_loss": loss, "valid_acc" : score, "valid_prec": precision, "valid_rec": recall, "valid_f1": f1, "valid_bacc": bacc}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar = True, logger= True
        )
        return loss

if __name__ == "__main__":
    isic_model = ISICModel()
    pl_model = ISICPL(isic_model)

    logger = pl.loggers.CSVLogger(save_dir="../logs/", name="b6_baseline")
    checkpoint_callback = ModelCheckpoint(monitor="valid_loss", save_top_k = 1, save_last = True, save_weights_only=True, filename='{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}', verbose=True, mode='min')

    # Configurado para CPU conforme sua GPU
    trainer = pl.Trainer(max_epochs=EPOCHS, logger=logger, accelerator="cpu", callbacks=[checkpoint_callback])

    # CORREÇÃO CRUCIAL: Descomentar para iniciar o treinamento
    print("Iniciando o treinamento do EfficientNet-B6 na CPU...")
    trainer.fit(pl_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    print("Treinamento concluído.")

    # Bloco de plotagem, executado APÓS o treinamento
    checkpoint = checkpoint_callback.best_model_path
    if checkpoint:
        print(f"Melhor checkpoint salvo em: {checkpoint}")
    else:
        print("Nenhum checkpoint foi salvo. Verifique o monitoramento ou se o treinamento concluiu com sucesso.")


    import matplotlib.pyplot as plt
    try:
        log_dir = trainer.logger.log_dir
        metrics = pd.read_csv(f'{log_dir}/metrics.csv')

        # Assegura que as colunas existem e lida com NaNs para plotagem
        train_acc = metrics['train_acc'].dropna().reset_index(drop=True)
        valid_acc = metrics['valid_acc'].dropna().reset_index(drop=True)

        fig = plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(train_acc, color="r", marker="o", label='train/acc')
        plt.plot(valid_acc, color="b", marker="x", label='valid/acc')
        plt.ylabel('Accuracy', fontsize=24)
        plt.xlabel('Epoch', fontsize=24)
        plt.legend(loc='lower right', fontsize=18)
        plt.savefig(f'{log_dir}/acc.png')

        train_loss = metrics['train_loss'].dropna().reset_index(drop=True)
        valid_loss = metrics['valid_loss'].dropna().reset_index(drop=True)

        fig = plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(train_loss, color="r", marker="o", label='train/loss')
        plt.plot(valid_loss, color="b", marker="x", label='valid/loss')
        plt.ylabel('Loss', fontsize=24)
        plt.xlabel('Epoch', fontsize=24)
        plt.legend(loc='upper right', fontsize=18)
        plt.savefig(f'{log_dir}/loss.png')

        train_prec = metrics['train_prec'].dropna().reset_index(drop=True)
        valid_prec = metrics['valid_prec'].dropna().reset_index(drop=True)

        fig = plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(train_prec, color="r", marker="o", label='train/precision')
        plt.plot(valid_prec, color="b", marker="x", label='valid/precision')
        plt.ylabel('Precision', fontsize=24)
        plt.xlabel('Epoch', fontsize=24)
        plt.legend(loc='lower right', fontsize=18)
        plt.savefig(f'{log_dir}/prec.png')

        train_rec = metrics['train_rec'].dropna().reset_index(drop=True)
        valid_rec = metrics['valid_rec'].dropna().reset_index(drop=True)

        fig = plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(train_rec, color="r", marker="o", label='train/recall')
        plt.plot(valid_rec, color="b", marker="x", label='valid/recall')
        plt.ylabel('Recall', fontsize=24)
        plt.xlabel('Epoch', fontsize=24)
        plt.legend(loc='lower right', fontsize=18)
        plt.savefig(f'{log_dir}/rec.png')

        train_f1 = metrics['train_f1'].dropna().reset_index(drop=True)
        valid_f1 = metrics['valid_f1'].dropna().reset_index(drop=True)

        fig = plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(train_f1, color="r", marker="o", label='train/f1')
        plt.plot(valid_f1, color="b", marker="x", label='valid/f1')
        plt.ylabel('F1', fontsize=24)
        plt.xlabel('Epoch', fontsize=24)
        plt.legend(loc='lower right', fontsize=18)
        plt.savefig(f'{log_dir}/f1.png')

        train_bacc = metrics['train_bacc'].dropna().reset_index(drop=True)
        valid_bacc = metrics['valid_bacc'].dropna().reset_index(drop=True) # CORREÇÃO: Pegar de 'valid_bacc'

        fig = plt.figure(figsize=(7, 6))
        plt.grid(True)
        plt.plot(train_bacc, color="r", marker="o", label='train/bacc')
        plt.plot(valid_bacc, color="b", marker="x", label='valid/bacc')
        plt.ylabel('Bacc', fontsize=24)
        plt.xlabel('Epoch', fontsize=24)
        plt.legend(loc='lower right', fontsize=18)
        plt.savefig(f'{log_dir}/bacc.png')

    except FileNotFoundError:
        print(f"Erro: O arquivo de métricas não foi encontrado em {log_dir}/metrics.csv. Certifique-se de que o treinamento foi concluído e os logs foram salvos.")
    except KeyError as e:
        print(f"Erro: Coluna de métrica não encontrada no arquivo metrics.csv: {e}. Verifique se os nomes das colunas estão corretos e se as métricas foram logadas.")