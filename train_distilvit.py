from transformers import ViTFeatureExtractor
from model.distilvit.modeling_distilvit import DistilViTForImageClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch
import torchmetrics

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

from data.dataset import ISICDataset

from evaluate import load as load_metric
from model.distilvit.configuration_distilvit import DistilViTConfig

import os, sys
os.chdir(sys.path[0])

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

class VITPreprocess(ImageOnlyTransform):

    def __init__(self, feature_extractor, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply, p)
        self.feature_extractor = feature_extractor

    def apply(self, img, **params):
        return self.feature_extractor(img, data_format="channels_last")['pixel_values'][0]

train_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=450),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=90, p=0.75),
        A.RandomCrop(height=400, width=400),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        VITPreprocess(feature_extractor),
        ToTensorV2()
    ]
)
valid_transform = A.Compose(
    [
        VITPreprocess(feature_extractor),
        ToTensorV2()
    ]
)

full_dataset = ISICDataset("../dataset/ISIC_2019_Training_GroundTruth.csv", 
                    "../ISIC_2019_Training_Input", 
                    transform=train_transform, 
                    val_transform=valid_transform
                )
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size

SPLIT_SEED = 42
BATCH_SIZE = 64
EPOCHS = 20

train_indices, valid_indices, _, _ = train_test_split( 
                                range(len(full_dataset)), 
                                full_dataset.labels, 
                                stratify=full_dataset.labels,
                                test_size=valid_size,
                                random_state=42)

full_dataset.set_indices(train_indices, valid_indices)

train_dataset = Subset(full_dataset, train_indices)
valid_dataset = Subset(full_dataset, valid_indices)

metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=full_dataset.classes, average="weighted", top_k=1)
metric_precision = torchmetrics.Precision(task = "multiclass", num_classes=full_dataset.classes, average="weighted", top_k=1)
metric_recall = torchmetrics.Recall(task = "multiclass", num_classes=full_dataset.classes, average="weighted", top_k=1)
metric_f1 = torchmetrics.F1Score(task = "multiclass", num_classes=full_dataset.classes, average="weighted", top_k=1)
metric_bacc = torchmetrics.Recall(task = "multiclass", num_classes=full_dataset.classes, average="macro", top_k=1)

def compute_metrics(p):

    logits, labels = p
    predictions=np.argmax(logits, axis=1)
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)

    acc = metric_acc(predictions, labels)
    prec = metric_precision(predictions, labels)
    recall = metric_recall(predictions, labels)
    f1 = metric_f1(predictions, labels)
    bacc = metric_bacc(predictions, labels)

    #acc = metric_acc.compute(predictions=predictions, references=labels)
    #prec = metric_prec.compute(predictions=predictions, references=labels, average="weighted")
    #recall = metric_recall.compute(predictions=predictions, references=labels, average="weighted")
    #f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc, "precision": prec, "recall": recall, "f1": f1, "bacc":bacc}
    
def collate_fn(batch):
    images = torch.stack( [x[0] for x in batch])
    labels = torch.tensor([x[1] for x in batch])
    return {
        "pixel_values": images, 
        "labels": labels
    }


# model = DistilViTForImageClassification._from_config(
#         DistilViTConfig(
#             id2label = {str(i): c for i, c in enumerate(full_dataset.classes_names)},
#             label2id = {c: str(i) for i, c in enumerate(full_dataset.classes_names)}
#         ))



config = DistilViTConfig(
    num_labels = len(full_dataset.classes_names),
    id2label = {str(i): c for i, c in enumerate(full_dataset.classes_names)},
    label2id = {c: str(i) for i, c in enumerate(full_dataset.classes_names)}
)
model = DistilViTForImageClassification(config)

training_args = TrainingArguments(
    output_dir = "../experiments/DistilViT_trained_3",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    num_train_epochs=EPOCHS,
    save_steps=1000,
    logging_steps=1000,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=True,
    push_to_hub=False,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_recall"
)

trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=feature_extractor
        )


metrics = trainer.evaluate(valid_dataset, ignore_keys=["t_logits","s_hidden_states","t_hidden_states"])
print(metrics)
exit()

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(valid_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
