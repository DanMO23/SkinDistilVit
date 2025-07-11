from transformers import ViTFeatureExtractor, ViTForImageClassification
from model.fcvit.modeling_fcvit import FCViTForImageClassification, FCViTForImageClassificationProbs
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

import os, sys
import random
os.chdir(sys.path[0])
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Warning fix: Use ViTImageProcessor instead of ViTFeatureExtractor
# This will also require adjusting the VITPreprocess class if the new processor has a different interface.
# For now, we'll keep ViTFeatureExtractor as per your original code to address the immediate error.
# However, be aware of this future deprecation.
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
        # Warning fix: Use Affine instead of ShiftScaleRotate
        # For now, we'll keep ShiftScaleRotate as per your original code to address the immediate error.
        # However, be aware of this future deprecation.
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
EPOCHS = 60
MODEL_TYPE = "ViT"

if MODEL_TYPE == "ViT":
    ARCHITECTURE = ViTForImageClassification
elif MODEL_TYPE == "FCViT":
    ARCHITECTURE = FCViTForImageClassification
elif MODEL_TYPE == "FCViTProbs":
    ARCHITECTURE = FCViTForImageClassificationProbs



train_indices, valid_indices, _, _ = train_test_split(
    range(len(full_dataset)), # Usa todos os índices do dataset completo
    full_dataset.labels,      # Usa todos os rótulos do dataset completo
    stratify=full_dataset.labels,
    test_size=valid_size,     # Usa o `valid_size` calculado a partir do `len(full_dataset)`
    random_state=SPLIT_SEED
)

full_dataset.set_indices(train_indices, valid_indices)

train_dataset = Subset(full_dataset, train_indices)
valid_dataset = Subset(full_dataset, valid_indices)


num_classes = len(full_dataset.classes_names)
metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average="weighted", top_k=1)
metric_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="weighted", top_k=1)
metric_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="weighted", top_k=1)
metric_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="weighted", top_k=1)
metric_bacc = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro", top_k=1)

def compute_metrics(p):

    logits, labels = p
    predictions=np.argmax(logits, axis=1)
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)

    print(predictions[:10])
    print(labels[:10])
    acc = metric_acc(predictions, labels)
    prec = metric_precision(predictions, labels)
    recall = metric_recall(predictions, labels)
    f1 = metric_f1(predictions, labels)
    bacc = metric_bacc(predictions, labels)

    return {"accuracy": acc, "precision": prec, "recall": recall, "f1": f1, "bacc": bacc}

if MODEL_TYPE == "ViT":
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels = len(full_dataset.classes_names),
        id2label = {str(i): c for i, c in enumerate(full_dataset.classes_names)},
        label2id = {c: str(i) for i, c in enumerate(full_dataset.classes_names)}
    )
else:
    from transformers import ViTConfig
    config = ViTConfig.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels = len(full_dataset.classes_names),
        id2label = {str(i): c for i, c in enumerate(full_dataset.classes_names)},
        label2id = {c: str(i) for i, c in enumerate(full_dataset.classes_names)}
    )
    model = ARCHITECTURE(config)


def collate_fn(batch):
    images = torch.stack( [x[0] for x in batch])
    labels = torch.tensor([x[1] for x in batch])
    return {
        "pixel_values": images,
        "labels": labels,
    }

max_steps = int( EPOCHS*len(train_dataset)/BATCH_SIZE/1000)*1000+1

training_args = TrainingArguments(
    output_dir = f"../experiments/{MODEL_TYPE}",
    per_device_train_batch_size=BATCH_SIZE,
    # evaluate_during_training=True, # REMOVED THIS LINE
    evaluation_strategy="steps", # ADDED THIS LINE - or "epoch" depending on your preference
    eval_steps=1000, # ADDED THIS LINE - or whatever frequency you want to evaluate
    num_train_epochs=EPOCHS,
    max_steps = max_steps,
    save_steps=1000,
    logging_steps=1000,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=True,
    push_to_hub=False,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_bacc"
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

#train_results = trainer.train(ignore_keys_for_eval=["all_logits"])
trainer.evaluate(ignore_keys=["all_logits"])

trainer.save_model()
# trainer.log_metrics("train", train_results.metrics)
# trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


metrics = trainer.evaluate(valid_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)