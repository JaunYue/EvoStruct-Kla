import torch
from utils import build_ensemble_data_list
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_x_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "train"
    / "x.pt"
)

train_edge_index_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "train"
    / "edge_index.pt"
)

train_y_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "train"
    / "y.pt"
)

train_masks_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "train"
    / "mask.pt"
)

train_esm_contacts_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "train"
    / "esm_contact.pt"
)

val_x_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "val"
    / "x.pt"
)

val_edge_index_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "val"
    / "edge_index.pt"
)

val_y_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "val"
    / "y.pt"
)

val_masks_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "val"
    / "mask.pt"
)

val_esm_contacts_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "val"
    / "esm_contact.pt"
)

test_x_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "test"
    / "x.pt"
)

test_edge_index_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "test"
    / "edge_index.pt"
)

test_y_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "test"
    / "y.pt"
)

test_masks_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "test"
    / "mask.pt"
)

test_esm_contacts_path = (
    PROJECT_ROOT
    / "Data"
    / "dataset_demo"
    / "test"
    / "esm_contact.pt"
)

train_x = torch.load(train_x_path)
train_edge_index = torch.load(train_edge_index_path)
train_y = torch.load(train_y_path)
train_masks = torch.load(train_masks_path)
train_esm_contacts = torch.load(train_esm_contacts_path)

val_x = torch.load(val_x_path)
val_edge_index = torch.load(val_edge_index_path)
val_y = torch.load(val_y_path)
val_masks = torch.load(val_masks_path)
val_esm_contacts = torch.load(val_esm_contacts_path)

test_x = torch.load(test_x_path)
test_edge_index = torch.load(test_edge_index_path)
test_y = torch.load(test_y_path)
test_masks = torch.load(test_masks_path)
test_esm_contacts = torch.load(test_esm_contacts_path)

train_list, val_list, test_list = build_ensemble_data_list(train_x, train_edge_index, train_y, train_masks, train_esm_contacts, train_masks),\
                                build_ensemble_data_list(val_x, val_edge_index, val_y, val_masks, val_esm_contacts, val_masks),\
                                build_ensemble_data_list(test_x, test_edge_index, test_y, test_masks, test_esm_contacts, test_masks)