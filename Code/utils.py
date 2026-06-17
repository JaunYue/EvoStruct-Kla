import torch
from torch_geometric.data import Data
import numpy as np
from typing import Sequence, Union, List, Tuple

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence]


def build_data(x: ArrayLike,
               edge_index: ArrayLike,
               y: Union[int, float, torch.Tensor, None] = None,
               x_dtype=torch.float32,
               edge_dtype=torch.long,
               y_dtype=torch.long) -> Data:
    """
    torch_geometric.data.Data。
    - x: (N_nodes, feat_dim) or (N_nodes,) array-like or tensor
    - edge_index: (2, E) or (E, 2) numpy/torch or list/tuple of two iterables
    - y: graph-level label (scalar) or tensor
    """
    # x -> tensor
    if isinstance(x, torch.Tensor):
        x_t = x.to(dtype=x_dtype)
    else:
        x_t = torch.tensor(np.asarray(x), dtype=x_dtype)
    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(1)

    # edge_index imput may torch.Tensor / np.ndarray / list-of-two / (E,2) list
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index
    else:
        ei = torch.tensor(np.asarray(edge_index))

    # (2, E)
    if ei.ndim == 2 and ei.shape[0] == 2:
        edge_index_t = ei.to(dtype=edge_dtype)
    elif ei.ndim == 2 and ei.shape[1] == 2:
        edge_index_t = ei.t().to(dtype=edge_dtype)
    else:
        raise ValueError(f"Unsupported edge_index shape: {tuple(ei.shape)}")

    n_nodes = x_t.size(0)
    if edge_index_t.numel() > 0:
        if int(edge_index_t.max()) >= n_nodes or int(edge_index_t.min()) < 0:
            raise ValueError(f"edge_index contains out-of-range indices (n_nodes={n_nodes})")

    data = Data(x=x_t, edge_index=edge_index_t)

    if y is not None:
        if isinstance(y, torch.Tensor):
            y_t = y.clone().detach()
            if y_t.dim() == 0:
                y_t = y_t.unsqueeze(0)
        else:
            y_t = torch.tensor([y], dtype=y_dtype)
        data.y = y_t.to(dtype=y_dtype)

    return data



def build_data_list(xs: Sequence[ArrayLike],
                    edge_indices: Sequence[ArrayLike],
                    ys: Union[Sequence[Union[int,float,torch.Tensor]], None] = None,
                    masks: Union[Sequence[torch.Tensor], None] = None,
                    x_dtype=torch.float32,
                    edge_dtype=torch.long,
                    y_dtype=torch.long):

    if not (len(xs) == len(edge_indices)):
        raise ValueError("xs and edge_indices must have same length")
    if ys is not None and len(ys) != len(xs):
        raise ValueError("ys must be same length as xs if provided")
    if masks is not None and len(masks) != len(xs):
        raise ValueError("masks must have same length as xs if provided")
    
    data_list = []
    for i, (x, ei) in enumerate(zip(xs, edge_indices)):
        y = None if ys is None else ys[i]

        # adjacent matrix [N,N]，to edge_index [2,E]
        ei = torch.as_tensor(ei)
        if ei.ndim == 2 and ei.shape[0] == ei.shape[1]:
            ei = ei.nonzero(as_tuple=False).t().contiguous()  # [2,E]

        data = build_data(x, ei, y=y, x_dtype=x_dtype, edge_dtype=edge_dtype, y_dtype=y_dtype)
        if masks is not None:
            data.node_mask = masks[i]   # input mask
        data_list.append(data)
    return data_list



class EnsembleData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        # N=35 batch may [B, ...]
        if key in ["x_seq", "esm_contact", "seq_mask"]:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


def build_ensemble_data(
    x,
    edge_index,
    esm_contact=None,
    seq_mask=None,
    y: Union[int, float, torch.Tensor, None] = None,
    x_dtype=torch.float32,
    edge_dtype=torch.long,
    y_dtype=torch.long
) -> EnsembleData:

    # x -> tensor
    if isinstance(x, torch.Tensor):
        x_t = x.to(dtype=x_dtype)
    else:
        x_t = torch.tensor(np.asarray(x), dtype=x_dtype)
    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(1)

    # edge_index -> (2, E)
    if isinstance(edge_index, torch.Tensor):
        ei = edge_index
    else:
        ei = torch.tensor(np.asarray(edge_index))

    if ei.ndim == 2 and ei.shape[0] == 2:
        edge_index_t = ei.to(dtype=edge_dtype)
    elif ei.ndim == 2 and ei.shape[1] == 2:
        edge_index_t = ei.t().to(dtype=edge_dtype)
    else:
        raise ValueError(f"Unsupported edge_index shape: {tuple(ei.shape)}")

    # checking
    n_nodes = x_t.size(0)
    if edge_index_t.numel() > 0:
        if int(edge_index_t.max()) >= n_nodes or int(edge_index_t.min()) < 0:
            raise ValueError(f"edge_index contains out-of-range indices (n_nodes={n_nodes})")

    #  [B,N,F]
    data = EnsembleData(x=x_t, edge_index=edge_index_t, x_seq=x_t)

    # attention fields
    if esm_contact is not None:
        esm_t = esm_contact if isinstance(esm_contact, torch.Tensor) else torch.tensor(np.asarray(esm_contact))
        data.esm_contact = esm_t  # [N,N] -> [B,N,N]

    if seq_mask is not None:
        mask_t = seq_mask if isinstance(seq_mask, torch.Tensor) else torch.tensor(np.asarray(seq_mask))
        data.seq_mask = mask_t    # [N] -> batch [B,N]

    # y
    if y is not None:
        if isinstance(y, torch.Tensor):
            y_t = y.clone().detach()
            if y_t.dim() == 0:
                y_t = y_t.unsqueeze(0)
        else:
            y_t = torch.tensor([y], dtype=y_dtype)
        data.y = y_t.to(dtype=y_dtype)

    return data




def build_ensemble_data_list(
    xs: Sequence,
    edge_indices: Sequence,
    ys: Union[Sequence[Union[int,float,torch.Tensor]], None] = None,
    node_masks: Union[Sequence[torch.Tensor], None] = None,
    esm_contacts: Union[Sequence[torch.Tensor], None] = None,
    seq_masks: Union[Sequence[torch.Tensor], None] = None,
    x_dtype=torch.float32,
    edge_dtype=torch.long,
    y_dtype=torch.long
):
    if not (len(xs) == len(edge_indices)):
        raise ValueError("xs and edge_indices must have same length")
    if ys is not None and len(ys) != len(xs):
        raise ValueError("ys must be same length as xs if provided")
    if node_masks is not None and len(node_masks) != len(xs):
        raise ValueError("node_masks must have same length as xs if provided")
    if esm_contacts is not None and len(esm_contacts) != len(xs):
        raise ValueError("esm_contacts must have same length as xs if provided")
    if seq_masks is not None and len(seq_masks) != len(xs):
        raise ValueError("seq_masks must have same length as xs if provided")

    data_list = []
    for i, (x, ei) in enumerate(zip(xs, edge_indices)):
        y = None if ys is None else ys[i]

        # adjacent matrix [N,N] to edge_index [2,E]
        ei = torch.as_tensor(ei)
        if ei.ndim == 2 and ei.shape[0] == ei.shape[1]:
            ei = ei.nonzero(as_tuple=False).t().contiguous()  # [2,E]

        esm = None if esm_contacts is None else esm_contacts[i]
        sm  = None if seq_masks is None else seq_masks[i]

        data = build_ensemble_data(
            x, ei, esm_contact=esm, seq_mask=sm,
            y=y, x_dtype=x_dtype, edge_dtype=edge_dtype, y_dtype=y_dtype
        )

        if node_masks is not None:
            data.node_mask = node_masks[i]  # mask remained

        data_list.append(data)

    return data_list
