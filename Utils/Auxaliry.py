import torch


def create_uv_matrix_combined(node_embeddings, embed_dim, num_nodes):
    uv = [torch.cartesian_prod(node_embeddings.T[i], node_embeddings.T[i]) for i in range(0, embed_dim)]
    uv = torch.cat(uv, dim=1)
    return uv


def create_uv_matrix_ordered(node_embeddings, device):
    num_nodes, embed_dim = node_embeddings.shape[0], node_embeddings.shape[1]
    uv = [torch.cartesian_prod(node_embeddings.T[i], node_embeddings.T[i]) for i in range(0, embed_dim)]
    uv = torch.cat(uv, dim=1)

    even_indices = torch.tensor(list(range(0, embed_dim * 2, 2)), device=device)
    odd_indices = torch.tensor(list(range(1, embed_dim * 2, 2)), device=device)
    uv = torch.cat([torch.index_select(uv, 1, even_indices), torch.index_select(uv, 1, odd_indices)], dim=1)

    return uv
