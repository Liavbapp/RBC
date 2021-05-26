import datetime
import random
from itertools import product
from Utils.Auxaliry import create_uv_matrix_ordered
import torch
import numpy as np
from RBC.RBC import RBC
from Utils.Optimizer import Optimizer
from LearningRouting.Test.EmbeddingsParams import EmbeddingsParams
from Utils.CommonStr import ErrorTypes, HyperParams, EigenvectorMethod
import torch.utils.data
import networkx as nx


def train_model_optimize_st_routing(nn_model, samples_train, samples_val, p_man: EmbeddingsParams, optimizer):
    print(f'starting training')

    hyper_params = p_man.hyper_params
    embed_dim = p_man.embedding_dimensions
    loss_fn = get_loss_fun(hyper_params[HyperParams.error_type])
    batch_size, epochs = hyper_params[HyperParams.batch_size], hyper_params[HyperParams.epochs]
    train_loss, validation_loss = np.inf, np.inf

    train_loader = torch.utils.data.DataLoader(samples_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(samples_val, batch_size=batch_size, shuffle=False)

    n_batches = len(train_loader)
    n_inner_batches = p_man.num_nodes ** 2
    inner_batch_size = p_man.num_nodes ** 2

    for epoch in range(0, epochs):
        train_running_loss = 0.0
        for i, inputs in enumerate(train_loader):
            for j in range(0, n_inner_batches):
                start_idx, end_idx = j * inner_batch_size, (j+1) * inner_batch_size
                s_u_v_t_embedding = inputs[:, start_idx: end_idx, :embed_dim * 4]
                Expected_Rst = inputs[:, start_idx: end_idx, embed_dim * 4:]
                Actual_Rst = nn_model(s_u_v_t_embedding)
                train_loss = loss_fn(Expected_Rst, Actual_Rst)
                train_running_loss += train_loss.item()
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        if epoch % 5 == 0:
            nn_model.eval()
            with torch.no_grad():
                val_running_loss = 0.0
                n_batches_val = len(val_loader)
                for i, inputs in enumerate(val_loader):
                    s_u_v_t_embedding = inputs[:, :, :embed_dim * 4]
                    Expected_Rst = inputs[:, :, embed_dim * 4:]
                    Actual_Rst = nn_model(s_u_v_t_embedding)
                    val_loss = loss_fn(Expected_Rst, Actual_Rst)
                    val_running_loss += val_loss.item()
            nn_model.train()
            print(f'\n[{epoch}] validation loss: {val_running_loss / n_batches_val}\n')

        print(f'[{epoch}] train loss: {train_running_loss / n_batches}')

    return nn_model, train_loss.item(), validation_loss

    # nodes_embed_train = samples_train[0]
    # mult_const, add_const, Rs_train = samples_train[1], samples_train[2], samples_train[3]

    # for epoch in range(0, epochs):
    #     running_loss = 0
    #     nbatches = 0
    #     start = datetime.datetime.now()
    #     if batch_size == 1:
    #         nbatches = 1
    #         predicted_routing = nn_model(nodes_embed_train, mult_const, add_const)
    #         train_loss = loss_fn(predicted_routing, Rs_train)
    #         running_loss += train_loss.item()
    #         optimizer.zero_grad()
    #         train_loss.backward()
    #         optimizer.step()
    #     else:
    #         for i in range(0, len(samples_train), batch_size):
    #             nbatches += 1
    #             nodes_embeddings_batch = nodes_embed_train[i: i + batch_size]
    #             const_mult_batch = mult_const[i: i + batch_size]
    #             const_add_batch = add_const[i: i + batch_size]
    #             Rs_batch = Rs_train[i: i + batch_size]
    #             predicted_routing = nn_model(nodes_embeddings_batch, const_mult_batch, const_add_batch)
    #             train_loss = loss_fn(predicted_routing, Rs_batch)
    #             running_loss += train_loss.item()
    #             optimizer.zero_grad()
    #             train_loss.backward()
    #             optimizer.step()
    #
    #     print(f'[{epoch}] {running_loss / nbatches}, time: {datetime.datetime.now() - start}')

    # return nn_model, train_loss.item(), validation_loss


def train_model_optimize_centrality(nn_model, samples_train, samples_val, p_man, optimizer):
    print(f'starting training')
    hyper_params = p_man.hyper_params
    loss_fn = get_loss_fun(hyper_params[HyperParams.error_type])
    batch_size, epochs = hyper_params[HyperParams.batch_size], hyper_params[HyperParams.epochs]
    train_loss, validation_loss = np.inf, np.inf

    nodes_embed_train, graphs_embed_train, Ts_train = samples_train[0], samples_train[1], samples_train[2]
    mult_const, add_const, Rbcs_train = samples_train[3], samples_train[4], samples_train[5]

    for epoch in range(0, epochs):
        running_loss = 0
        nbatches = 0
        start = datetime.datetime.now()
        for i in range(0, len(samples_train), batch_size):
            nbatches += 1
            nodes_embeddings_batch = nodes_embed_train[i: i + batch_size]
            graphs_embeddings_batch = graphs_embed_train[i: i + batch_size]
            Ts_batch = Ts_train[i: i + batch_size]
            const_mult_batch = mult_const[i: i + batch_size]
            const_add_batch = add_const[i: i + batch_size]
            Rbcs_batch = Rbcs_train[i: i + batch_size]
            predicted_rbc = nn_model(nodes_embeddings_batch, graphs_embeddings_batch, Ts_batch, const_mult_batch,
                                     const_add_batch)
            train_loss = loss_fn(predicted_rbc, Rbcs_batch)
            running_loss += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        print(f'[{epoch}] {running_loss / nbatches}, time: {datetime.datetime.now() - start}')
    return nn_model, train_loss.item(), validation_loss


def train_model_embed_to_rbc(nn_model, train_samples, validation_samples, p_man: EmbeddingsParams,
                             optimizer: Optimizer):
    print(f'starting training')
    hyper_params = p_man.hyper_params
    loss_fn = get_loss_fun(hyper_params[HyperParams.error_type])
    batch_size, epochs = hyper_params[HyperParams.batch_size], hyper_params[HyperParams.epochs]
    train_loss, validation_loss = np.inf, np.inf

    features_validation = torch.stack([embedding for embedding, _, _, _, _ in validation_samples])
    graphs_validation = [graph for _, graph, _, _, _ in validation_samples]
    Rs_validation = [R for _, _, R, _, _, in validation_samples]
    Ts_validation = [T for _, _, _, T, _, in validation_samples]
    labels_validation = torch.stack([label for _, _, _, _, label in validation_samples])
    rbc = RBC(eigenvector_method=EigenvectorMethod.power_iteration, pi_max_error=0.00001,
              device=torch.device('cuda:0'),
              dtype=torch.float)
    for epoch in range(0, epochs):
        # random.shuffle(train_samples)
        features_train = torch.stack([embedding for embedding, _, _, _, _ in train_samples])
        graphs_train = [graph for _, graph, _, _, _ in train_samples]
        Rs_train = [R for _, _, R, _, _, in train_samples]
        Ts_train = [T for _, _, _, T, _, in train_samples]
        labels_train = torch.stack([label for _, _, _, _, label in train_samples])
        # print(epoch)

        for i in range(0, len(features_train), batch_size):
            features_batch = features_train[i:i + batch_size]
            labels_batch = labels_train[i: i + batch_size]
            graphs_batches = graphs_train[i: i + batch_size]
            Rs_batches = Rs_train[i: i + batch_size]
            Ts_batches = Ts_train[i: i + batch_size]
            if len(features_batch) > 1:
                y_pred = nn_model(features_batch)
                train_loss = loss_fn(rbc.compute_rbcs(graphs_batches, list(y_pred), Ts_batches), labels_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
        if epoch % 1 == 0:
            print(train_loss.item())
            nn_model.eval()
            with torch.no_grad():
                validation_pred = nn_model(features_validation)
                rbc_validation = rbc.compute_rbcs(graphs_validation, list(validation_pred), Ts_validation)
                validation_loss = loss_fn(rbc_validation, labels_validation)
                print(f'epoch: {epoch}, train loss: {train_loss.item()}')
                print(f'epoch: {epoch}, validation loss: {validation_loss.item()}\n')
                print(labels_validation)
                print(rbc_validation)
            nn_model.train()

    return nn_model, train_loss.item(), validation_loss.item()


def train_model_embed_to_routing(nn_model, train_samples, validation_samples, p_man: EmbeddingsParams,
                                 optimizer: Optimizer):
    print(f'starting training')
    hyper_params = p_man.hyper_params
    loss_fn = get_loss_fun(hyper_params[HyperParams.error_type])
    batch_size = hyper_params[HyperParams.batch_size]
    epochs = hyper_params[HyperParams.epochs]
    train_loss, validation_loss = np.inf, np.inf

    features_validation = torch.stack([embedding for embedding, label in validation_samples])
    labels_validation = torch.stack([label for embedding, label in validation_samples])

    for epoch in range(0, epochs):
        # print(epoch)
        random.shuffle(train_samples)
        features_train = torch.stack([embedding for embedding, label in train_samples])
        labels_train = torch.stack([label for embedding, label in train_samples])

        for i in range(0, len(features_train), batch_size):
            features_batch = features_train[i:i + batch_size]
            labels_batch = labels_train[i: i + batch_size]
            y_pred = nn_model(features_batch)

            train_loss = loss_fn(y_pred, labels_batch)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            nn_model.eval()
            with torch.no_grad():
                validation_pred = nn_model(features_validation)
                validation_loss = loss_fn(validation_pred, labels_validation)
                print(f'epoch: {epoch}, train loss: {train_loss.item()}')
                print(f'epoch: {epoch}, validation loss: {validation_loss.item()}\n')
            nn_model.train()

    return nn_model, train_loss.item(), validation_loss.item()


def train_model_st_routing(nn_model, train_samples, validation_samples, p_man: EmbeddingsParams, optimizer: Optimizer):
    print(f'starting training')
    hyper_params = p_man.hyper_params
    loss_fn = get_loss_fun(hyper_params[HyperParams.error_type])
    batch_size, epochs = hyper_params[HyperParams.batch_size], hyper_params[HyperParams.epochs]
    nodes_pow2 = p_man.num_nodes ** 2
    train_loss, validation_loss = np.inf, np.inf

    features_validation = torch.stack([embedding for embedding, label in validation_samples])
    labels_validation = torch.stack([label.view(nodes_pow2) for embedding, label in validation_samples])

    for epoch in range(0, epochs):
        random.shuffle(train_samples)
        features_train = torch.stack([embedding for embedding, label in train_samples])
        labels_train = torch.stack([label.view(nodes_pow2) for embedding, label in train_samples])
        # print(epoch)
        for i in range(0, len(features_train), batch_size):
            features_batch = features_train[i:i + batch_size]
            labels_batch = labels_train[i: i + batch_size]
            y_pred = nn_model(features_batch)
            train_loss = loss_fn(y_pred, labels_batch)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            nn_model.eval()
            with torch.no_grad():
                validation_pred = nn_model(features_validation)
                validation_loss = loss_fn(validation_pred, labels_validation)
                print(f'epoch: {epoch}, train loss: {train_loss.item()}')
                print(f'epoch: {epoch}, validation loss: {validation_loss.item()}\n')
            nn_model.train()

    return nn_model, train_loss.item(), validation_loss.item()


def train_model_4_embeddings(nn_model, train_samples, validation_samples, p_man: EmbeddingsParams,
                             optimizer: Optimizer):
    print(f'starting training')
    hyper_params = p_man.hyper_params
    loss_fn = get_loss_fun(hyper_params[HyperParams.error_type])
    batch_size, epochs = hyper_params[HyperParams.batch_size], hyper_params[HyperParams.epochs]

    train_loader = torch.utils.data.DataLoader(train_samples, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_samples, batch_size=batch_size, shuffle=True)

    for epoch in range(0, epochs):
        train_running_loss = 0.0
        for i, inputs in enumerate(train_loader):
            train_loss = compute_loss(inputs, nn_model, loss_fn)
            train_running_loss += train_loss.item() * (i + 1) ** -1
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            nn_model.eval()
            with torch.no_grad():
                val_running_loss = 0.0
                for i, inputs in enumerate(validation_loader):
                    val_loss = compute_loss(inputs, nn_model, loss_fn)
                    val_running_loss += val_loss.item() * (i + 1) ** -1
            nn_model.train()

            print(f'epoch: {epoch}, train loss: %.6f' % train_running_loss)
            print(f'epoch: {epoch}, validation loss %.6f \n' % val_running_loss)

    return nn_model, train_running_loss, val_running_loss


def compute_loss(inputs, model, loss_fn):
    features_batch, labels_batch = inputs[:, :-1], inputs[:, -1]
    y_pred = model(features_batch)
    train_loss = loss_fn(y_pred, labels_batch)

    return train_loss


def predict_centrality_direct(model, nodes_embed, T, mult_const, add_const):
    model.eval()
    with torch.no_grad():
        predicted_rbc = model(nodes_embed, None, T, mult_const, add_const)
    model.train()
    return predicted_rbc


def predict_routing(model, embeddings, p_man: EmbeddingsParams):
    model.eval()  # set the model for testing
    with torch.no_grad():
        num_nodes = len(embeddings)
        embd_torch = torch.tensor(embeddings, device=p_man.device, dtype=p_man.dtype)
        predicted_R = torch.full(size=(num_nodes,) * 4, fill_value=0.0, dtype=p_man.dtype, device=p_man.device)
        start_time = datetime.datetime.now()
        for s in range(0, num_nodes):
            for t in range(0, num_nodes):
                for u in range(0, num_nodes):
                    for v in range(0, num_nodes):
                        features = torch.stack([embd_torch[s], embd_torch[u], embd_torch[v], embd_torch[t]]).unsqueeze(
                            0)
                        predicted_R[s, t][v, u] = model(features)
        # print(f'learn routing test takes {datetime.datetime.now() - start_time} time')
    model.train()
    return predicted_R


def predict_s_t_routing(model, embeddings, p_man: EmbeddingsParams):
    model.eval()
    with torch.no_grad():
        num_nodes = p_man.num_nodes
        predicted_R = torch.full(size=(num_nodes,) * 4, fill_value=0.0, dtype=p_man.dtype, device=p_man.device)

        for s in range(0, num_nodes):
            for t in range(0, num_nodes):
                features = torch.stack(
                    [torch.tensor([embeddings[s], embeddings[t]], device=p_man.device, dtype=p_man.dtype)])
                predicted_R[s, t] = model(features).view(num_nodes, num_nodes)
    model.train()
    return predicted_R


def predict_routing_all_nodes_embed(model, embeddings, p_man: EmbeddingsParams):
    model.eval()
    with torch.no_grad():
        model_pred = model(torch.tensor(embeddings, device=p_man.device, dtype=p_man.dtype).view(1, embeddings.shape[0],
                                                                                                 embeddings.shape[1]))
    model.trian()
    return model_pred.view((embeddings.shape[0],) * 4)


def predict_graph_embedding(model, embeddings, p_man: EmbeddingsParams):
    model.eval()
    with torch.no_grad():
        model_pred = model(
            torch.tensor(embeddings, device=p_man.device, dtype=p_man.dtype).view(1, embeddings.shape[0]))
    model.train()
    return model_pred.view((p_man.num_nodes,) * 4)


# def predict_routing_policy_optimize_st_routing(model, embeddings, p_man: EmbeddingsParams):
#     model.eval()
#     with torch.no_grad():
#         num_nodes = p_man.num_nodes
#         R_lst = []
#         for s in range(num_nodes):
#             for t in range(num_nodes):
#                 R_lst.append(model(torch.tensor(embeddings[s], device=p_man.device, dtype=p_man.dtype).unsqueeze(dim=0),
#                                    torch.tensor(embeddings[t], device=p_man.device, dtype=p_man.dtype).unsqueeze(
#                                        dim=0)))
#         predicted_R = torch.stack(R_lst).view(p_man.num_nodes, p_man.num_nodes, p_man.num_nodes, p_man.num_nodes)
#         return predicted_R

def predict_routing_policy_optimize_st_routing(model, embeddings, p_man: EmbeddingsParams):
    model.eval()
    with torch.no_grad():
        num_nodes, embd_dim = embeddings.shape[0], embeddings.shape[1]
        embeddings = torch.from_numpy(embeddings).to(device=p_man.device)
        uv_tensor = create_uv_matrix_ordered(embeddings, p_man.device)
        st_tuples = list(product(range(num_nodes), range(num_nodes)))
        s_lst, t_lst = zip(*st_tuples)

        input = list(map(lambda s, t: torch.cat([embeddings[s].repeat(repeats=(num_nodes ** 2, 1)),
                                                 uv_tensor,
                                                 embeddings[t].repeat(repeats=(num_nodes ** 2, 1))],
                                                dim=1), s_lst, t_lst))
        input = torch.cat(input, dim=0).unsqueeze(dim=0).to(dtype=p_man.dtype)
        pred = model(input).view(num_nodes, num_nodes, num_nodes, num_nodes)

        return pred




def get_loss_fun(error_type):
    if error_type == ErrorTypes.mse:
        return torch.nn.MSELoss(reduction='mean')
    if error_type == ErrorTypes.L1:
        return torch.nn.L1Loss()
    if error_type == ErrorTypes.SmoothL1:
        return torch.nn.SmoothL1Loss()
