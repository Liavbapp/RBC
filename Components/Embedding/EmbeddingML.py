import datetime
import random
import torch
import numpy as np
from Components.RBC_REG.RBC import RBC
from Utils.Optimizer import Optimizer
from Tests.RBC_ML.EmbeddingsParams import EmbeddingsParams
from Utils.CommonStr import ErrorTypes, HyperParams, EigenvectorMethod


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


def train_model(nn_model, train_samples, validation_samples, p_man: EmbeddingsParams, optimizer: Optimizer):
    print(f'starting training')
    hyper_params = p_man.hyper_params
    loss_fn = get_loss_fun(hyper_params[HyperParams.error_type])
    batch_size, epochs = hyper_params[HyperParams.batch_size], hyper_params[HyperParams.epochs]
    train_loss, validation_loss = np.inf, np.inf

    features_validation = torch.stack([embedding for embedding, label in validation_samples])
    labels_validation = torch.stack([label for embedding, label in validation_samples])

    for epoch in range(0, epochs):
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
        if epoch % 1 == 0:
            nn_model.eval()
            with torch.no_grad():
                validation_pred = nn_model(features_validation)
                validation_loss = loss_fn(validation_pred, labels_validation)
                print(f'epoch: {epoch}, train loss: {train_loss.item()}')
                print(f'epoch: {epoch}, validation loss: {validation_loss.item()}\n')
            nn_model.train()

    return nn_model, train_loss.item(), validation_loss.item()


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
        model_pred = model(torch.tensor(embeddings, device=p_man.device, dtype=p_man.dtype).view(1, embeddings.shape[0]))
    model.train()
    return model_pred.view((p_man.num_nodes,) * 4)


def get_loss_fun(error_type):
    if error_type == ErrorTypes.mse:
        return torch.nn.MSELoss(reduction='mean')
    if error_type == ErrorTypes.L1:
        return torch.nn.L1Loss()
    if error_type == ErrorTypes.SmoothL1:
        return torch.nn.SmoothL1Loss()
