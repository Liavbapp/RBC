import random
import torch

from Components.Embedding.NeuralNetwork import NeuralNetwork
from Components.RBC_ML.Optimizer import Optimizer
from Tests.RBC_ML.EmbeddingsParams import EmbeddingsParams
from Utils.CommonStr import ErrorTypes, HyperParams

DEVICE = torch.device('cuda:0')
DTYPE = torch.float



def train_model(self, nn_model, samples, p_man: EmbeddingsParams, optimizer: Optimizer):
    print(f'starting training')
    hyper_params = p_man.hyper_params
    loss_fn = get_loss_fun(hyper_params[HyperParams.error_type])
    loss = None

    random.shuffle(samples)
    features = torch.stack([embedding for embedding, label in samples])
    labels = torch.stack([label for embedding, label in samples])

    batch_size = 512

    for epoch in range(0, 10):
        for i in range(0, len(features), batch_size):
            features_batch = features[i:i + batch_size]
            labels_batch = labels[i: i + batch_size]
            y_pred = nn_model(features_batch)
            loss = loss_fn(y_pred, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, loss.item()) if epoch % 50 == 0 else 1

    return nn_model

def predict_routing(self, model, embeddings):
    num_nodes = len(embeddings)
    predicted_R = torch.full(size=(num_nodes, num_nodes, num_nodes, num_nodes), fill_value=0.0, dtype=torch.float,
                             device=DEVICE)
    model.eval()
    for s in range(0, num_nodes):
        for t in range(0, num_nodes):
            for u in range(0, num_nodes):
                for v in range(0, num_nodes):
                    features = torch.tensor([[embeddings[s], embeddings[u], embeddings[v], embeddings[t]]],
                                            device=DEVICE, dtype=DTYPE)
                    predicted_R[s, t][v, u] = model(features)

    return predicted_R


def get_loss_fun(error_type):
    if error_type == ErrorTypes.mse:
        return torch.nn.MSELoss(reduction='sum')
    else:
        return torch.nn.MSELoss(reduction='mean')
