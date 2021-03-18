import random
import torch

from Components.Embedding.NeuralNetwork import NeuralNetwork

DEVICE = torch.device('cuda:0')
DTYPE = torch.float


class EmbeddingML:

    def train_model(self, samples, embeddings_dimensions):
        print(f'starting training')
        loss_fn = torch.nn.MSELoss(reduction='sum')
        learning_rate = 1e-3
        model = NeuralNetwork(embeddings_dimensions)
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
        loss = None

        random.shuffle(samples)
        features = torch.stack([embedding for embedding, label in samples])
        labels = torch.stack([label for embedding, label in samples])

        batch_size = 512

        for epoch in range(0, 10):
            for i in range(0, len(features), batch_size):
                features_batch = features[i:i + batch_size]
                labels_batch = labels[i: i + batch_size]
                y_pred = model(features_batch)
                loss = loss_fn(y_pred, labels_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(epoch, loss.item()) if epoch % 50 == 0 else 1

        return model

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
