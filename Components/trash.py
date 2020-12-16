


def predict_degree(graph, traffic_matrix, target_degree):
    random_rbc = get_randon_degree_rbc(graph)
    num_epoch = 122
    for i in range(1, num_epoch):
        loss_func = torch.nn.MSELoss()
        target = torch.tensor([1, 2, 1])
        loss = loss_func(approx_degree, target)
        a = 1


def get_randon_degree_rbc(graph: nx.DiGraph):
    n_nodes = graph.number_of_nodes()
    node_map = {k: v for v, k in enumerate(list(g.nodes()))}
    matrix = torch.full(size=(n_nodes, n_nodes, n_nodes, n_nodes), fill_value=0.0, requires_grad=True)
    for s, s_map in node_map:
        for t, t_map in node_map:
            if graph.has_edge(s, t):
                matrix[node_map[s], node_map[t]] = torch.rand(size=(n_nodes, n_nodes))
    return matrix


def approximate_degree(graph: nx.DiGraph, rand_rbc, traffic_matrix):
    node_map = {k: v for v, k in enumerate(list(g.nodes()))}
    for s, s_map in node_map.items():
        for t, t_map in node_map.items():
            rand_rbc[s_map, t_map] = rand_rbc[s_map, t_map] * t_tensor[s_map, t_map]
    approx_degree = torch.sum(randodm_rbc, dim=(0, 2, 3))



if __name__ == '__main__':


    g_nisuy = nx.read_edgelist(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_Graph_Dataset\test.edgelist', create_using=nx.DiGraph)
    deg_policy = Policy.DegreePolicy()
    node_map = {k: v for v, k in enumerate(list(g_nisuy.nodes()))}
    R = deg_policy.get_policy_tensor(g_nisuy, node_map)
    T = deg_policy.get_t_tensor(g_nisuy)
    R_flat = torch.flatten(R, start_dim=1)
    T_flat = torch.flatten(T, start_dim=1)
    data_set = MyOwnDataset(r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_Graph_Dataset')
    data = data_set[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(data_set, R_flat, T_flat).to(device)
    data = data_set[0].to(device)
    optimzer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    num_epocs = 200
    for epoch in range(num_epocs):
        print(f'{epoch} out of {num_epocs}')
        optimzer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        print(loss)
        loss.backward()
        optimzer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / sum(data.test_mask)
    print('Accuracy: {:.4f}'.format(acc))
