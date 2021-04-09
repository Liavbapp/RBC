train_combined = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\15',
                  r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\9',
                  r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\10',
                  r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\7_nodes\2_edges\12',
                  r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\396',
                  r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\397',
                  r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\398',
                  r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\399',
                  r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\400'
                  ]


class Same_Graph_Same_Routing_4_nodes:
    paths = [r'C:\Users\LiavB\OneDrive\Desktop\Msc\Thesis\Code\RBC_results\SPBC\4_nodes\2_edges\390'] * 10
    graphs_desc = f'Same Graph {len(paths)} times, With Same Routing, 4 Nodes'


class Same_Graph_Same_Routing_15_nodes:
    paths = [r'C:\Users\LiavB\PycharmProjects\RBC\results\matrices\SPBC\15_nodes\2_edges\9'] * 10
    graphs_desc = f'Same Graph {len(paths)} times, With Same Routing, 15 Nodes'


class Same_Graph_Same_Routing_11_nodes:
    paths = [r'C:\Users\LiavB\PycharmProjects\RBC\results\matrices\SPBC\11_nodes\2_edges\0'] * 10
    graph_desc = f'Same Graph {len(paths)} times, With Same Routing, 11 Nodes'


class Same_Graph_Same_Routing_9_nodes:
    paths = [r'C:\Users\LiavB\PycharmProjects\RBC\results\matrices\SPBC\9_nodes\2_edges\3'] * 10
    graph_desc = f'Same Graph {len(paths)} times, With Same Routing, 9 Nodes'


class Same_Graph_Different_Routing_4_nodes:
    paths = [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\SPBC\\4_nodes\\2_edges\\{str(i)}'
             for i in range(390, 401)]
    graph_desc = f'Same Graph {len(paths)} times, Different Routing, 4 Nodes'


class Same_Graph_Different_Routing_7_nodes:
    paths = [f'C:\\Users\\LiavB\\OneDrive\\Desktop\\Msc\\Thesis\\Code\\RBC_results\\SPBC\\7_nodes\\2_edges\\{str(i)}'
             for i in range(12, 23)]
    graph_desc = f'Same Graph {len(paths)} times, Different Routing, 4 Nodes'
