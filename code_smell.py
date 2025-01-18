

import ast

def parse_code_to_ast(code):
    return ast.parse(code)


from radon.complexity import cc_visit
from radon.metrics import h_visit, HalsteadReport
from radon.raw import analyze

def extract_static_metrics(code):
    metrics = {}
    
    complexity = cc_visit(code)
    metrics['cyclomatic_complexity'] = sum([c.complexity for c in complexity])
    
    
    halstead = h_visit(code)
    metrics.update(halstead._asdict())

    
    raw = analyze(code)
    metrics.update(raw._asdict())
    
    return metrics


from transformers import RobertaTokenizer, RobertaModel
import torch

tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
model = RobertaModel.from_pretrained("huggingface/CodeBERTa-small-v1")

def get_semantic_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  
    return embedding.detach().numpy()


import numpy as np

def flatten_halstead_report(report):
    if isinstance(report, HalsteadReport):
        return [
            report.h1, report.h2, report.N1, report.N2, 
            report.vocabulary, report.length, report.calculated_length,
            report.volume, report.difficulty, report.effort,
            report.time, report.bugs
        ]
    return [0] * 12  

def process_metrics(metrics):
    
    numeric_features = [
        metrics.get('cyclomatic_complexity', 0),
        metrics.get('loc', 0),
        metrics.get('lloc', 0),
        metrics.get('sloc', 0),
        metrics.get('comments', 0),
        metrics.get('multi', 0),
        metrics.get('blank', 0),
        metrics.get('single_comments', 0),
    ]

    
    if 'total' in metrics:
        numeric_features += flatten_halstead_report(metrics['total'])
    else:
        numeric_features += [0] * 12

    
    function_reports = metrics.get('functions', [])
    if function_reports:
        for func_name, report in function_reports:
            numeric_features += flatten_halstead_report(report)
    else:
        
        numeric_features += [0] * 12

    return np.array(numeric_features)

def combine_features(metrics, embedding):
    
    metrics_vector = process_metrics(metrics)
    
    
    feature_vector = np.concatenate([metrics_vector, embedding.flatten()])
    
    return feature_vector


from sklearn.decomposition import PCA

def reduce_dimensions(features, n_components=20):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features


import networkx as nx

def build_graph(tree, features):
    graph = nx.DiGraph()
    nodes = list(ast.walk(tree))  

    
    if len(features) != len(nodes):
        raise ValueError("The number of features does not match the number of AST nodes.")

    
    for i, node in enumerate(nodes):
        graph.add_node(i, features=features[i])  

    
    for parent_index, parent_node in enumerate(nodes):
        for child_node in ast.iter_child_nodes(parent_node):
            child_index = nodes.index(child_node)  
            graph.add_edge(parent_index, child_index)

    return graph



def optimize_graph(graph):
    
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)
    return graph


from torch_geometric.data import Data

def create_graph_data(graph):
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = torch.tensor([graph.nodes[n]['features'] for n in graph.nodes], dtype=torch.float)
    y = torch.zeros(len(graph.nodes))  
    return Data(x=x, edge_index=edge_index, y=y)


code = """
def example_function(a, b):
    result = a + b
    if result > 10:
        return "Large"
    else:
        return "Small"
"""


tree = parse_code_to_ast(code)
metrics = extract_static_metrics(code)
embedding = get_semantic_embedding(code)
feature_vector = combine_features(metrics, embedding)


features = [feature_vector] * len(list(ast.walk(tree)))  
reduced_features = reduce_dimensions(np.array(features))


graph = build_graph(tree, reduced_features)
optimized_graph = optimize_graph(graph)


graph_data = create_graph_data(optimized_graph)



import torch
from torch_geometric.nn import GCNConv


class GCNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def predict(data, model):
    with torch.no_grad():
        logits = model(data)  
        predictions = torch.argmax(logits, dim=1)  
    return predictions


input_dim = graph_data.x.shape[1]  
hidden_dim = 16  
output_dim = 2  


model = GCNClassifier(input_dim, hidden_dim, output_dim)


predictions = predict(graph_data, model)


for i, prediction in enumerate(predictions):
    print(f"Node {i}: {'Smelly' if prediction.item() == 1 else 'Not Smelly'}")






