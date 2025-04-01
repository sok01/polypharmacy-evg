import json
from collections import defaultdict
from itertools import product

def load_graph(filename):
    # Загрузка графа и создание файлов
    with open(filename, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    # Создаем prepare_doses.json
    prepare_doses = {}
    for node in graph['nodes']:
        if node.get('label') == 'prepare':
            prepare_doses[node['id']] = {
                'name': node['name'],
                'dose': node.get('dose', 0.0)
            }
    
    with open('prepare_doses.json', 'w', encoding='utf-8') as f:
        json.dump(prepare_doses, f, ensure_ascii=False, indent=4)
    
    # Создаем probabilities.json
    parent_map = defaultdict(list)
    for link in graph['links']:
        parent_map[link['target']].append(link['source'])
    
    id_to_node = {n['id']: n for n in graph['nodes']}
    probabilities = {}
    
    for node in graph['nodes']:
        parents = parent_map.get(node['id'], [])
        
        if node.get('label') == 'prepare' or any(id_to_node[p].get('label') == 'prepare' for p in parents):
            probabilities[node['name']] = {"": 0.0}
        else:
            if not parents:
                probabilities[node['name']] = {"": 0.5}
            else:
                combs = list(product([0, 1], repeat=len(parents)))
                probabilities[node['name']] = {
                    ','.join(map(str, c)): 0.5 for c in combs
                }
    
    with open('probabilities.json', 'w', encoding='utf-8') as f:
        json.dump(probabilities, f, ensure_ascii=False, indent=4)
    
    return graph

class BayesianNode:
    def __init__(self, node_id, name, parents, label=None):
        self.id = node_id
        self.name = name
        self.parents = parents
        self.label = label
        self.prob_table = self._load_probabilities()
    
    def _load_probabilities(self):
        if self.label == 'prepare':
            with open('prepare_doses.json', 'r', encoding='utf-8') as f:
                doses = json.load(f)
                return {(): float(doses.get(self.id, {}).get('dose', 0.0))}
        else:
            with open('probabilities.json', 'r', encoding='utf-8') as f:
                probs = json.load(f)
                return self._parse_probabilities(probs.get(self.name, {}))
    
    def _parse_probabilities(self, prob_data):
        total = sum(float(v) for v in prob_data.values()) or 1
        parsed = {}
        for k, v in prob_data.items():
            key = tuple(map(int, k.split(','))) if k else ()
            parsed[key] = float(v) / total
        return parsed

def topological_sort(node_ids, parent_map):
    # Выполняет топологическую сортировку узлов для обработки узлов в правильном порядке. ДЛЯ расчета сети байеса (если больше времени - можно завязать на уровни)
    visited = set()
    result = []

    def visit(node_id):
        if node_id not in visited:
            visited.add(node_id)
            for parent in parent_map.get(node_id, []):
                visit(parent)
            result.append(node_id)

    for node_id in node_ids:
        visit(node_id)

    return result
def build_network(graph_data):
    nodes = {}
    id_to_parents = defaultdict(list)
    
    for link in graph_data['links']:
        id_to_parents[link['target']].append(link['source'])
    
    for node in graph_data['nodes']:
        node_id = node['id']
        nodes[node_id] = BayesianNode(
            node_id=node_id,
            name=node['name'],
            parents=id_to_parents[node_id],
            label=node.get('label')
        )
    
    return nodes

def calculate_prior_probabilities(bayesian_net):
    # Расчет априорных вероятностей для каждого узла в байесовской сети
    parent_map = {}
    for node in bayesian_net.values():
        parent_map[node.id] = node.parents  # Простой способ вместо сложного выражения

    sorted_ids = topological_sort(bayesian_net.keys(), parent_map)  # Топологическая сортировка
    prior_probs = {}

    for node_id in sorted_ids:
        node = bayesian_net[node_id]
        if not node.parents:
            prior_probs[node_id] = node.prob_table.get((), 0)  # 
        else:
            total = 0.0
            # Для каждой комбинации состояний родителей
            for comb, p_node in node.prob_table.items():
                prob_comb = 1.0
                for parent_id, state in zip(node.parents, comb):
                    parent_prob = prior_probs.get(parent_id, 0)  # Изменение здесь
                    prob_comb *= parent_prob if state == 1 else (1 - parent_prob)
                total += p_node * prob_comb
            prior_probs[node_id] = total

    return prior_probs

def save_results(network, filename='results.json'):
    results = []
    for node_id, node in network.items():
        entry = {
            'name': node.name,
            'label': node.label,
            'probability': node.prob_table.get((), 0.0),
            'negative_probability': 1 - node.prob_table.get((), 0.0)
        }
        
        if node.label == 'side_e':
            entry['ancestors'] = [
                network[p].name 
                for p in node.parents 
                if p in network
            ]
        
        results.append(entry)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# Пример использования
if __name__ == "__main__":
    graph = load_graph('drug_allopurinol_Parse.json')
    network = build_network(graph)
    probabilities = calculate_prior_probabilities(network)
    save_results(network)