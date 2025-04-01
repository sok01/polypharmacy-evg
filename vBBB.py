
import json
import random
from pyvis.network import Network
from collections import defaultdict
from itertools import product

def load_graph(filename):
    # Загружает Полинин граф из JSON файла
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_probabilities(filename='probabilities.json'):
    # Загружает исходные вероятности из JSON файла, если файл существует. если нет создает новый. ТАМ типа узел и комбинации его родителей.
    # например почка: на нее влияют три препората. для нее будет построена комбинация влияния препоратов с вероятностями (начально 1) 
    # тут надо дописать чтоб была рашифровка где какой препорат (я пытался вышло херово)
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def visualize_graph(graph_data, output_file='graph.html'):
    #оздаёт визуализацию графа в формате HTML - перестало работать. наверное пакет обновить надо
    net = Network(height='750px', width='100%', directed=True)

    for node in graph_data['nodes']:
        # Добавляем узлы в визуализацию
        net.add_node(
            node['id'],
            label=node['name'],
            level=node['level'],
            title=f"{node['name']} (Level {node['level']})"
        )

    for link in graph_data['links']:
        # Соединяем узлы рёбрами
        net.add_edge(link['source'], link['target'])

    net.show(output_file)

class BayesianNode:
    # ласс узла в байесовской сети
    def __init__(self, node_id, name, parents, prob_data):
        self.id = node_id
        self.name = name
        self.parents = parents
        self.prob_table = self._parse_probabilities(prob_data)

    def _parse_probabilities(self, prob_data):
        # парсит данные вероятностей, преобразуя их в удобную для работы форму. добавил когда строил файл результатов. чтоб сеть считалась по структуре 
        parsed = {}
        total = 0.0
        
        # Считаем сумму всех вероятностей
        for key_str, value in prob_data.items():
            total += float(value)
        
        # Защита от деления на ноль
        if total == 0:
            total = 1.0
            
        # Нормализуем каждое значение
        for key_str, value in prob_data.items():
            # Преобразуем ключ в кортеж чисел
            if key_str == "":
                key = ()
            else:
                key = tuple(map(int, key_str.split(',')))
            
            # Сохраняем нормализованное значение
            parsed[key] = float(value) / total
        
        return parsed
        # for k, v in prob_data.items():
        #     key = tuple(map(int, k.split(','))) if k else ()
        #     parsed[key] = float(v)  # Здесь ожидается только числовое значение
        # return parsed

def generate_default_probabilities(graph_data):
    # Генерирует вероятности по умолчанию для всех узлов в графе
    parent_map = defaultdict(list)
    for link in graph_data['links']:
        parent_map[link['target']].append(link['source'])

    # Составляем словарь, чтобы сопоставить идентификаторы узлов с их именами
    id_to_name = {}
    for node in graph_data['nodes']:
        id_to_name[node['id']] = node['name']
   

    default_probabilities = {}

    for node in graph_data['nodes']:
        node_id = node['id']
        node_name = node['name']
        parents = parent_map.get(node_id, [])

        if not parents:
            # Узел без родителей, вероятность 0. но по сути я его просто недобавляю, можно выводить 
            default_probabilities[node_name] = {"": 0}  # Пустая строка как ключ ОЧЕНЬ СВЯЗАНО С  prior_probs[node_id] = node.prob_table.get((), 0)
        else:
            # Узел с родителями, устанавливаем вероятности 1 для всех комбинаций
            state_probabilities = {}
            combs = list(product([0, 1], repeat=len(parents))) # вот ТУТ  менять чтоб поменять структуру файла вероятностей 

            for comb in combs:
                key = ','.join(map(str, comb))
                state_probabilities[key] = random.random() # Вероятность 1 для всех комбинаций

            default_probabilities[node_name] = state_probabilities

    return default_probabilities

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

def build_bayesian_network(graph_data, prob_data):
    # создаёт байесовскую сеть из графа и данных вероятностей
    parent_map = defaultdict(list)
    for link in graph_data['links']:
        parent_map[link['target']].append(link['source'])

    nodes = {}
    for node in graph_data['nodes']:
        node_id = node['id']
        parents = parent_map.get(node_id, [])
        nodes[node_id] = BayesianNode(
            node_id=node_id,
            name=node['name'],
            parents=parents,
            prob_data=prob_data.get(node['name'], {})  # Используем имя узла вместо id
        )
    return nodes

# def calculate_prior_probabilities(bayesian_net):
#     # Расчет априорных вероятностей для каждого узла в байесовской сети
#     parent_map = {node.id: node.parents for node in bayesian_net.values()}
#     sorted_ids = topological_sort(bayesian_net.keys(), parent_map)  # Топологическая сортировка
#     prior_probs = {}

#     for node_id in sorted_ids:
#         node = bayesian_net[node_id]
#         if not node.parents:
#             prior_probs[node_id] = node.prob_table[()]
#         else:
#             total = 0.0
#             # Для каждой комбинации состояний родителей
#             for comb, p_node in node.prob_table.items():
#                 prob_comb = 1.0
#                 for parent_id, state in zip(node.parents, comb):
#                     parent_prob = prior_probs[parent_id]
#                     prob_comb *= parent_prob if state == 1 else (1 - parent_prob)
#                 total += p_node * prob_comb
#             prior_probs[node_id] = total

#     return prior_probs

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


def save_results_to_json(prior_probs, bayesian_net, filename='results.json'):
    # сохраняет результаты предсказаний в JSON файл
    results = []
    for node_id, prob in prior_probs.items():
        results.append({
            'node_id': node_id,
            'node_name': bayesian_net[node_id].name,
            'probability': prob
        })

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    graph_data = load_graph('drug_allopurinol_Parse.json')
    
    # Генерируем вероятности по умолчанию
    default_probabilities = generate_default_probabilities(graph_data)
    with open('probabilities.json', 'w', encoding='utf-8') as f:  # Обеспечиваем utf-8 кодировку
        json.dump(default_probabilities, f, indent=4, ensure_ascii=False)  # Оставляем структуру без изменений

    # Визуализируем граф структуры НЕ РАБОТАЕТ БИБЛИОТЕКА
    # visualize_graph(graph_data)

    # Загружаем вероятности и строим байесовскую сеть
    prob_data = load_probabilities()
    bayesian_net = build_bayesian_network(graph_data, prob_data)
    
    # Вычисляем приоритетные вероятности
    prior_probs = calculate_prior_probabilities(bayesian_net)

    # Сохраняем результаты
    save_results_to_json(prior_probs, bayesian_net)

    print("Байесовская сеть успешно создана, вероятности записаны в файл probabilities.json и предсказания в файл results.json.")
