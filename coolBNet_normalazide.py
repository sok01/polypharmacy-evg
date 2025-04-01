import json
import random
from collections import defaultdict
from itertools import product

class BayesianNode:
    def __init__(self, node_id, name, parents, prob_data):
        self.id = node_id
        self.name = name
        self.parents = parents
        self.prob_table = self._parse_probabilities(prob_data)

    def _parse_probabilities(self, prob_data):
        parsed = {}
        for k, v in prob_data.items():
            key = tuple(map(int, k.split(','))) if k else ()
            parsed[key] = float(v)
        return parsed

    def get_probability(self, parent_states):
        return self.prob_table.get(parent_states, 0.0)

def load_graph(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def load_probabilities(filename='probabilities.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def create_base_probabilities(graph_data, output_file='probabilities.json'):
    # Инициализация структур данных
    parent_map = defaultdict(list)
    id_to_node = {n['id']: n for n in graph_data['nodes']}
    
    # Построение карты родителей
    for link in graph_data['links']:
        parent_map[link['target']].append(link['source'])
    
    # Находим все prepare-узлы и их предков
    prepare_nodes = [n['id'] for n in graph_data['nodes'] if n.get('label') == 'prepare']
    zero_nodes = set()
    
    # Рекурсивный поиск предков
    def find_ancestors(node_id):
        ancestors = set()
        for parent in parent_map.get(node_id, []):
            ancestors.add(parent)
            ancestors.update(find_ancestors(parent))
        return ancestors
    
    # Собираем все узлы для обнуления
    for node_id in prepare_nodes:
        zero_nodes.add(node_id)
        zero_nodes.update(find_ancestors(node_id))
    
    # Генерация вероятностей
    probabilities = {}
    for node in graph_data['nodes']:
        node_id = node['id']
        parents = parent_map.get(node_id, [])
        
        # Для нулевых узлов
        if node_id in zero_nodes:
            if not parents:
                probabilities[node['name']] = {"": 0.0}
            else:
                combs = product([0], repeat=len(parents))  # Все комбинации нулей
                probabilities[node['name']] = {','.join(map(str, c)): 0.0 for c in combs}
        else:
            # Случайные вероятности для остальных
            if not parents:
                probabilities[node['name']] = {"": random.random()}
            else:
                combs = product([0, 1], repeat=len(parents))
                probabilities[node['name']] = {','.join(map(str, c)): random.random() for c in combs}
    
    # Сохранение в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(probabilities, f, indent=4, ensure_ascii=False)


# 2. Функция создания файла доз
def create_doses_file(graph_data, output_file='prepare_doses.json'):
    prepare_nodes = [n['name'] for n in graph_data['nodes'] if n.get('label') == 'prepare']
    doses = {name: 0.0 for name in prepare_nodes}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(doses, f, indent=4, ensure_ascii=False)


# 3. Функция загрузки и объединения данных
def load_combined_data(prob_file='probabilities.json', doses_file='prepare_doses.json'):
    with open(prob_file) as f:
        probabilities = json.load(f)
    
    try:
        with open(doses_file) as f:
            doses = json.load(f)
    except FileNotFoundError:
        return probabilities
    
    # Обновляем вероятности из файла доз
    for name, value in doses.items():
        if name in probabilities:
            # Для узлов без родителей
            if "" in probabilities[name]:
                probabilities[name][""] = float(value)
            else:
                # Для узлов с родителями (обновляем все комбинации)
                probabilities[name] = {k: float(value) for k in probabilities[name]}
    
    return probabilities



def topological_sort(nodes, parent_map):
    visited = set()
    result = []
    stack = []
    
    for node in nodes:
        if node not in visited:
            stack.append((node, False))
            while stack:
                current, processed = stack.pop()
                if processed:
                    result.append(current)
                    continue
                if current in visited:
                    continue
                visited.add(current)
                stack.append((current, True))
                for parent in reversed(parent_map.get(current, [])):
                    if parent not in visited:
                        stack.append((parent, False))
    return result

def build_network(graph_data, prob_data):
    parent_map = defaultdict(list)
    for link in graph_data['links']:
        parent_map[link['target']].append(link['source'])
    
    nodes = {}
    for node in graph_data['nodes']:
        node_id = node['id']
        nodes[node_id] = BayesianNode(
            node_id=node_id,
            name=node['name'],
            parents=parent_map.get(node_id, []),
            prob_data=prob_data.get(node['name'], {})
        )
    return nodes

def calculate_probabilities(network):
    """Вычисляет априорные вероятности для всех узлов сети"""
    
    # 1. Создаем словарь связей "узел -> его родители"
    parent_map = {nid: node.parents for nid, node in network.items()}
    
    # 2. Топологическая сортировка узлов
    sorted_nodes = topological_sort(network.keys(), parent_map)
    
    # 3. Словарь для накопления результатов
    probabilities = {}

    # Открываем файл для записи логов
    with open('calculation_trace.txt', 'w', encoding='utf-8') as f:
        # 4. Заголовок лога
        f.write("\n" + "="*50 + "\n")
        f.write("НАЧАЛО РАСЧЕТА ВЕРОЯТНОСТЕЙ\n")
        f.write(f"Порядок обработки узлов: {sorted_nodes}\n")
        f.write("="*50 + "\n\n")

        # 5. Обработка узлов
        for node_id in sorted_nodes:
            node = network[node_id]
            f.write(f"[Узел {node_id} '{node.name}']\n")
            f.write(f"Тип: {'Корневой' if not node.parents else 'Дочерний'}\n")
            f.write(f"Родители: {node.parents or 'нет'}\n")

            if not node.parents:
                # Узел без родителей
                prob = node.get_probability(())
                probabilities[node_id] = prob
                f.write(f"Вероятность из таблицы: {prob:.4f}\n")
                f.write("-"*50 + "\n\n")
                continue

            total_conditional = sum(node.prob_table.values())
            normalized_probs = {}
            
            if abs(total_conditional - 1.0) > 1e-9:
                f.write(f"! Нормализация условных вероятностей (исходная сумма: {total_conditional:.4f})\n")
                for comb, p in node.prob_table.items():
                    normalized_probs[comb] = p / total_conditional if total_conditional != 0 else 0.0
            else:
                normalized_probs = node.prob_table


            # Расчет для узлов с родителями
            total = 0.0
            # f.write(f"Комбинации состояний родителей ({len(node.prob_table)}):\n")

            f.write(f"Комбинации состояний родителей ({len(normalized_probs)}):\n")
            
            for i, (comb, p_node) in enumerate(normalized_probs.items(), 1):
                prob_comb = 1.0
                comb_str = ",".join(map(str, comb))
                f.write(f"\nКомбинация {i}: {comb_str}\n")
                f.write(f"P({node.name}|{comb_str}) = {p_node:.4f}\n")

                for j, (parent_id, state) in enumerate(zip(node.parents, comb), 1):
                    parent_prob = probabilities.get(parent_id, 0.0)
                    parent = network[parent_id]
                    operation = "P" if state == 1 else "1-P"
                    value = parent_prob if state == 1 else (1 - parent_prob)
                    
                    f.write(f"  Родитель {j}: {parent.name} (ID {parent_id})\n")
                    f.write(f"  Состояние: {state} → {operation}({parent_prob:.4f}) = {value:.4f}\n")

                    prob_comb *= value
                    f.write(f"  Текущая prob_comb: {prob_comb:.4f}\n")

                contribution = p_node * prob_comb   
                tr_temp = total
                total = total + contribution
                f.write(f"Вклад комбинации: {p_node:.4f} * {prob_comb:.4f} = {contribution:.4f}\n")
                f.write(f"!!!Накопление суммы влияний: {tr_temp:.4f} + {contribution:.4f} = {total:.4f}\n")
                f.write(f"Накопленный сумма (свертка): {total:.4f}\n")

            probabilities[node_id] = total
            f.write(f"\nИтоговая вероятность: {total:.4f}\n")
            f.write("="*50 + "\n\n")

    return probabilities

def get_conditional_probability(network, node_id, parent_states):
    node = network.get(node_id)
    if not node:
        raise ValueError(f"Узел {node_id} не найден")
    return node.get_probability(parent_states)

def save_results(probs, network, graph_data, filename='results.json'):
    parent_map = defaultdict(list)
    for link in graph_data['links']:
        parent_map[link['target']].append(link['source'])
    
    def get_ancestors(node_id):
        ancestors = set()
        def dfs(current_id):
            for parent in parent_map.get(current_id, []):
                if parent not in ancestors:
                    ancestors.add(parent)
                    dfs(parent)
        dfs(node_id)
        return ancestors
    
    results = []
    for node_id, prob in probs.items():
        node_info = next(n for n in graph_data['nodes'] if n['id'] == node_id)
        entry = {
            'id': node_id,
            'name': network[node_id].name,
            'probability': round(prob, 4)
        }
        if node_info.get('label') == 'side_e':
            ancestors = get_ancestors(node_id)
            entry['ancestors'] = [network[anc].name for anc in ancestors]
        results.append(entry)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Загрузка графа
    with open('fozinopril_ramipril_Parse.json') as f:
        graph = json.load(f)
    
    # Шаг 1: Создать базовые вероятности
    # create_base_probabilities(graph)
    
    # Шаг 2: Создать файл доз
    # create_doses_file(graph)
    
    # Шаг 3: Загрузить объединенные данные
    prob_data = load_combined_data()
    
    # Построение и расчет сети
    network = build_network(graph, prob_data)
    final_probs = calculate_probabilities(network)
    
    # Сохранение результатов
    save_results(final_probs, network, graph)
    
    print("Расчеты успешно завершены. Результаты в results.json")