import json
from collections import defaultdict
from itertools import product

class BayesianNode:
    """Класс для представления узла байесовской сети.
    
    Атрибуты:
        id (int): Уникальный идентификатор узла
        name (str): Человекочитаемое название узла
        parents (list): Список идентификаторов родительских узлов
        prob_table (dict): Таблица вероятностей в формате {набор статусов состояйний: вероятность}
    """
    
    def __init__(self, node_id, name, parents, prob_data):
        self.id = node_id
        self.name = name
        self.parents = parents
        self.prob_table = self._parse_probabilities(prob_data)

    def _parse_probabilities(self, prob_data):
        """Преобразует строковые ключи в кортежи целых чисел."""
        parsed = {}
        for k, v in prob_data.items():
            key = tuple(map(int, k.split(','))) if k else ()
            parsed[key] = float(v)
        return parsed

    def get_probability(self, parent_states):
        """Возвращает вероятность для заданной комбинации состояний родителей."""
        return self.prob_table.get(parent_states, 0.0)

def load_graph(filename):
    """Загружает структуру графа из JSON файла."""
    with open(filename, 'r') as f:
        return json.load(f)

def load_probabilities(filename='probabilities.json'):
    """Загружает таблицы условных вероятностей из JSON файла."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def generate_default_probabilities(graph_data):
    """Генерирует вероятности по умолчанию для всех узлов."""
    parent_map = defaultdict(list)
    for link in graph_data['links']:
        parent_map[link['target']].append(link['source'])

    default_probs = {}
    for node in graph_data['nodes']:
        node_name = node['name']
        parents = parent_map.get(node['id'], [])
        
        if not parents:
            # Узлы без родителей
            default_probs[node_name] = {"": 0.5}
        else:
            # Генерация всех комбинаций состояний родителей
            combs = list(product([0, 1], repeat=len(parents)))
            default_probs[node_name] = {
                ','.join(map(str, comb)): 0.5 for comb in combs
            }
    
    return default_probs

def topological_sort(nodes, parent_map):
    """Топологическая сортировка узлов графа (итеративная реализация)."""
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
                
                # Добавляем родителей в обратном порядке для сохранения порядка обработки
                for parent in reversed(parent_map.get(current, [])):
                    if parent not in visited:
                        stack.append((parent, False))
    
    return result

def build_network(graph_data, prob_data):
    """Строит байесовскую сеть на основе данных графа и вероятностей."""
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
    """Вычисляет априорные вероятности для всех узлов сети."""
    parent_map = {nid: node.parents for nid, node in network.items()}
    sorted_nodes = topological_sort(network.keys(), parent_map)
    probabilities = {}

    for node_id in sorted_nodes:
        node = network[node_id]
        
        if not node.parents:
            probabilities[node_id] = node.get_probability(())
            continue
            
        total = 0.0
        for comb, p_node in node.prob_table.items():
            # Вычисление вероятности комбинации состояний родителей
            prob_comb = 1.0
            for parent_id, state in zip(node.parents, comb):
                parent_prob = probabilities.get(parent_id, 0.0)
                prob_comb *= parent_prob if state == 1 else (1 - parent_prob)
            
            total += p_node * prob_comb
        
        probabilities[node_id] = total
    
    return probabilities

def get_conditional_probability(network, node_id, parent_states):
    """Возвращает условную вероятность узла для заданных состояний родителей."""
    node = network.get(node_id)
    if not node:
        raise ValueError(f"Узел {node_id} не найден в сети")
    
    if len(parent_states) != len(node.parents):
        raise ValueError("Неверное количество состояний родителей")
    
    return node.get_probability(parent_states)

def save_results(probs, network, filename='results.json'):
    """Сохраняет результаты расчетов в JSON файл."""
    results = []
    for node_id, prob in probs.items():
        results.append({
            'id': node_id,
            'name': network[node_id].name,
            'probability': round(prob, 4)
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Загрузка данных и инициализация сети
    graph = load_graph('drug_allopurinol_Parse.json')
    
    # Генерируем вероятности по умолчанию
    default_probabilities = generate_default_probabilities(graph)
    with open('probabilities.json', 'w', encoding='utf-8') as f:  # Обеспечиваем utf-8 кодировку
        json.dump(default_probabilities, f, indent=4, ensure_ascii=False)  # Оставляем структуру без изменений

    
    # Построение и расчет сети
    prob_data = load_probabilities()
    network = build_network(graph, prob_data)
    final_probs = calculate_probabilities(network)
    
    # Пример использования: получение условной вероятности
    try:
        sample_node_id = 1  # ID целевого узла
        parent_comb = (1, 0)  # Пример комбинации состояний родителей
        cond_prob = get_conditional_probability(network, sample_node_id, parent_comb)
        print(f"Условная вероятность для узла {sample_node_id}: {cond_prob:.2f}")
    except ValueError as e:
        print(f"Ошибка: {e}")
    
    # Сохранение результатов
    save_results(final_probs, network)
    print("Расчеты успешно завершены. Результаты сохранены в results.json")