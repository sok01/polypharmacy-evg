import json
import random
from collections import defaultdict
from itertools import product

class BayesianNode:
    # Класс для представления узла байесовской сети.
    
    # Атрибуты:
    #     id (int): Уникальный идентификатор узла
    #     name (str): Человекочитаемое название узла
    #     parents (list): Список идентификаторов родительских узлов
    #     prob_table (dict): Таблица вероятностей в формате {набор статусов состояйний: вероятность}

    
    def __init__(self, node_id, name, parents, prob_data):
        self.id = node_id
        self.name = name
        self.parents = parents
        self.prob_table = self._parse_probabilities(prob_data)

    def _parse_probabilities(self, prob_data):
        # Преобразует строковые ключи в кортежи целых чисел.
        parsed = {}
        for k, v in prob_data.items():
            key = tuple(map(int, k.split(','))) if k else ()
            parsed[key] = float(v)
        return parsed

    def get_probability(self, parent_states):
        # Возвращает вероятность для заданной комбинации состояний родителей.
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
    # Генерирует вероятности по умолчанию для всех узлов графа
    
    # 1. Создаем словарь связей "потомок -> родители"
    parent_map = defaultdict(list)
    for link in graph_data['links']:
        parent_map[link['target']].append(link['source'])  # target - узел, source - его родитель

    # 2. Инициализируем словарь для хранения вероятностей
    default_probs = {}

    # 3. Обрабатываем каждый узел в графе
    for node in graph_data['nodes']:
        node_name = node['name']
        node_id = node['id']
        
        # 4. Получаем список родительских ID для текущего узла
        parents = parent_map.get(node_id, [])  # Пустой список если родителей нет

        if not parents:
            # 5. Случай узла БЕЗ родителей
            # Используем специальный ключ "" и базовую вероятность 0
            default_probs[node_name] = {"": 0} # вводить нули или единицы для (приняли не приняли) скорее всего отдельный json (интерфейс)
        else:
            default_probs[node_name] = {}
            # 6. Случай узла С родителями
            # Генерируем все возможные комбинации состояний родителей (0/1)
            combs = list(product([0, 1], repeat=len(parents)))
            
            # 7. Создаем записи для каждой комбинации
            # Ключ: строка вида "0,1,0", Значение: 0.5 по умолчанию
            # default_probs[node_name] = {
            #     ','.join(map(str, comb)): 0.5 for comb in combs
            # }
            for comb in combs:
                # Преобразуем комбинацию в строковый ключ
                # # Например, кортеж (0, 1) -> строка "0,1"
                key = ','.join(map(str, comb))
                # Устанавливаем вероятность по умолчанию 0.5
                default_probs[node_name][key] = random.random() 

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

            # Расчет для узлов с родителями
            total = 0.0
            f.write(f"Комбинации состояний родителей ({len(node.prob_table)}):\n")
            
            for i, (comb, p_node) in enumerate(node.prob_table.items(), 1):
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
    # graph = load_graph('drug_allopurinol_Parse.json')
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