
import json
from collections import defaultdict

# Загружаем граф из JSON файла
with open('s_graph.json', 'r', encoding='utf-8') as file:
    graph_data = json.load(file)

# Извлекаем узлы и ссылки
nodes = graph_data['nodes']
links = graph_data['links']

# Создаем отображение для хранения связей
graph_dict = defaultdict(list)

# Формируем граф как словарь
for link in links:
    source = link['source']
    target = link['target']
    graph_dict[source].append(target)

# **Формируем таблицы для каждого узла**
tables = {}
for node in nodes:
    node_id = node['id']
    # Заполняем таблицу для данного узла
    if node_id in graph_dict:
        tables[node_id] = graph_dict[node_id]
    else:
        tables[node_id] = []

# Выводим таблицы
for node_id, connected_nodes in tables.items():
    print(f'Узел: {node_id}')
    print('Входящие узлы:')
    for connected_node in connected_nodes:
        print(f'- {connected_node}')
    print()
