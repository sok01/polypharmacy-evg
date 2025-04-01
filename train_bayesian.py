"""
Модуль для оптимизации вероятностей в байесовской сети
"""

import json
from collections import defaultdict
from itertools import product
import numpy as np
from scipy.optimize import minimize
from coolBNet import load_graph, build_network, calculate_probabilities, load_combined_data

# Константы
CONVERT_RATE = {
    "Очень часто": 0.55,
    "Часто": 0.055,
    "Нечасто": 0.0055,
    "Редко": 0.00055,
    "Очень редко": 0.00055,
    "Частота неизвестна": 0.000055,
}

GRAPH_FILE = 'merged_graph_fozinopril_ramipril_Parse.json'
TARGET_FILE = 'sef_dataset.json'
PROB_FILE_OPT = 'probabilities_opt.json'
DEFAULT_PROB = 0.0  # Значение по умолчанию для неизвестных вероятностей

def load_target_data() -> dict:
    """Загрузка и преобразование целевых данных из JSON"""
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return {
        drug: {
            effect: CONVERT_RATE.get(freq, DEFAULT_PROB)
            for effect, freq in info['side_e_parts'].items()
        }
        for drug, info in data.items()
    }

def get_drug_mapping(graph_data: dict) -> dict:
    """
    Создает карту препаратов и связанных с ними побочных эффектов
    Формат: {drug_name: (prepare_node_id, [side_effect_ids])}
    """
    adj = defaultdict(list)
    for link in graph_data['links']:
        adj[link['target']].append(link['source'])

    drug_map = {}
    for node in graph_data['nodes']:
        if node.get('label') != 'prepare':
            continue

        prepare_id = node['id']
        drug_name = node['name']
        visited = set()
        stack = [prepare_id]
        side_effects = []

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            # Поиск информации о текущем узле с обработкой отсутствия узла
            current_node = next(
                (n for n in graph_data['nodes'] if n['id'] == current), 
                None
            )
            if not current_node:
                continue

            if current_node.get('label') in {'side_effect', 'effect'}:
                side_effects.append(current)

            # Добавляем связанные узлы в стек
            stack.extend(adj.get(current, []))

        drug_map[drug_name] = (prepare_id, side_effects)
    
    return drug_map

def prepare_optimization_data(graph_data: dict, target_data: dict) -> list:
    """Подготовка данных для оптимизации в формате (node_id, target_value)"""
    drug_map = get_drug_mapping(graph_data)
    loss_targets = []

    for drug_name, effects in target_data.items():
        if drug_name not in drug_map:
            continue

        prepare_id, se_nodes = drug_map[drug_name]
        for se_node in se_nodes:
            # Безопасный поиск узла с обработкой отсутствия
            node_info = next(
                (n for n in graph_data['nodes'] if n['id'] == se_node),
                None
            )
            if not node_info:
                continue

            se_name = node_info.get('name')
            if se_name in effects:
                loss_targets.append((
                    se_node,
                    effects[se_name]
                ))
    
    return loss_targets

def get_zero_nodes(graph_data: dict) -> set:
    """Находит узлы с нулевыми вероятностями (prepare и их предков)"""
    parent_map = defaultdict(list)
    for link in graph_data['links']:
        parent_map[link['target']].append(link['source'])

    zero_nodes = set()
    prepare_nodes = [
        n['id'] for n in graph_data['nodes'] 
        if n.get('label') == 'prepare'
    ]

    def add_ancestors(node_id: str):
        """Рекурсивно добавляет предков узла"""
        stack = [node_id]
        while stack:
            current = stack.pop()
            zero_nodes.add(current)
            stack.extend(parent_map.get(current, []))

    for node_id in prepare_nodes:
        add_ancestors(node_id)
    
    return zero_nodes

class ProbabilityOptimizer:
    """Класс для оптимизации вероятностей в байесовской сети"""
    
    def __init__(self, graph_data: dict, target_data: dict):
        self.graph_data = graph_data
        self.target_pairs = prepare_optimization_data(graph_data, target_data)
        self.initial_probs = load_combined_data()
        self.zero_nodes = get_zero_nodes(graph_data)
        self.param_map = []
        self.initial_params = []
        self._prepare_optimization_parameters()

    def _prepare_optimization_parameters(self):
        """Подготовка параметров для оптимизации"""
        for node in self.graph_data['nodes']:
            if node['id'] in self.zero_nodes:
                continue

            node_name = node['name']
            probs = self.initial_probs.get(node_name, {})
            
            for key in probs:
                self.param_map.append((node_name, key))
                self.initial_params.append(probs[key])

        self.bounds = [(0.0, 1.0)] * len(self.initial_params)

    def update_probs(self, params: np.ndarray) -> dict:
        """Обновление вероятностей по текущим параметрам"""
        new_probs = defaultdict(dict)
        for i, (name, key) in enumerate(self.param_map):
            new_probs[name][key] = params[i]
        return new_probs

    def calculate_loss(self, params: np.ndarray) -> float:
        """Вычисление функции потерь MSE"""
        probs = self.update_probs(params)
        network = build_network(self.graph_data, probs)
        calculated = calculate_probabilities(network)
        
        loss = 0.0
        for node_id, target in self.target_pairs:
            loss += (calculated.get(node_id, DEFAULT_PROB) - target) ** 2
        return loss

    def optimize(self, method: str = 'L-BFGS-B', maxiter: int = 100):
        """Запуск оптимизации"""
        result = minimize(
            self.calculate_loss,
            self.initial_params,
            method=method,
            bounds=self.bounds,
            options={
                'maxiter': maxiter,
                'disp': True
            }
        )
        
        optimized_probs = self.update_probs(result.x)
        with open(PROB_FILE_OPT, 'w', encoding='utf-8') as f:
            json.dump(optimized_probs, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Инициализация и запуск
    try:
        graph = load_graph(GRAPH_FILE)
        targets = load_target_data()
        
        optimizer = ProbabilityOptimizer(graph, targets)
        optimizer.optimize()
        
        print(f"Оптимизация завершена. Результаты сохранены в {PROB_FILE_OPT}")
    except Exception as e:
        print(f"Ошибка выполнения: {str(e)}")