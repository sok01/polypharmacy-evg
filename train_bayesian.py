"""
Оптимизация вероятностей для байесовской сети
Упрощенная версия для работы с sef_dataset.json
"""

import json
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

# Импорт функций из модуля coolBNet
from coolBNet import load_graph, build_network, calculate_probabilities, load_combined_data

# Константы для преобразования частот в вероятности
FREQUENCY_TO_PROB = {
    "Очень часто": 0.55,
    "Часто": 0.055,
    "Нечасто": 0.0055,
    "Редко": 0.00055,
    "Очень редко": 0.00055,
    "Частота неизвестна": 0.000055,
}

# Конфигурационные файлы
GRAPH_FILE = 'merged_graph_fozinopril_ramipril_Parse.json'
TARGET_FILE = 'sef_dataset.json'
OUTPUT_FILE = 'probabilities_opt.json'

def load_target_effects():
    """Загружает данные о побочных эффектах из sef_dataset.json"""
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    effects = {}
    for drug, info in data.items():
        # Используем правильное имя поля 'side_e_parts' вместо 'side_effects'
        effects[drug] = {
            effect: FREQUENCY_TO_PROB.get(freq, 0.0)
            for effect, freq in info['side_e_parts'].items()
        }
    return effects

def map_drugs_to_effects(graph):
    """Создает соответствие между препаратами и их побочными эффектами"""
    drug_effects = {}
    
    # Находим все препараты (узлы с label='prepare')
    drugs = [node for node in graph['nodes'] if node.get('label') == 'prepare']
    
    # Для каждого препарата находим связанные эффекты
    for drug in drugs:
        effect_ids = find_effects(graph, drug['id'])
        drug_effects[drug['name']] = (drug['id'], effect_ids)
    
    return drug_effects

def find_effects(graph, start_node_id):
    """Находит все побочные эффекты для заданного узла"""
    effects = []
    visited = set()
    stack = [start_node_id]
    
    # Строим словарь связей
    links = defaultdict(list)
    for link in graph['links']:
        links[link['target']].append(link['source'])
    
    # Обход графа в глубину
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        
        # Находим узел по ID
        node = next((n for n in graph['nodes'] if n['id'] == node_id), None)
        if not node:
            continue
        
        # Проверяем, является ли узел побочным эффектом
        if node.get('label') in ['side_effect', 'effect']:
            effects.append(node_id)
        
        # Добавляем связанные узлы в стек
        stack.extend(links.get(node_id, []))
    
    return effects

class ProbabilityOptimizer:
    """Класс для оптимизации вероятностей"""
    
    def __init__(self, graph, target_effects):
        self.graph = graph
        self.target_effects = target_effects
        self.drug_map = map_drugs_to_effects(graph)
        self.initial_probs = load_combined_data()
        
        # Подготовка данных для оптимизации
        self.prepare_optimization_data()
        
        # Ограничения для параметров (0 <= p <= 1)
        self.bounds = [(0.0, 1.0) for _ in self.initial_params]
    
    def prepare_optimization_data(self):
        """Подготавливает данные для оптимизации"""
        self.param_map = []  # Соответствие параметров узлам
        self.initial_params = []  # Начальные значения параметров
        self.target_pairs = []  # Целевые значения
        
        # 1. Заполняем параметры для всех узлов (кроме препаратов)
        for node in self.graph['nodes']:
            if node.get('label') == 'prepare':
                continue
                
            probs = self.initial_probs.get(node['name'], {})
            for key, value in probs.items():
                self.param_map.append((node['name'], key))
                self.initial_params.append(value)
        
        # 2. Создаем пары (ID эффекта, целевая вероятность)
        for drug_name, effects in self.target_effects.items():
            if drug_name not in self.drug_map:
                print(f'Препарат {drug_name} не найден в графе')
                continue
                
            _, effect_ids = self.drug_map[drug_name]
            for effect_id in effect_ids:
                # Находим название эффекта по ID
                effect_node = next(
                    (n for n in self.graph['nodes'] if n['id'] == effect_id), 
                    None
                )
                if effect_node and effect_node['name'] in effects:
                    self.target_pairs.append((
                        effect_id, 
                        effects[effect_node['name']]
                    ))
    
    def optimize(self):
        """Запускает процесс оптимизации"""
        result = minimize(
            self.calculate_loss,
            self.initial_params,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': 100}
        )
        
        self.save_results(result.x)
    
    def calculate_loss(self, params):
        """Вычисляет функцию потерь"""
        # Обновляем вероятности
        probs = self.update_probs(params)
        
        # Строим сеть и вычисляем вероятности
        network = build_network(self.graph, probs)
        calculated = calculate_probabilities(network)
        
        # Вычисляем среднеквадратичную ошибку
        loss = 0.0
        for node_id, target in self.target_pairs:
            loss += (calculated.get(node_id, 0.0) - target) ** 2
        
        return loss
    
    def update_probs(self, params):
        """Обновляет словарь вероятностей по параметрам"""
        new_probs = defaultdict(dict)
        for i, (name, key) in enumerate(self.param_map):
            # Корректируем значение, если оно вышло за границы
            value = max(0.0, min(1.0, params[i]))
            new_probs[name][key] = value
        return new_probs
    
    def save_results(self, params):
        """Сохраняет оптимизированные вероятности в файл"""
        optimized_probs = self.update_probs(params)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(optimized_probs, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    print("Начало оптимизации...")
    
    try:
        # 1. Загружаем данные
        graph = load_graph(GRAPH_FILE)
        target_effects = load_target_effects()
        
        # 2. Создаем и запускаем оптимизатор
        optimizer = ProbabilityOptimizer(graph, target_effects)
        optimizer.optimize()
        
        print(f"Оптимизация завершена. Результаты сохранены в {OUTPUT_FILE}")
    
    except Exception as e:
        print(f"Ошибка: {str(e)}")