# optimizer.py
import json
import copy
import random
import numpy as np
from coolBNet import load_graph, create_base_probabilities, load_combined_data, build_network, calculate_probabilities, save_results
from dbCode import ComputeObjectF, DB_SIDE_E, CONVERT_RATE

class ProbabilityOptimizer:
    def __init__(self, graph_file, db_path):
        # Загрузка исходных данных
        self.graph = load_graph(graph_file)
        self.original_probs = load_combined_data()
        self.current_probs = copy.deepcopy(self.original_probs)
        self.best_probs = copy.deepcopy(self.original_probs)
        self.min_loss = float('inf')
        
        with open('merged_graph_fozinopril_ramipril_Tables.json') as f:
            drug_table_data = json.load(f)
            
        self.compute_obj = ComputeObjectF(
            drug_table_data,  # Передаем загруженные табличные данные
            self.graph,
            DB_SIDE_E
        )
        
        # Собираем информацию о графе
        self._prepare_graph_info()
        
    def _prepare_graph_info(self):
        """Собирает информацию о связях между узлами"""
        self.node_map = {n['id']: n for n in self.graph['nodes']}
        self.parent_map = defaultdict(list)
        self.child_map = defaultdict(list)
        
        for link in self.graph['links']:
            self.parent_map[link['target']].append(link['source'])
            self.child_map[link['source']].append(link['target'])
            
        # Находим все side_e узлы
        self.side_e_nodes = [
            n['id'] for n in self.graph['nodes'] 
            if n.get('label') == 'side_e'
        ]
        
    def _get_affected_nodes(self, node_id):
        """Возвращает все узлы, которые зависят от измененного узла"""
        affected = set()
        queue = [node_id]
        
        while queue:
            current = queue.pop()
            affected.add(current)
            # Добавляем всех детей, которые еще не обработаны
            for child in self.child_map.get(current, []):
                if child not in affected:
                    queue.append(child)
                    
        return affected
    
    def calculate_current_loss(self):
        """Вычисляет текущее отклонение от данных БД"""
        # 1. Пересчитываем вероятности сети
        network = build_network(self.graph, self.current_probs)
        probs = calculate_probabilities(network)
        save_results(probs, network, self.graph)
        
        # 2. Сравниваем с БД
        results = self.compute_obj.calculate()
        
        # 3. Суммируем ошибки по всем препаратам
        total_loss = 0
        for drug_data in results.values():
            if drug_data['summ_loss'] is not None:
                total_loss += drug_data['summ_loss']
                
        return total_loss
    
    def _mutate_probabilities(self, node_id, mutation_strength=0.1):
        """Случайно изменяет вероятности для указанного узла"""
        if node_id not in self.current_probs:
            return
            
        # Для узлов без родителей - простое изменение
        if "" in self.current_probs[node_id]:
            current_val = self.current_probs[node_id][""]
            new_val = np.clip(current_val + random.uniform(-mutation_strength, mutation_strength), 0, 1)
            self.current_probs[node_id][""] = new_val
        else:
            # Для узлов с родителями - изменяем все значения с сохранением суммы
            keys = list(self.current_probs[node_id].keys())
            values = np.array(list(self.current_probs[node_id].values()))
            
            # Добавляем случайные изменения
            noise = np.random.uniform(-mutation_strength, mutation_strength, len(values))
            new_values = np.clip(values + noise, 0, 1)
            
            # Нормализуем чтобы сумма была 1
            new_values /= new_values.sum()
            
            for i, k in enumerate(keys):
                self.current_probs[node_id][k] = new_values[i]
    
    def optimize(self, iterations=100, mutation_strength=0.1):
        """Основной цикл оптимизации методом Монте-Карло"""
        # Определяем узлы для оптимизации (предки side_e узлов)
        optimization_nodes = set()
        for se_node in self.side_e_nodes:
            optimization_nodes.update(self._get_affected_nodes(se_node))
            
        for iter in range(iterations):
            # Сохраняем текущее состояние
            backup_probs = copy.deepcopy(self.current_probs)
            
            # Случайно выбираем узел для изменения
            node_id = random.choice(list(optimization_nodes))
            node_name = self.node_map[node_id]['name']
            
            # Вносим изменения
            self._mutate_probabilities(node_name, mutation_strength)
            
            # Рассчитываем новую ошибку
            current_loss = self.calculate_current_loss()
            
            # Если улучшение - сохраняем
            if current_loss < self.min_loss:
                self.min_loss = current_loss
                self.best_probs = copy.deepcopy(self.current_probs)
                print(f"Iteration {iter}: New best loss {current_loss:.4f}")
            else:
                # Возвращаем предыдущие значения
                self.current_probs = backup_probs
                
            # Постепенно уменьшаем размер мутаций
            mutation_strength *= 0.995
            
    def save_best_probabilities(self, filename='optimized_probabilities.json'):
        """Сохраняет лучшие найденные вероятности"""
        with open(filename, 'w') as f:
            json.dump(self.best_probs, f, indent=4)

if __name__ == "__main__":
    # Инициализация оптимизатора
    optimizer = ProbabilityOptimizer(
        graph_file='fozinopril_ramipril_Parse.json',
        db_path=DB_SIDE_E
    )
    
    # Запуск оптимизации
    optimizer.optimize(iterations=500)
    
    # Сохранение результатов
    optimizer.save_best_probabilities()
    
    print("Оптимизация завершена. Результаты сохранены в optimized_probabilities.json")