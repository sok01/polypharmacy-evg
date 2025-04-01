import json
import random
import copy
import numpy as np
from tqdm import tqdm  # Для визуализации прогресса
from coolBNet import BayesianNode, build_network, calculate_probabilities, save_results, load_graph
from dbCode import ComputeObjectF, CONVERT_RATE

class BayesianOptimizer:
    def __init__(self, graph_file, db_path, prob_file='probabilities.json'):
        """
        Инициализация оптимизатора
        :param graph_file: путь к файлу графа (JSON)
        :param db_path: путь к базе данных с реальными частотами
        :param prob_file: путь к файлу с вероятностями для оптимизации
        """
        # Загрузка данных
        self.graph = load_graph(graph_file)
        self.db_path = db_path
        self.prob_file = prob_file
        
        # Загрузка начальных вероятностей
        with open(prob_file, 'r') as f:
            self.current_probs = json.load(f)
            
        # Параметры оптимизации
        self.learning_rate = 0.01  # Скорость обучения
        self.max_iters = 100       # Макс. число итераций
        self.tolerance = 1e-4      # Минимальное изменение для остановки
        self.best_loss = float('inf')
        self.best_probs = None
        
    def _evaluate_current_loss(self):
        """Вычисление текущих потерь с использованием ComputeObjectF"""
        # 1. Сохраняем текущие вероятности
        with open(self.prob_file, 'w') as f:
            json.dump(self.current_probs, f)
            
        # 2. Пересчитываем сеть и получаем результаты
        network = build_network(self.graph, self.current_probs)
        final_probs = calculate_probabilities(network)
        save_results(final_probs, network, self.graph, 'temp_results.json')
        
        # 3. Загружаем временные результаты для вычисления потерь
        with open('temp_results.json') as f:
            results_data = {'tables': json.load(f)}
            
        # 4. Создаем объект для расчета потерь
        compute_obj = ComputeObjectF(
            results_data, 
            self.graph, 
            self.db_path
        )
        
        # 5. Вычисляем общую ошибку по всем препаратам
        loss_data = compute_obj.calculate()
        total_loss = 0
        for drug, data in loss_data.items():
            if data['summ_loss'] is not None:
                total_loss += data['summ_loss']
                
        return total_loss, loss_data
    
    def _get_side_e_nodes(self):
        """Возвращает список узлов типа side_e и их родителей"""
        side_e_nodes = []
        parent_map = defaultdict(list)
        for link in self.graph['links']:
            parent_map[link['target']].append(link['source'])
            
        for node in self.graph['nodes']:
            if node.get('label') == 'side_e':
                parents = parent_map.get(node['id'], [])
                side_e_nodes.append({
                    'name': node['name'],
                    'parents': parents
                })
        return side_e_nodes
    
    def _adjust_probabilities(self, loss_data):
        """
        Корректировка вероятностей на основе данных о потерях
        :param loss_data: результаты вычисления потерь от ComputeObjectF
        """
        # 1. Находим все side_e узлы
        side_e_nodes = self._get_side_e_nodes()
        
        # 2. Для каждого side_e узла вычисляем градиент
        for node_info in side_e_nodes:
            node_name = node_info['name']
            parents = node_info['parents']
            
            # 3. Получаем текущие вероятности для узла
            current_probs = self.current_probs.get(node_name, {})
            
            # 4. Для каждой комбинации родителей корректируем вероятность
            for comb in current_probs.keys():
                # Расчет градиента (упрощенный метод)
                grad = 0
                
                # Собираем вклады в ошибку для данного побочного эффекта
                for drug_data in loss_data.values():
                    for effect in drug_data['finded']:
                        effect_name, real_freq, pred_freq = effect
                        if effect_name == node_name:
                            # Разница между реальной и предсказанной вероятностью
                            error = (CONVERT_RATE.get(real_freq, 0) - pred_freq)
                            grad += 2 * error * (-1)  # Производная MSE
                            
                # Применяем обновление с учетом скорости обучения
                delta = -self.learning_rate * grad
                new_prob = current_probs[comb] + delta
                
                # Ограничиваем вероятность в диапазоне [0, 1]
                new_prob = max(0.0, min(1.0, new_prob))
                current_probs[comb] = new_prob
                
            # Обновляем вероятности в текущей конфигурации
            self.current_probs[node_name] = current_probs
            
    def optimize(self):
        """Основной цикл оптимизации"""
        prev_loss = float('inf')
        progress_bar = tqdm(range(self.max_iters))
        
        for iter in progress_bar:
            # 1. Оценка текущих потерь
            current_loss, loss_data = self._evaluate_current_loss()
            
            # 2. Проверка критериев остановки
            if abs(prev_loss - current_loss) < self.tolerance:
                print(f"Оптимизация остановлена на итерации {iter}: изменение потерь < tolerance")
                break
                
            # 3. Сохранение лучшей конфигурации
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_probs = copy.deepcopy(self.current_probs)
                
            # 4. Корректировка вероятностей
            self._adjust_probabilities(loss_data)
            
            # 5. Обновление прогресса
            progress_bar.set_description(f"Loss: {current_loss:.4f}, Best: {self.best_loss:.4f}")
            prev_loss = current_loss
            
        # Восстанавливаем лучшую конфигурацию
        self.current_probs = self.best_probs
        with open(self.prob_file, 'w') as f:
            json.dump(self.current_probs, f, indent=4)
            
        print(f"Оптимизация завершена. Лучшие потери: {self.best_loss:.4f}")

if __name__ == "__main__":
    # Инициализация оптимизатора
    optimizer = BayesianOptimizer(
        graph_file='drug_allopurinol_Parse.json',
        db_path='orlov_side_effects_dataset.db'
    )
    
    # Запуск оптимизации
    optimizer.optimize()