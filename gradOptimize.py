# optimizer.py
import json
import copy
import random
import numpy as np
from coolBNet import load_graph, create_base_probabilities, load_combined_data, build_network, calculate_probabilities, save_results


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
            
        self.side_effect_dataset = self._get_side_effect_freq_dataset()
        
        # Собираем информацию о графе
        self._prepare_graph_info()

     def _get_side_effect_freq_dataset(self):
        """
        Получает датасет побочных эффектов для каждого ЛС из БД.
        Возвращает словарь: ключ - название препарата (нижний регистр), значение - список побочных эффектов.
        """
        side_effect_dict = {}
        drugs = set([t[1] for t in self.drug_list])
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            # Подготавливаем список плейсхолдеров для оператора IN
            placeholders = ','.join('?' for _ in drugs)
            query = f"""
                SELECT d.drug, se.effect, dse.frequency
                FROM drug_side_effects dse
                JOIN drugs d ON dse.drug_id = d.id
                JOIN side_effects se ON dse.side_effect_id = se.id
                WHERE d.drug IN ({placeholders})
            """
            cur.execute(query, tuple(drugs))
            for drug, effect, frequency in cur.fetchall():
                drug_lower = drug.lower()
                if drug_lower not in side_effect_dict:
                    side_effect_dict[drug_lower] = {}
                side_effect_dict[drug_lower][effect] = frequency
        except Exception as e:
            logger.error("Ошибка при выполнении SQL-запроса: %s", e)
        finally:
            if conn:
                conn.close()

        missing = drugs - set(side_effect_dict.keys())
        if missing:
            logger.error("Датасеты не найдены для препаратов: %s", ", ".join(missing))
        return side_effect_dict
        # name --- prob

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
        graph_file='merged_graph_fozinopril_ramipril_Parse.json',
        db_path=DB_SIDE_E
    )
    
    # Запуск оптимизации
    optimizer.optimize(iterations=500)
    
    # Сохранение результатов
    optimizer.save_best_probabilities()
    
    print("Оптимизация завершена. Результаты сохранены в optimized_probabilities.json")