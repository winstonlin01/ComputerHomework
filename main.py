import json
import math
import random
import tkinter as tk
from tkinter import ttk, messagebox, font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from typing import List, Tuple, Dict, Any

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SimulatedAnnealing:
    """模拟退火算法类 - 结构优化"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initial_temp = config['algorithm']['initial_temperature']
        self.cooling_rate = config['algorithm']['cooling_rate']
        self.min_temp = config['algorithm']['min_temperature']
        self.max_iterations = config['algorithm']['max_iterations']
        self.iterations_per_temp = config['algorithm']['iterations_per_temp']
        
        self.current_solution = None
        self.best_solution = None
        self.current_cost = float('inf')
        self.best_cost = float('inf')
        self.temperature = self.initial_temp
        self.iteration = 0
        self.cost_history = []
        self.stress_history = []
        
    def initialize_solution(self, problem_type: str, num_nodes: int, search_space: Dict) -> Dict:
        """初始化解決方案 - 結構節點位置"""
        if problem_type == "structure":
            # 結構優化：生成節點位置
            solution = {}
            for i in range(num_nodes):
                solution[f'node_{i}'] = {
                    'x': random.uniform(search_space['x_min'], search_space['x_max']),
                    'y': random.uniform(search_space['y_min'], search_space['y_max']),
                    'type': 'free'  # 節點類型：free, fixed, load
                }
            return solution
        return {}
    
    def check_stability(self, solution: Dict, connections: List[Tuple[int, int, float]], 
                       fixed_nodes: List[int], load_nodes: List[Tuple[int, float, float]]) -> Tuple[bool, float]:
        """檢查結構穩定性，返回(是否穩定, 穩定性懲罰值)"""
        if len(solution) == 0:
            return False, float('inf')
        
        node_positions = [(solution[f'node_{i}']['x'], solution[f'node_{i}']['y']) 
                         for i in range(len(solution))]
        
        # 1. 檢查是否有足夠的固定節點（至少2個，防止剛體位移和旋轉）
        fixed_node_indices = []
        for i in range(len(solution)):
            node = solution[f'node_{i}']
            if node.get('type') == 'fixed':
                fixed_node_indices.append(i)
        
        if len(fixed_node_indices) < 2:
            return False, 10000.0  # 不穩定，大懲罰
        
        # 2. 檢查結構重心是否在支撐範圍內（防止傾倒）
        # 計算所有節點的重心（包括載荷）
        total_mass = 0.0
        center_x = 0.0
        center_y = 0.0
        
        for i in range(len(node_positions)):
            x, y = node_positions[i]
            # 節點質量（假設與連接數相關）
            node_mass = 1.0
            for conn in connections:
                if conn[0] == i or conn[1] == i:
                    node_mass += 0.5
            total_mass += node_mass
            center_x += x * node_mass
            center_y += y * node_mass
        
        # 添加載荷影響
        for load_node_idx, fx, fy in load_nodes:
            if load_node_idx < len(node_positions):
                load_mass = abs(fy) * 0.1  # 載荷等效質量
                x, y = node_positions[load_node_idx]
                total_mass += load_mass
                center_x += x * load_mass
                center_y += y * load_mass
        
        if total_mass > 0:
            center_x /= total_mass
            center_y /= total_mass
        else:
            return False, 10000.0
        
        # 計算固定節點的支撐範圍（最小和最大x座標）
        if fixed_node_indices:
            fixed_x_coords = [node_positions[i][0] for i in fixed_node_indices]
            support_min_x = min(fixed_x_coords)
            support_max_x = max(fixed_x_coords)
            support_width = support_max_x - support_min_x
            
            # 重心應該在支撐範圍內（允許一些餘量）
            margin = support_width * 0.1
            if center_x < support_min_x - margin or center_x > support_max_x + margin:
                # 重心超出支撐範圍，結構可能傾倒
                stability_penalty = abs(center_x - (support_min_x + support_max_x) / 2) * 100
                return False, stability_penalty
        
        # 3. 檢查載荷節點和固定節點之間是否有連接（關鍵檢查）
        load_node_indices = []
        for i in range(len(solution)):
            node = solution[f'node_{i}']
            if node.get('type') == 'load':
                load_node_indices.append(i)
        
        # 構建連接圖
        connection_set = set()
        connection_graph = {}  # 鄰接表
        for i, j, _ in connections:
            connection_set.add((i, j))
            connection_set.add((j, i))
            if i not in connection_graph:
                connection_graph[i] = []
            if j not in connection_graph:
                connection_graph[j] = []
            connection_graph[i].append(j)
            connection_graph[j].append(i)
        
        # 對每個載荷節點，檢查是否能同時到達至少兩個固定節點
        if load_node_indices and len(fixed_node_indices) >= 2:
            for load_idx in load_node_indices:
                # 使用BFS找出所有能到達的固定節點
                reachable_fixed_nodes = set()
                visited = set()
                queue = [load_idx]
                visited.add(load_idx)
                
                while queue:
                    current = queue.pop(0)
                    if current in fixed_node_indices:
                        reachable_fixed_nodes.add(current)
                    
                    # 檢查所有與當前節點連接的節點
                    if current in connection_graph:
                        for neighbor in connection_graph[current]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                
                # 載荷節點必須能同時連接到至少兩個固定節點
                if len(reachable_fixed_nodes) < 2:
                    return False, 20000.0  # 非常大的懲罰
        
        # 如果載荷節點和固定節點之間沒有連接，結構無效
        if load_node_indices:
            has_any_connection = False
            for load_idx in load_node_indices:
                if load_idx in connection_graph:
                    for neighbor in connection_graph[load_idx]:
                        if neighbor in fixed_node_indices:
                            has_any_connection = True
                            break
                    if has_any_connection:
                        break
            if not has_any_connection:
                return False, 20000.0  # 非常大的懲罰
        
        # 4. 檢查結構是否形成穩定的幾何形狀（至少有一些三角形）
        # 計算三角形數量（三個節點相互連接形成三角形）
        triangles = set()
        
        # 檢查所有可能的三元組是否形成三角形
        for i in range(len(node_positions)):
            for j in range(i + 1, len(node_positions)):
                for k in range(j + 1, len(node_positions)):
                    if (i, j) in connection_set and (j, k) in connection_set and (i, k) in connection_set:
                        triangle = tuple(sorted([i, j, k]))
                        triangles.add(triangle)
        
        # 如果沒有形成任何三角形且連接數較多，結構可能不穩定
        if len(triangles) == 0 and len(connections) > 3:
            return False, 5000.0
        
        # 5. 檢查結構的寬高比（過於細長的結構容易傾倒）
        if node_positions:
            x_coords = [pos[0] for pos in node_positions]
            y_coords = [pos[1] for pos in node_positions]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            if height > 0:
                aspect_ratio = width / height
                # 如果結構過於細長（寬高比太小），容易傾倒
                if aspect_ratio < 0.3:
                    return False, 3000.0 * (0.3 - aspect_ratio)
        
        return True, 0.0
    
    def solve_statics(self, solution: Dict, connections: List[Tuple[int, int, float]], 
                     fixed_nodes: List[int], load_nodes: List[Tuple[int, float, float]]) -> Tuple[np.ndarray, Dict]:
        """使用直接剛度法求解靜力學問題
        
        返回: (節點位移向量, 杆件內力字典)
        """
        num_nodes = len(solution)
        if num_nodes == 0:
            return None, {}
        
        node_positions = [(solution[f'node_{i}']['x'], solution[f'node_{i}']['y']) 
                         for i in range(num_nodes)]
        
        # 材料參數
        E = 200000.0  # 彈性模量 (MPa)
        A = 100.0     # 截面積 (mm²)
        
        # 構建全局剛度矩陣 (2*num_nodes x 2*num_nodes)
        # 每個節點有2個自由度 (x, y)
        K_global = np.zeros((2 * num_nodes, 2 * num_nodes))
        
        # 構建每個杆件的局部剛度矩陣並組裝到全局矩陣
        element_forces = {}  # 存儲杆件內力
        
        for i, j, length in connections:
            if length < 1e-6:  # 避免除零
                continue
            
            # 杆件方向向量
            dx = node_positions[j][0] - node_positions[i][0]
            dy = node_positions[j][1] - node_positions[i][1]
            cos_theta = dx / length
            sin_theta = dy / length
            
            # 杆件剛度 k = EA/L
            k = E * A / length
            
            # 局部剛度矩陣 (4x4 for 2D truss element)
            # [u_i, v_i, u_j, v_j]
            T = np.array([
                [cos_theta, sin_theta, 0, 0],
                [0, 0, cos_theta, sin_theta]
            ])
            
            k_local = k * np.array([
                [1, -1],
                [-1, 1]
            ])
            
            # 轉換到全局坐標系
            k_global_element = T.T @ k_local @ T
            
            # 組裝到全局剛度矩陣
            dof_i = [2*i, 2*i+1]  # 節點i的自由度
            dof_j = [2*j, 2*j+1]  # 節點j的自由度
            dofs = dof_i + dof_j
            
            for idx1, dof1 in enumerate(dofs):
                for idx2, dof2 in enumerate(dofs):
                    K_global[dof1, dof2] += k_global_element[idx1, idx2]
        
        # 構建載荷向量
        F = np.zeros(2 * num_nodes)
        for load_node_idx, fx, fy in load_nodes:
            if load_node_idx < num_nodes:
                F[2 * load_node_idx] = fx
                F[2 * load_node_idx + 1] = fy
        
        # 識別固定節點和載荷節點（約束）
        fixed_dofs = []
        for i in range(num_nodes):
            node = solution[f'node_{i}']
            if node.get('type') == 'fixed' or node.get('type') == 'load':
                # 固定節點和載荷節點都被約束（無法移動）
                fixed_dofs.append(2 * i)      # x方向固定
                fixed_dofs.append(2 * i + 1)  # y方向固定
        
        # 應用邊界條件：固定自由度
        # 方法：將固定自由度的對應行和列設為單位矩陣，載荷設為0
        K_reduced = K_global.copy()
        F_reduced = F.copy()
        
        for dof in fixed_dofs:
            K_reduced[dof, :] = 0
            K_reduced[:, dof] = 0
            K_reduced[dof, dof] = 1
            F_reduced[dof] = 0
        
        # 求解線性方程組 K * u = F
        try:
            # 檢查矩陣是否奇異
            if np.linalg.cond(K_reduced) > 1e12:
                return None, {}
            
            u = np.linalg.solve(K_reduced, F_reduced)
        except np.linalg.LinAlgError:
            # 矩陣奇異，結構不穩定
            return None, {}
        
        # 計算杆件內力
        for i, j, length in connections:
            if length < 1e-6:
                continue
            
            # 節點位移
            u_i = u[2*i]
            v_i = u[2*i+1]
            u_j = u[2*j]
            v_j = u[2*j+1]
            
            # 杆件方向
            dx = node_positions[j][0] - node_positions[i][0]
            dy = node_positions[j][1] - node_positions[i][1]
            cos_theta = dx / length
            sin_theta = dy / length
            
            # 局部位移
            u_local_i = u_i * cos_theta + v_i * sin_theta
            u_local_j = u_j * cos_theta + v_j * sin_theta
            
            # 杆件應變
            strain = (u_local_j - u_local_i) / length
            
            # 杆件內力 (N = EA * strain)
            force = E * A * strain
            
            # 杆件應力 (σ = N / A = E * strain)
            stress = E * strain
            
            element_forces[(i, j)] = {
                'force': force,
                'stress': abs(stress),
                'strain': strain,
                'length': length,
                'direction': (cos_theta, sin_theta)  # 保存方向用于力矩计算
            }
        
        # 檢查固定節點的力矩平衡（固定節點不應提供力矩）
        fixed_node_indices = []
        for i in range(num_nodes):
            node = solution[f'node_{i}']
            if node.get('type') == 'fixed':
                fixed_node_indices.append(i)
        
        for fixed_idx in fixed_node_indices:
            # 計算以固定節點為原點的力矩和
            total_moment = 0.0
            fixed_x, fixed_y = node_positions[fixed_idx]
            
            # 檢查所有連接到該固定節點的杆件
            for (i, j), force_data in element_forces.items():
                if i == fixed_idx or j == fixed_idx:
                    # 找到杆件的另一端
                    other_idx = j if i == fixed_idx else i
                    other_x, other_y = node_positions[other_idx]
                    
                    # 杆件內力（軸向力）
                    force = force_data['force']
                    cos_theta, sin_theta = force_data['direction']
                    
                    # 杆件內力在全局坐標系中的分量
                    force_x = force * cos_theta
                    force_y = force * sin_theta
                    
                    # 如果杆件從固定節點指向另一端，力方向需要反轉
                    if i == fixed_idx:
                        force_x = -force_x
                        force_y = -force_y
                    
                    # 計算力矩：M = r × F = (x - x0) * Fy - (y - y0) * Fx
                    r_x = other_x - fixed_x
                    r_y = other_y - fixed_y
                    moment = r_x * force_y - r_y * force_x
                    total_moment += moment
            
            # 檢查力矩平衡（允許小的數值誤差）
            if abs(total_moment) > 1e-3:  # 如果力矩不平衡超過閾值
                # 固定節點提供了力矩，這不符合鉸接支撐的假設
                return None, {}
        
        return u, element_forces
    
    def calculate_stress(self, solution: Dict, fixed_nodes: List[int], load_nodes: List[Tuple[int, float, float]], 
                        max_stress: float) -> Tuple[float, float]:
        """使用靜力學求解計算結構應力和成本"""
        # 刪除孤立節點
        solution = self.remove_isolated_nodes(solution, fixed_nodes, load_nodes)
        
        if len(solution) == 0:
            return float('inf'), float('inf')
        
        node_positions = [(solution[f'node_{i}']['x'], solution[f'node_{i}']['y']) 
                         for i in range(len(solution))]
        
        # 計算連接
        connections = []
        if len(node_positions) > 1:
            distances = []
            for i in range(len(node_positions)):
                for j in range(i + 1, len(node_positions)):
                    dx = node_positions[j][0] - node_positions[i][0]
                    dy = node_positions[j][1] - node_positions[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    distances.append((i, j, dist))
            
            if distances:
                sorted_distances = sorted(distances, key=lambda x: x[2])
                threshold = sorted_distances[len(sorted_distances) // 2][2] * 1.5
                
                for i, j, dist in distances:
                    if dist <= threshold:
                        connections.append((i, j, dist))
        
        # 檢查結構穩定性
        is_stable, stability_penalty = self.check_stability(solution, connections, fixed_nodes, load_nodes)
        if not is_stable:
            return float('inf'), float('inf')
        
        # 使用靜力學求解
        u, element_forces = self.solve_statics(solution, connections, fixed_nodes, load_nodes)
        
        if u is None or len(element_forces) == 0:
            # 求解失敗，結構不穩定
            return float('inf'), float('inf')
        
        # 計算最大應力和總重量
        max_element_stress = 0.0
        total_length = 0.0
        
        for (i, j), force_data in element_forces.items():
            stress = force_data['stress']
            length = force_data['length']
            max_element_stress = max(max_element_stress, stress)
            total_length += length
        
        # 成本 = 結構重量 + 應力懲罰
        density = 7.85e-6  # 鋼材密度 (kg/mm³)
        A = 100.0  # 截面積 (mm²)
        weight = total_length * A * density * 9.81  # 重量 (N)
        
        stress_penalty = 0.0
        if max_element_stress > max_stress:
            # 超過強度限制的懲罰
            stress_penalty = (max_element_stress - max_stress) * 10000
        
        cost = weight + stress_penalty
        
        return cost, max_element_stress
    
    def calculate_cost(self, solution: Dict, problem_type: str, 
                      fixed_nodes: List[int] = None, 
                      load_nodes: List[Tuple[int, float, float]] = None,
                      max_stress: float = 100.0) -> float:
        """计算解决方案的成本"""
        if problem_type == "structure":
            cost, _ = self.calculate_stress(solution, fixed_nodes or [], load_nodes or [], max_stress)
            return cost
        return 0.0
    
    def get_connected_nodes(self, solution: Dict) -> set:
        """獲取有連接的節點集合"""
        node_positions = [(solution[f'node_{i}']['x'], solution[f'node_{i}']['y']) 
                         for i in range(len(solution))]
        
        if len(node_positions) <= 1:
            return set(range(len(node_positions)))
        
        # 計算連接閾值
        distances = []
        for i in range(len(node_positions)):
            for j in range(i + 1, len(node_positions)):
                dx = node_positions[j][0] - node_positions[i][0]
                dy = node_positions[j][1] - node_positions[i][1]
                dist = math.sqrt(dx*dx + dy*dy)
                distances.append(dist)
        
        if not distances:
            return set(range(len(node_positions)))
        
        distances.sort()
        threshold = distances[len(distances) // 2] * 1.5
        
        # 找出所有有連接的節點
        connected_nodes = set()
        for i in range(len(node_positions)):
            for j in range(i + 1, len(node_positions)):
                dx = node_positions[j][0] - node_positions[i][0]
                dy = node_positions[j][1] - node_positions[i][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist <= threshold:
                    connected_nodes.add(i)
                    connected_nodes.add(j)
        
        return connected_nodes
    
    def get_nodes_connected_to_critical(self, solution: Dict) -> set:
        """獲取與固定節點或載荷節點有連接的節點集合"""
        node_positions = [(solution[f'node_{i}']['x'], solution[f'node_{i}']['y']) 
                         for i in range(len(solution))]
        
        if len(node_positions) <= 1:
            return set(range(len(node_positions)))
        
        # 找出固定節點和載荷節點的索引
        critical_node_indices = set()
        for i in range(len(solution)):
            node = solution[f'node_{i}']
            if node.get('type') == 'fixed' or node.get('type') == 'load':
                critical_node_indices.add(i)
        
        if not critical_node_indices:
            return set(range(len(node_positions)))
        
        # 計算連接閾值
        distances = []
        for i in range(len(node_positions)):
            for j in range(i + 1, len(node_positions)):
                dx = node_positions[j][0] - node_positions[i][0]
                dy = node_positions[j][1] - node_positions[i][1]
                dist = math.sqrt(dx*dx + dy*dy)
                distances.append(dist)
        
        if not distances:
            return set(range(len(node_positions)))
        
        distances.sort()
        threshold = distances[len(distances) // 2] * 1.5
        
        # 使用BFS找出所有與關鍵節點連通的節點
        connected_to_critical = set(critical_node_indices)
        queue = list(critical_node_indices)
        
        while queue:
            current = queue.pop(0)
            for i in range(len(node_positions)):
                if i not in connected_to_critical:
                    dx = node_positions[i][0] - node_positions[current][0]
                    dy = node_positions[i][1] - node_positions[current][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist <= threshold:
                        connected_to_critical.add(i)
                        queue.append(i)
        
        return connected_to_critical
    
    def remove_isolated_nodes(self, solution: Dict, fixed_nodes: List[int], 
                              load_nodes: List[Tuple[int, float, float]]) -> Dict:
        """刪除沒有連接到固定節點或載荷節點的節點"""
        if len(solution) == 0:
            return solution
        
        # 找出與固定節點或載荷節點有連接的節點
        nodes_connected_to_critical = self.get_nodes_connected_to_critical(solution)
        
        # 固定節點和載荷節點必須保留（即使沒有連接）
        critical_nodes = set()
        for i in range(len(solution)):
            node = solution[f'node_{i}']
            if node.get('type') == 'fixed' or node.get('type') == 'load':
                critical_nodes.add(i)
        
        # 要保留的節點：與關鍵節點連通的節點或關鍵節點本身
        nodes_to_keep = nodes_connected_to_critical | critical_nodes
        
        # 如果沒有節點要保留，返回原解
        if not nodes_to_keep:
            return solution
        
        # 創建新的解，只保留需要的節點
        new_solution = {}
        node_mapping = {}  # 舊節點索引 -> 新節點索引
        new_index = 0
        
        for i in range(len(solution)):
            if i in nodes_to_keep:
                new_solution[f'node_{new_index}'] = solution[f'node_{i}'].copy()
                node_mapping[i] = new_index
                new_index += 1
        
        return new_solution
    
    def generate_neighbor(self, solution: Dict) -> Dict:
        """生成鄰域解 - 微調節點位置（保持固定節點和載荷節點的相對位置）"""
        neighbor = {}
        perturbation = 2.0  # 擾動範圍
        
        for key, value in solution.items():
            node_type = value.get('type', 'free')
            
            # 固定節點和載荷節點的擾動應該更小，保持相對位置
            if node_type == 'fixed' or node_type == 'load':
                perturbation_factor = 0.1  # 固定節點和載荷節點擾動很小
            else:
                perturbation_factor = 1.0
            
            neighbor[key] = {
                'x': value['x'] + random.uniform(-perturbation * perturbation_factor, 
                                                perturbation * perturbation_factor),
                'y': value['y'] + random.uniform(-perturbation * perturbation_factor, 
                                                perturbation * perturbation_factor),
                'type': node_type  # 保留節點類型
            }
            # 限制在搜索空間內
            search_space = self.config['optimization']['search_space']
            neighbor[key]['x'] = max(search_space['x_min'], 
                                   min(search_space['x_max'], neighbor[key]['x']))
            neighbor[key]['y'] = max(search_space['y_min'], 
                                   min(search_space['y_max'], neighbor[key]['y']))
        
        return neighbor
    
    def accept_probability(self, current_cost: float, neighbor_cost: float, temperature: float) -> float:
        """计算接受概率"""
        if neighbor_cost < current_cost:
            return 1.0
        return math.exp(-(neighbor_cost - current_cost) / temperature)
    
    def step(self, fixed_nodes: List[int] = None, 
            load_nodes: List[Tuple[int, float, float]] = None,
            max_stress: float = 100.0) -> Dict[str, Any]:
        """執行一步優化"""
        if self.iteration >= self.max_iterations:
            return {"finished": True}
        
        for _ in range(self.iterations_per_temp):
            # 生成鄰域解
            neighbor = self.generate_neighbor(self.current_solution)
            
            # 刪除孤立節點
            neighbor = self.remove_isolated_nodes(neighbor, fixed_nodes or [], load_nodes or [])
            
            # 如果刪除後沒有節點，跳過
            if len(neighbor) == 0:
                continue
            
            neighbor_cost = self.calculate_cost(neighbor, "structure", fixed_nodes, load_nodes, max_stress)
            
            # 決定是否接受新解
            if random.random() < self.accept_probability(self.current_cost, neighbor_cost, self.temperature):
                self.current_solution = neighbor
                self.current_cost = neighbor_cost
                
                # 更新最優解
                if neighbor_cost < self.best_cost:
                    self.best_solution = neighbor.copy()
                    self.best_cost = neighbor_cost
        
        # 刪除最優解中的孤立節點
        self.best_solution = self.remove_isolated_nodes(
            self.best_solution, fixed_nodes or [], load_nodes or []
        )
        
        # 計算當前最優解的應力
        _, max_stress_value = self.calculate_stress(
            self.best_solution, fixed_nodes or [], load_nodes or [], max_stress
        )
        self.cost_history.append(self.best_cost)
        self.stress_history.append(max_stress_value)
        self.iteration += 1
        
        # 降低溫度
        if self.iteration % self.iterations_per_temp == 0:
            self.temperature *= self.cooling_rate
        
        return {
            "finished": self.temperature < self.min_temp or self.iteration >= self.max_iterations,
            "current_solution": self.current_solution.copy(),
            "best_solution": self.best_solution.copy(),
            "current_cost": self.current_cost,
            "best_cost": self.best_cost,
            "temperature": self.temperature,
            "iteration": self.iteration
        }


class OptimizationGUI:
    """优化程序图形界面"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("結構優化程式 - 模擬退火演算法")
        
        # 设置深色主题
        self.setup_dark_theme()
        
        # 加载配置
        self.config = self.load_config()
        
        # 设置窗口大小
        window_size = self.config['display']['window_size']
        self.root.geometry(f"{window_size['width']}x{window_size['height']}")
        
        # 初始化算法
        self.algorithm = None
        self.fixed_nodes = []
        self.load_nodes = []
        self.max_stress = 100.0
        self.is_running = False
        self.update_interval = self.config['display']['update_interval']
        
        # 字体大小相关
        self.base_font_size = 10
        self.base_window_width = 1200
        self.base_window_height = 800
        
        # 绑定窗口大小变化事件
        self.root.bind('<Configure>', self.on_window_resize)
        
        # 绑定窗口关闭事件，确保正确清理资源
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 创建界面
        self.create_widgets()
        
        # 初始化问题
        self.initialize_problem()
    
    def on_closing(self):
        """窗口关闭时的清理工作"""
        if self.is_running:
            self.stop_optimization()
        # 清理matplotlib资源
        plt.close('all')
        # 销毁窗口
        self.root.destroy()
    
    def setup_dark_theme(self):
        """设置深色主题"""
        # 设置matplotlib深色主题
        plt.style.use('dark_background')
        
        # 设置tkinter深色主题
        self.root.configure(bg='#2b2b2b')
        
        # 创建深色主题样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置深色颜色
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabelFrame', background='#2b2b2b', foreground='#ffffff', 
                       bordercolor='#555555', borderwidth=2)
        style.configure('TLabelFrame.Label', background='#2b2b2b', foreground='#ffffff')
        style.configure('TButton', background='#404040', foreground='#ffffff', 
                       borderwidth=1, focuscolor='none')
        style.map('TButton', background=[('active', '#505050')])
        style.configure('TText', background='#1e1e1e', foreground='#ffffff', 
                       insertbackground='#ffffff', selectbackground='#404040')
        style.configure('TScrollbar', background='#404040', troughcolor='#2b2b2b', 
                       borderwidth=0, arrowcolor='#ffffff', darkcolor='#404040', 
                       lightcolor='#404040')
        
    def load_config(self) -> Dict[str, Any]:
        """从JSON文件加载配置"""
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            messagebox.showerror("錯誤", "找不到配置檔案 config.json")
            return self.get_default_config()
        except json.JSONDecodeError:
            messagebox.showerror("錯誤", "配置檔案格式錯誤")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "algorithm": {
                "initial_temperature": 1000.0,
                "cooling_rate": 0.95,
                "min_temperature": 0.01,
                "max_iterations": 1000,
                "iterations_per_temp": 10
            },
            "optimization": {
                "problem_type": "structure",
                "num_nodes": 8,
                "max_stress": 100.0,
                "search_space": {
                    "x_min": 0,
                    "x_max": 100,
                    "y_min": 0,
                    "y_max": 100
                }
            },
            "display": {
                "update_interval": 10,
                "show_animation": True,
                "window_size": {
                    "width": 1200,
                    "height": 800
                }
            }
        }
    
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        control_frame.columnconfigure(0, weight=1)
        control_frame.rowconfigure(0, weight=1)
        
        # 按鈕字體（使用變量以便後續調整）
        self.button_font = font.Font(family='Microsoft YaHei', size=self.base_font_size)
        
        # 按鈕
        self.start_button = ttk.Button(control_frame, text="開始優化", command=self.start_optimization)
        self.start_button.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.stop_button = ttk.Button(control_frame, text="停止", command=self.stop_optimization, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.reset_button = ttk.Button(control_frame, text="重置", command=self.reset_problem)
        self.reset_button.grid(row=0, column=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 資訊顯示
        info_frame = ttk.LabelFrame(main_frame, text="優化資訊", padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        # 设置字体（使用变量以便后续调整）
        self.info_font = font.Font(family='Microsoft YaHei', size=self.base_font_size)
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=self.base_font_size)
        
        self.info_text = tk.Text(info_frame, height=10, width=35, font=self.info_font,
                                 bg='#1e1e1e', fg='#ffffff', insertbackground='#ffffff',
                                 selectbackground='#404040', selectforeground='#ffffff')
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        # 顏色說明（左下角）
        color_legend_frame = ttk.LabelFrame(main_frame, text="節點顏色說明", padding="5")
        color_legend_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # 創建顏色標籤（使用實例變量以便後續調整字體大小）
        self.color_info_font = font.Font(family='Microsoft YaHei', size=max(8, int(self.base_font_size * 0.85)))
        self.color_dot_font_size = 16
        
        self.red_label = tk.Label(color_legend_frame, text="●", fg='red', bg='#2b2b2b', font=(None, self.color_dot_font_size))
        self.red_label.grid(row=0, column=0, padx=5, sticky=tk.W)
        self.red_text = tk.Label(color_legend_frame, text="固定節點（約束）", fg='white', bg='#2b2b2b', 
                font=self.color_info_font)
        self.red_text.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        self.yellow_label = tk.Label(color_legend_frame, text="●", fg='yellow', bg='#2b2b2b', font=(None, self.color_dot_font_size))
        self.yellow_label.grid(row=1, column=0, padx=5, sticky=tk.W)
        self.yellow_text = tk.Label(color_legend_frame, text="載荷節點", fg='white', bg='#2b2b2b', 
                font=self.color_info_font)
        self.yellow_text.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        self.blue_label = tk.Label(color_legend_frame, text="●", fg='lightblue', bg='#2b2b2b', font=(None, self.color_dot_font_size))
        self.blue_label.grid(row=2, column=0, padx=5, sticky=tk.W)
        self.blue_text = tk.Label(color_legend_frame, text="自由節點", fg='white', bg='#2b2b2b', 
                font=self.color_info_font)
        self.blue_text.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # 圖形顯示區域
        plot_frame = ttk.LabelFrame(main_frame, text="視覺化", padding="10")
        plot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # 创建matplotlib图形 - 使用深色主题
        self.fig = plt.figure(figsize=(10, 8), facecolor='#1e1e1e')
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)
        
        # 设置子图背景色
        self.ax1.set_facecolor('#1e1e1e')
        self.ax2.set_facecolor('#1e1e1e')
        
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置網格權重，使所有面板都能隨窗口調整
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)  # 資訊面板列
        main_frame.columnconfigure(1, weight=3)  # 圖形面板列（權重更大）
        main_frame.rowconfigure(0, weight=1)     # 第一行（控制面板）
        main_frame.rowconfigure(1, weight=3)     # 第二行（資訊和圖形面板，權重更大）
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
    
    def calculate_font_size(self, width: int, height: int) -> int:
        """根據窗口大小計算字體大小"""
        # 基於窗口寬度和高度的平均值來計算字體大小
        avg_size = (width + height) / 2
        base_avg = (self.base_window_width + self.base_window_height) / 2
        scale_factor = avg_size / base_avg
        # 字體大小在8到20之間
        new_font_size = max(8, min(20, int(self.base_font_size * scale_factor)))
        return new_font_size
    
    def update_font_sizes(self, font_size: int):
        """更新所有字體大小"""
        # 更新資訊文字字體
        if hasattr(self, 'info_font'):
            self.info_font.configure(size=font_size)
        if hasattr(self, 'info_text'):
            self.info_text.configure(font=self.info_font)
        
        # 更新按鈕字體
        if hasattr(self, 'button_font'):
            self.button_font.configure(size=font_size)
        # 注意：ttk.Button 不直接支持 font 参数，需要通过样式设置
        style = ttk.Style()
        style.configure('TButton', font=self.button_font)
        
        # 更新顏色說明字體
        if hasattr(self, 'color_info_font'):
            self.color_info_font.configure(size=max(8, int(font_size * 0.85)))
            if hasattr(self, 'red_text'):
                self.red_text.configure(font=self.color_info_font)
                self.yellow_text.configure(font=self.color_info_font)
                self.blue_text.configure(font=self.color_info_font)
        
        # 更新顏色點的大小
        if hasattr(self, 'red_label'):
            dot_size = max(12, int(font_size * 1.2))
            self.color_dot_font_size = dot_size
            self.red_label.configure(font=(None, dot_size))
            self.yellow_label.configure(font=(None, dot_size))
            self.blue_label.configure(font=(None, dot_size))
    
    def on_window_resize(self, event):
        """窗口大小變化時的回調"""
        if event.widget == self.root:
            # 更新圖形大小
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            if width > 100 and height > 100:  # 避免初始化時的無效尺寸
                # 計算新的圖形尺寸（考慮邊距）
                fig_width = max(6, (width - 400) / 100)
                fig_height = max(4, (height - 200) / 100)
                self.fig.set_size_inches(fig_width, fig_height)
                
                # 計算並更新字體大小
                new_font_size = self.calculate_font_size(width, height)
                self.update_font_sizes(new_font_size)
                
                # 如果正在顯示，更新顯示以應用新的字體大小
                if hasattr(self, 'algorithm') and self.algorithm:
                    self.update_display()
                else:
                    self.canvas.draw()
        
    def initialize_problem(self):
        """初始化优化问题"""
        problem_config = self.config['optimization']
        num_nodes = problem_config['num_nodes']
        search_space = problem_config['search_space']
        self.max_stress = problem_config.get('max_stress', 100.0)
        
        # 设置固定节点（底部两个节点）
        self.fixed_nodes = [0, num_nodes // 2] if num_nodes >= 2 else [0]
        
        # 设置载荷节点（顶部节点）
        load_magnitude = 10.0
        self.load_nodes = [(num_nodes - 1, 0.0, -load_magnitude)]
        
        # 初始化算法
        self.algorithm = SimulatedAnnealing(self.config)
        self.algorithm.current_solution = self.algorithm.initialize_solution(
            problem_config['problem_type'],
            num_nodes,
            search_space
        )
        
        # 設置固定節點和載荷節點的類型和位置
        self._fix_critical_nodes_position()
        
        self.algorithm.best_solution = self.algorithm.current_solution.copy()
        self.algorithm.current_cost = self.algorithm.calculate_cost(
            self.algorithm.current_solution,
            problem_config['problem_type'],
            self.fixed_nodes,
            self.load_nodes,
            self.max_stress
        )
        self.algorithm.best_cost = self.algorithm.current_cost
        
        # 更新显示
        self.update_display()
    
    def _fix_critical_nodes_position(self):
        """固定關鍵節點（固定節點和載荷節點）的相對位置和類型"""
        solution = self.algorithm.current_solution
        search_space = self.config['optimization']['search_space']
        
        # 固定節點位置（底部，左右兩側）
        if len(self.fixed_nodes) >= 2:
            # 左側固定節點
            solution[f'node_{self.fixed_nodes[0]}'] = {
                'x': search_space['x_min'] + (search_space['x_max'] - search_space['x_min']) * 0.2,
                'y': search_space['y_min'] + (search_space['y_max'] - search_space['y_min']) * 0.1,
                'type': 'fixed'
            }
            # 右側固定節點
            solution[f'node_{self.fixed_nodes[1]}'] = {
                'x': search_space['x_min'] + (search_space['x_max'] - search_space['x_min']) * 0.8,
                'y': search_space['y_min'] + (search_space['y_max'] - search_space['y_min']) * 0.1,
                'type': 'fixed'
            }
        elif len(self.fixed_nodes) == 1:
            solution[f'node_{self.fixed_nodes[0]}'] = {
                'x': search_space['x_min'] + (search_space['x_max'] - search_space['x_min']) * 0.5,
                'y': search_space['y_min'] + (search_space['y_max'] - search_space['y_min']) * 0.1,
                'type': 'fixed'
            }
        
        # 載荷節點位置（頂部中央）
        for load_node_idx, _, _ in self.load_nodes:
            solution[f'node_{load_node_idx}'] = {
                'x': search_space['x_min'] + (search_space['x_max'] - search_space['x_min']) * 0.5,
                'y': search_space['y_min'] + (search_space['y_max'] - search_space['y_min']) * 0.9,
                'type': 'load'
            }
        
    def start_optimization(self):
        """开始优化"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.DISABLED)
        
        self.run_optimization_step()
    
    def stop_optimization(self):
        """停止优化"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.NORMAL)
    
    def reset_problem(self):
        """重置问题"""
        self.stop_optimization()
        self.initialize_problem()
    
    def run_optimization_step(self):
        """执行优化步骤"""
        if not self.is_running:
            return
        
        # 执行算法步骤
        result = self.algorithm.step(self.fixed_nodes, self.load_nodes, self.max_stress)
        
        # 更新显示
        if self.algorithm.iteration % self.update_interval == 0:
            self.update_display()
        
        # 檢查是否完成
        if result.get("finished", False):
            self.stop_optimization()
            messagebox.showinfo("完成", "優化完成！")
        else:
            # 继续优化
            self.root.after(1, self.run_optimization_step)
    
    def update_display(self):
        """更新顯示"""
        # 計算字體大小（基於窗口大小）
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        if width > 100 and height > 100:
            font_size = self.calculate_font_size(width, height)
            node_label_fontsize = max(6, int(font_size * 0.8))
            title_fontsize = max(9, int(font_size * 1.1))
            label_fontsize = max(8, int(font_size * 0.9))
            legend_fontsize = max(7, int(font_size * 0.85))
        else:
            node_label_fontsize = 8
            title_fontsize = 11
            label_fontsize = 10
            legend_fontsize = 9
        
        # 更新資訊文字
        self.info_text.delete(1.0, tk.END)
        max_stress_value = self.algorithm.stress_history[-1] if self.algorithm.stress_history else 0.0
        info = f"迭代次數: {self.algorithm.iteration}\n"
        info += f"當前溫度: {self.algorithm.temperature:.2f}\n"
        info += f"當前成本: {self.algorithm.current_cost:.2f}\n"
        info += f"最優成本: {self.algorithm.best_cost:.2f}\n"
        info += f"最大應力: {max_stress_value:.2f} MPa\n"
        info += f"強度限制: {self.max_stress:.2f} MPa\n"
        info += f"溫度比率: {self.algorithm.temperature/self.algorithm.initial_temp*100:.1f}%\n"
        info += f"優化進度: {self.algorithm.iteration/self.algorithm.max_iterations*100:.1f}%"
        self.info_text.insert(1.0, info)
        
        # 更新圖形
        self.ax1.clear()
        self.ax2.clear()
        
        # 設置深色主題顏色
        self.ax1.set_facecolor('#1e1e1e')
        self.ax2.set_facecolor('#1e1e1e')
        
        # 繪製結構
        if self.algorithm.best_solution:
            solution = self.algorithm.best_solution
            node_positions = []
            for i in range(len(solution)):
                node = solution[f'node_{i}']
                node_positions.append((node['x'], node['y']))
            
            # 繪製連接線（非全連接結構 - 基於距離閾值）
            # 計算連接閾值（基於平均節點間距）
            if len(node_positions) > 1:
                # 計算所有節點對之間的距離
                distances = []
                for i in range(len(node_positions)):
                    for j in range(i + 1, len(node_positions)):
                        dx = node_positions[j][0] - node_positions[i][0]
                        dy = node_positions[j][1] - node_positions[i][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        distances.append(dist)
                
                # 使用中位數距離作為閾值，只連接距離較近的節點
                if distances:
                    distances.sort()
                    threshold = distances[len(distances) // 2] * 1.5  # 中位數的1.5倍作為閾值
                    
                    # 只繪製距離小於閾值的連接
                    for i in range(len(node_positions)):
                        for j in range(i + 1, len(node_positions)):
                            dx = node_positions[j][0] - node_positions[i][0]
                            dy = node_positions[j][1] - node_positions[i][1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            
                            if dist <= threshold:
                                x_coords = [node_positions[i][0], node_positions[j][0]]
                                y_coords = [node_positions[i][1], node_positions[j][1]]
                                self.ax1.plot(x_coords, y_coords, 'cyan', linewidth=1.5, alpha=0.5, zorder=1)
            
            # 繪製節點（根據節點類型判斷顏色）
            node_colors = []
            node_sizes = []
            for i in range(len(node_positions)):
                node = solution[f'node_{i}']
                node_type = node.get('type', 'free')
                
                if node_type == 'fixed':
                    node_colors.append('red')  # 固定節點
                    node_sizes.append(100)
                elif node_type == 'load':
                    node_colors.append('yellow')  # 載荷節點
                    node_sizes.append(100)
                else:
                    node_colors.append('lightblue')  # 普通節點
                    node_sizes.append(80)
            
            for i, (x, y) in enumerate(node_positions):
                self.ax1.scatter(x, y, c=node_colors[i], s=node_sizes[i], 
                               zorder=5, edgecolors='white', linewidths=1.5)
                self.ax1.annotate(f'N{i}', (x, y), xytext=(5, 5), 
                                textcoords='offset points', color='white', fontsize=node_label_fontsize)
            
            # 圖例已移至左下角，不再在圖表中顯示
            
            self.ax1.set_title(f'最優結構 (成本: {self.algorithm.best_cost:.2f}, 最大應力: {max_stress_value:.2f} MPa)', 
                             color='white', fontsize=title_fontsize)
            self.ax1.set_xlabel('X座標 (m)', color='white', fontsize=label_fontsize)
            self.ax1.set_ylabel('Y座標 (m)', color='white', fontsize=label_fontsize)
            self.ax1.tick_params(colors='white', labelsize=max(7, int(label_fontsize * 0.85)))
            self.ax1.grid(True, alpha=0.2, color='gray')
            self.ax1.set_aspect('equal', adjustable='box')
        
        # 繪製成本歷史
        if len(self.algorithm.cost_history) > 0:
            self.ax2.plot(self.algorithm.cost_history, 'lime', linewidth=1.5, label='成本')
            if len(self.algorithm.stress_history) > 0:
                # 歸一化應力以便在同一圖中顯示
                stress_normalized = [s / max(self.algorithm.stress_history) * max(self.algorithm.cost_history) 
                                   if max(self.algorithm.stress_history) > 0 else 0 
                                   for s in self.algorithm.stress_history]
                self.ax2.plot(stress_normalized, 'orange', linewidth=1.5, alpha=0.7, label='應力(歸一化)')
            self.ax2.set_title('優化歷史', color='white', fontsize=title_fontsize)
            self.ax2.set_xlabel('迭代次數', color='white', fontsize=label_fontsize)
            self.ax2.set_ylabel('成本 / 應力', color='white', fontsize=label_fontsize)
            self.ax2.tick_params(colors='white', labelsize=max(7, int(label_fontsize * 0.85)))
            self.ax2.grid(True, alpha=0.2, color='gray')
            self.ax2.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='white', 
                          labelcolor='white', fontsize=legend_fontsize)
        
        self.canvas.draw()


def main():
    """主函数"""
    root = tk.Tk()
    app = OptimizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

