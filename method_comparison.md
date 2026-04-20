# V1 vs V2 辅助方法对比分析

## 1. 可解性检查方法

### 1.1 `_is_trapped` (V1) vs `is_trapped` (V2)

**V1 实现** (lines 308-324):
- 检查8个方向的移动空间（4个正交 + 4个对角）
- 使用固定偏移量 (0.05, 0.035)
- 使用 `_check_tblock_obstacle_collision` 进行碰撞检测
- 阈值：6个方向被阻挡则认为被困

**V2 实现** (lines 153-194):
- 先检查初始位置是否与障碍物碰撞（使用 `analytic_obs_collision_check`）
- 然后检查8个方向的包围情况
- 使用归一化方向向量 + 固定距离 (0.08)
- 阈值：6个方向被阻挡则认为被困

**评价**：
- ✅ **V2 更好**：先检查初始碰撞更合理，使用归一化方向向量更规范
- ⚠️ V1 的固定偏移量可能不够精确

---

### 1.2 `_has_path` (V1) vs `has_path` (V2)

**V1 实现** (lines 326-354):
- 在起点和终点之间采样10个点
- 对每个点检查碰撞，如果碰撞则尝试4个方向的偏移（0.05, 0.08, 0.10）
- 如果所有偏移都失败，认为路径不存在

**V2 实现** (lines 215-260):
- 先检查直线路径是否畅通 (`is_straight_path_clear`)
- 如果不通，检查左右两侧的绕行空间 (`check_side_clearance`)
- 如果障碍物≤3个，默认连通

**评价**：
- ✅ **V1 更好**：更系统地尝试偏移，覆盖更多情况
- ⚠️ V2 的"障碍物≤3个默认连通"假设可能不准确

---

### 1.3 `_is_path_blocked` (V1) vs `is_straight_path_clear` (V2)

**V1 实现** (lines 661-680):
- 采样20个点
- 检查每个点到障碍物的距离：`dist < (obstacle_size/2 + tblock_width/2 + 0.01)`
- 考虑T-block和障碍物的实际尺寸

**V2 实现** (lines 262-288):
- 采样点数根据路径长度动态计算：`int(path_length / 0.02) + 1`
- 检查距离：`dist < 0.07`（固定值）
- 使用 `tblock_radius + obstacle_radius` 的概念

**评价**：
- ✅ **V1 更好**：考虑实际尺寸，更精确
- ⚠️ V2 的固定阈值0.07可能不够准确

---

## 2. 路径计算相关

### 2.1 `_compute_shortest_path_length` (V1) vs `estimate_detour_ratio` (V2)

**V1 实现** (lines 356-385):
- 如果直线路径畅通，返回直线距离
- 如果被阻挡，计算绕过每个障碍物的路径长度（起点->障碍物->终点）
- 取最小绕行距离，如果失败则返回 `direct_length * 1.5`

**V2 实现** (lines 402-428):
- 在路径上采样10个点
- 统计被阻挡的采样点数量
- 根据阻挡比例估算绕行比例：`1.0 + block_ratio * 1.5`

**评价**：
- ✅ **V1 更好**：基于几何距离计算，更准确
- ⚠️ V2 的统计方法可能不够精确

---

### 2.2 `_sample_path_points` (V1 独有)

**V1 实现** (lines 446-547):
- 如果直线路径畅通，使用直线路径采样
- 如果被阻挡，找到最近的阻挡障碍物，计算绕行点
- 生成两段路径：起点->绕行点->终点

**评价**：
- ✅ **V1 独有功能**：更精细的路径采样，考虑实际绕行
- 这个功能在 `_compute_rotation_space` 中被使用，使得旋转空间计算更准确

---

## 3. 空间约束相关

### 3.1 `_compute_min_clearance` (V1) vs `compute_min_clearance` (V2)

**V1 实现** (lines 387-411):
- 沿直线路径采样20个点
- 计算每个点到障碍物的距离，减去 `obstacle_size/2 + tblock_width/2`
- 返回最小间隙，如果无穷大则返回0.2

**V2 实现** (lines 451-476):
- 沿直线路径采样20个点
- 计算距离，减去 `tblock_radius + obstacle_radius`
- 返回最小间隙（可能为负）

**评价**：
- ✅ **V1 更好**：有默认值保护，避免返回负值或无穷大
- ⚠️ V2 可能返回负值，需要调用者处理

---

### 3.2 `_compute_rotation_space` (V1) vs `compute_rotation_space` (V2)

**V1 实现** (lines 413-444):
- 使用 `_sample_path_points` 获取实际路径上的采样点（考虑绕行）
- 对每个路径点测试16个角度
- 计算每个点的旋转空间 = 1 - 碰撞比例
- 返回所有路径点的最小旋转空间（最保守）

**V2 实现** (lines 501-519):
- 只考虑单个位置（起点或终点）
- 找到最近的障碍物距离
- 线性映射到0-1：`(min_dist - 0.07) / (0.15 - 0.07)`

**评价**：
- ✅ **V1 明显更好**：
  - 考虑整个路径，而不仅仅是起点/终点
  - 使用实际路径采样（考虑绕行）
  - 测试多个角度，更准确
- ⚠️ V2 只考虑单个点，不够全面

---

## 4. 特征提取和相似度

### 4.1 `_extract_features` (V1) vs `extract_features` (V2)

**V1 实现** (lines 549-570):
- 提取：`num_obstacles`, `centroid`, `spread`, `symmetry`, `clustering`
- `spread` 使用 `np.std(positions, axis=0)`（每个维度的标准差）

**V2 实现** (lines 568-602):
- 提取：`num_obstacles`, `centroid`, `spread`, `symmetry`, `clustering`
- `spread` 同样使用 `np.std(positions, axis=0)`

**评价**：
- ✅ **基本相同**，都合理

---

### 4.2 `_compute_similarity` (V1) vs `compute_similarity` (V2)

**V1 实现** (lines 572-595):
- 使用4个维度：`spatial_sim`, `struct_sim`, `spread_sim`, `clustering_sim`
- 平均：`(spatial_sim + struct_sim + spread_sim + clustering_sim) / 4.0`
- **不包含数量相似度**

**V2 实现** (lines 654-677):
- 使用3个维度：`num_sim`, `spatial_sim`, `struct_sim`
- 加权平均：`num_sim * 0.3 + spatial_sim * 0.4 + struct_sim * 0.3`
- **包含数量相似度**

**评价**：
- ✅ **V1 更好**：根据用户需求，不包含数量相似度，包含 `spread_sim` 和 `clustering_sim`
- ⚠️ V2 的 `struct_sim` 是 `(symmetry_diff + clustering_diff) / 2.0`，而 V1 分开计算

---

### 4.3 `_compute_symmetry` (V1) vs `compute_symmetry` (V2)

**V1 实现** (lines 597-612):
- 检查关于**原点**的对称性
- 对每个点找对称点 `-pos`，检查是否存在（距离<0.05）

**V2 实现** (lines 604-628):
- 检查关于**质心**的对称性
- 对每个点找镜像点 `2 * centroid - pos`，检查是否存在（距离<0.05）

**评价**：
- ✅ **V2 更好**：关于质心对称更合理（配置可能不在原点）
- ⚠️ V1 的关于原点对称假设配置在原点附近

---

### 4.4 `_compute_clustering` (V1) vs `compute_clustering` (V2)

**V1 实现** (lines 614-628):
- 使用 `cdist` 计算所有点对距离
- 计算平均最近邻距离
- 归一化：`1.0 - min(avg_min_dist / 0.1, 1.0)`

**V2 实现** (lines 630-652):
- 计算所有点对距离（手动循环）
- 计算平均距离
- 归一化：`max(0, 1.0 - (avg_dist - 0.05) / 0.20)`

**评价**：
- ✅ **V1 更好**：
  - 使用 `cdist` 更高效
  - 使用最近邻距离更合理（反映局部聚集）
- ⚠️ V2 使用平均距离，可能被远距离点影响

---

## 5. 有效性检查相关

### 5.1 `_distance_to_line` (V1) vs `distance_to_line` (V2)

**V1 实现** (lines 630-646):
- 计算点到**直线**的距离（投影到无限直线）
- 使用 `np.clip(t, 0, 1)` 限制投影参数

**V2 实现** (lines 736-759):
- 计算点到**线段**的距离
- 如果投影在线段外，返回到端点的距离

**评价**：
- ✅ **V2 更好**：计算点到线段的距离更准确（障碍物距离路径，应该考虑线段范围）
- ⚠️ V1 计算到无限直线的距离，可能高估

---

### 5.2 `_can_push_straight` (V1) vs `can_push_straight` (V2)

**V1 实现** (lines 720-732):
- 调用 `_is_path_blocked`，取反

**V2 实现** (lines 822-827):
- 调用 `is_straight_path_clear`

**评价**：
- ✅ **基本相同**，都合理

---

## 6. 其他方法

### 6.1 `_check_tblock_obstacle_collision` (V1)

**V1 实现** (lines 734-748):
- 封装 `analytic_obs_collision_check`
- 处理坐标转换：`obs_center = obs_pos - tblock_pos`

**评价**：
- ✅ **V1 有封装**，代码更清晰

---

## 总结

### V1 更优的方法：
1. ✅ `_has_path` - 更系统的路径检查
2. ✅ `_is_path_blocked` - 考虑实际尺寸
3. ✅ `_compute_shortest_path_length` - 基于几何距离
4. ✅ `_sample_path_points` - 独有功能，考虑绕行
5. ✅ `_compute_min_clearance` - 有默认值保护
6. ✅ `_compute_rotation_space` - 考虑整个路径，测试多个角度
7. ✅ `_compute_similarity` - 符合用户需求（不包含数量，包含spread和clustering）
8. ✅ `_compute_clustering` - 使用最近邻距离，更高效

### V2 更优的方法：
1. ✅ `is_trapped` - 先检查初始碰撞
2. ✅ `compute_symmetry` - 关于质心对称
3. ✅ `distance_to_line` - 计算点到线段距离

### 建议：
- **主要使用 V1 的方法**，但可以借鉴 V2 的以下改进：
  1. `is_trapped` 中先检查初始碰撞
  2. `compute_symmetry` 关于质心对称
  3. `distance_to_line` 计算点到线段距离

