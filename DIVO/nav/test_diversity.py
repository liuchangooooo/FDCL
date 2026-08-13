"""任务 9 组件 sanity:Form B 多样性纯函数。"""
import numpy as np

from nav import nav_env as NE
from nav.diversity import (embed_route, soft_gate, traj_quality, compute_rdiv,
                           action_var, sat_rate)


def main():
    start, goal = NE.START, NE.GOAL

    # 1) 轨迹嵌入:平移/旋转/尺度不变的任务坐标系。直线 start->goal -> ty≈0, tx 从 0 到 1
    straight = np.stack([np.linspace(start[0], goal[0], 20), np.zeros(20)], 1)
    e = embed_route(straight, start, goal, n=8)
    tx = e.reshape(-1, 2)[:, 0]; ty = e.reshape(-1, 2)[:, 1]
    assert abs(tx[0]) < 1e-6 and abs(tx[-1] - 1.0) < 1e-6, "tx 应从0到1"
    assert np.max(np.abs(ty)) < 1e-6, "直线横向应≈0"
    print(f"  1) 任务坐标嵌入:tx[0]={tx[0]:.3f} tx[-1]={tx[-1]:.3f} max|ty|={np.abs(ty).max():.3f} OK")

    # 绕行路径 ty 明显非零,且与直线嵌入距离大
    detour = np.stack([np.linspace(start[0], goal[0], 20),
                       0.3 * np.sin(np.linspace(0, np.pi, 20))], 1)
    e2 = embed_route(detour, start, goal, n=8)
    d_line_detour = np.linalg.norm(e - e2)
    print(f"  2) 直线 vs 绕行 嵌入距离={d_line_detour:.3f}(>0 => 路径可分)")
    assert d_line_detour > 0.1

    # 3) soft gate 单调
    assert soft_gate(1.0) > soft_gate(0.0) > soft_gate(-1.0)
    print("  3) soft_gate 单调 OK")

    # 4) r_div:一个技能走直线、其余绕行 -> 直线技能(与众不同)r_div 较高;全同 -> r_div≈0
    embeds = np.stack([e, e2, e2, e2])          # 技能0 直线,1/2/3 绕行(相同)
    gates = np.ones(4)
    r = compute_rdiv(embeds, gates, margin=0.5)
    print(f"  4) r_div={np.round(r,3)}(技能0 与众不同应最高)")
    assert r[0] == max(r) and r[0] > r[1] + 1e-6

    embeds_same = np.stack([e, e, e, e])
    r_same = compute_rdiv(embeds_same, gates)
    print(f"     全相同 r_div={np.round(r_same,3)}(应≈0)")
    assert np.max(r_same) < 1e-6

    # 5) 门控:失败技能(gate≈0)不贡献也不得分
    q = traj_quality(success=[1, 0], progress=[1, 0.1], collision=[0, 1], timeout=[0, 1])
    g = soft_gate(q)
    print(f"  5) traj_quality={np.round(q,2)} gate={np.round(g,3)}(成功>失败)OK")
    assert g[0] > g[1]

    # 6) 监控
    acts = np.array([[0.1, 0.2], [-0.1, 0.9], [0.0, -0.99], [0.3, 0.5]])
    print(f"  6) action_var={action_var(acts):.3f} sat_rate={sat_rate(acts):.3f} OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
