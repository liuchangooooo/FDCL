"""Form B 多样性(路径多样,V0 默认关 β_div=0)。

- 轨迹嵌入 φ:agent (x,y) 路径在 start->goal 任务坐标系(平移+旋转+按 |goal-start| 归一化)
  下定长重采样展平,使不同场景可比。
- 同布局配对:同一 layout(同 seed/同起点)枚举 K 技能得 τ_k;距离只在同布局内算。
- 软 near-success 门控:q=α·success+β·progress−γ·cost−η·timeout,g=σ((q−τ_q)/T)。
- margin 有界:距离超过 m 不再加分。
- ActionVar/SatRate:仅监控与塌缩探测,不作训练目标。

对外:
  embed_route(xy, start, goal, n)         -> [2n] task-frame 定长嵌入
  soft_gate(q, tau_q, T)                  -> g∈(0,1)
  compute_rdiv(embeds, gates, margin)     -> [K] 每技能 r_div(同布局内)
  paired_diversity_rollout(adapter, agent, layout, K, ...) -> dict(transitions, r_div, ...)
  action_var / sat_rate(actions)          -> 监控标量
"""
import numpy as np


# ----------------------- 轨迹嵌入 -----------------------
def embed_route(xy, start, goal, n=16):
    """把 agent (x,y) 路径嵌入到 start->goal 任务坐标系并定长重采样为 [2n]。

    x 轴 = start->goal 方向,原点 = start,尺度 = |goal-start|。
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    s = np.asarray(start, float)
    g = np.asarray(goal, float)
    d = g - s
    L = float(np.linalg.norm(d)) + 1e-9
    u = d / L
    perp = np.array([-u[1], u[0]])
    rel = xy - s[None, :]
    tx = rel @ u / L                      # 沿 start->goal 归一化坐标
    ty = rel @ perp / L                   # 横向归一化坐标
    if len(xy) < 2:
        prof = np.zeros((n, 2))
    else:
        t = np.linspace(0, 1, len(xy))
        tt = np.linspace(0, 1, n)
        prof = np.stack([np.interp(tt, t, tx), np.interp(tt, t, ty)], axis=1)
    return prof.reshape(-1)               # [2n]


def soft_gate(q, tau_q=0.0, T=0.5):
    # 对齐 Push-T DiversityConfig(temperature=0.5)
    return 1.0 / (1.0 + np.exp(-(np.asarray(q, float) - tau_q) / T))


def traj_quality(success, progress, collision, timeout,
                 alpha=1.0, beta=1.0, gamma=1.0, eta=0.5):
    """q(τ)=α·success+β·progress−γ·collision−η·timeout(对齐 Push-T α=β=γ=1, η=0.5)。"""
    return (alpha * np.asarray(success, float) + beta * np.asarray(progress, float)
            - gamma * np.asarray(collision, float) - eta * np.asarray(timeout, float))


def compute_rdiv(embeds, gates, margin=1.0, eps=1e-6):    # margin 对齐 Push-T(=1.0)
    """同布局内 K 技能的 r_div(软门 + margin 有界)。

    embeds: [K, D];gates: [K]。
    r_div(k) = g_k · Σ_{l≠k} g_l·min(d(φ_k,φ_l), m) / (Σ_{l≠k} g_l + eps)
    """
    E = np.asarray(embeds, float)
    g = np.asarray(gates, float)
    K = len(E)
    r = np.zeros(K)
    for k in range(K):
        num = 0.0
        den = 0.0
        for l in range(K):
            if l == k:
                continue
            dist = float(np.linalg.norm(E[k] - E[l]))
            num += g[l] * min(dist, margin)
            den += g[l]
        r[k] = g[k] * num / (den + eps)
    return r


# ----------------------- 监控(不作目标)-----------------------
def action_var(actions):
    """ActionVar:同 obs 多技能动作方差和(对标 DIVO 表 II 口径)。actions: [K, A]。"""
    return float(np.var(np.asarray(actions, float), axis=0).sum())


def sat_rate(actions, thresh=0.98):
    """SatRate:动作贴边界(|a|>thresh)的比例,塌缩/多样失败探测。"""
    a = np.abs(np.asarray(actions, float))
    return float((a > thresh).mean())


# ----------------------- 同布局配对 rollout -----------------------
def paired_diversity_rollout(adapter, act_fn, layout, K, start, max_steps=500, seed=0):
    """对同一 layout(同起点)枚举 w_1..w_K,收集轨迹与质量,算 r_div。

    act_fn(obs, skill_id)->action;返回 dict:
      transitions[k] = list of (obs,a,next_obs,done,reward_task),用于写 buffer(source=diversity)
      r_div[k],gates[k],embeds[k],action_var(首步),各技能 success
    只用 w_1..w_K(不含 w_0)。
    """
    from safety_gymnasium.utils.common_utils import ResamplingError
    goal = adapter.goal
    per_k_trans = {}
    embeds = []
    successes = []
    progresses = []
    collisions = []
    timeouts = []
    first_actions = []
    for k in range(1, K + 1):
        adapter.set_layout(layout, start=start)
        try:
            obs = adapter.reset(seed=seed, start=start)
        except ResamplingError:
            return None
        path = [adapter.obs2state(obs)[:2] if False else adapter._env.task.data.body("agent").xpos[:2].copy()]
        trans = []
        succ = False
        coll = 0.0
        steps = 0
        d0 = float(np.linalg.norm(np.array(goal) - path[0]))
        for t in range(max_steps):
            a = act_fn(obs, k)
            if t == 0:
                first_actions.append(np.asarray(a, float))
            nobs, r_task, term, trunc, info = adapter.step(a)
            coll += info.get("cost", 0.0)
            done = term or adapter.success(info)
            trans.append((obs, a, nobs, float(done), r_task))
            axy = adapter._env.task.data.body("agent").xpos[:2].copy()
            path.append(axy)
            obs = nobs
            steps = t + 1
            if adapter.success(info):
                succ = True
                break
            if term or trunc:
                break
        dT = float(np.linalg.norm(np.array(goal) - path[-1]))
        progress = np.clip((d0 - dT) / (d0 + 1e-9), 0, 1)
        per_k_trans[k] = trans
        embeds.append(embed_route(path, start, goal))
        successes.append(1.0 if succ else 0.0)
        progresses.append(progress)
        collisions.append(1.0 if coll > 0 else 0.0)
        timeouts.append(1.0 if (steps >= max_steps and not succ) else 0.0)
    q = traj_quality(successes, progresses, collisions, timeouts)
    gates = soft_gate(q)
    r_div = compute_rdiv(embeds, gates)
    return {
        "transitions": per_k_trans,          # {k: [(obs,a,next_obs,done,r_task)...]}
        "r_div": {k + 1: float(r_div[k]) for k in range(K)},
        "gates": {k + 1: float(gates[k]) for k in range(K)},
        "success": {k + 1: successes[k] for k in range(K)},
        "action_var": action_var(first_actions) if len(first_actions) == K else float("nan"),
        "sat_rate": sat_rate(first_actions) if len(first_actions) == K else float("nan"),
    }
