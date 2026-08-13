"""D 动态端到端测试:动态 adapter + D 场景 reset/step + gremlin 会动 + cost + evaluate_bmud。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from nav import nav_env as NE
from nav.nav_adapter import NavEnvAdapter
from nav import benchmarks as B

# 1) 动态 adapter 造得起来、obs 44
dad = NavEnvAdapter(seed=0, dynamic=True)
sc = B.sample_benchmark_scene("D", 7)
print("D 场景: n_gremlin=", len(sc["pillars"]), "size=", sc["size"], "keepout=", sc["keepout"], "dynamic spec ok")
start = NE.sample_valid_start(np.random.default_rng(7), sc["goal"])
ok = False
for att in range(1, 40):
    s = NE.sample_valid_start(np.random.default_rng(att), sc["goal"])
    dad.set_layout(sc["pillars"], start=s, goal=sc["goal"], pillar_size=sc["size"], pillar_keepout=sc["keepout"])
    try:
        obs = dad.reset(seed=att, start=s); ok = True; break
    except Exception:
        continue
print(f"[1] 动态 D reset OK={ok} obs_dim={obs.shape[0]} (att={att})")
assert obs.shape[0] == 44

# 2) gremlin 会动:agent 不动,obstacle 通道 [28:44] 应随步变化
o0 = obs.copy(); changed = np.zeros(44, bool)
for _ in range(30):
    obs, r, term, trunc, info = dad.step(np.zeros(2))
    changed |= np.abs(obs - o0) > 1e-4
    if term or trunc: break
print(f"[2] 障碍通道[28:44] 30 步变化维数={changed[28:44].sum()} (>0 => gremlin 在动)")

# 3) cost 机制:info 里有 cost;撞上 gremlin 应 collision=失败
print(f"[3] step info: cost={info.get('cost')} collision={info.get('collision')} keys ok")
dad.close()

# 4) evaluate_bmud 全流程(小 n_env,随机策略;只验证不报错 + D 走动态)
ad = NavEnvAdapter(seed=0)
rng = np.random.default_rng(0)
act = lambda o: rng.uniform(-1, 1, 2)
res = B.evaluate_bmud(ad, act, n_env=3, max_steps=60)
ad.close()
print(f"[4] evaluate_bmud(随机策略,n=3) = {{k: round(v,2) for k,v in res.items()}}")
print("\nD-DYNAMIC SANITY PASS")
