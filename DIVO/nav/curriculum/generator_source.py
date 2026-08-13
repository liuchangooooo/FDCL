"""generate_pillars 生成器契约 + sandbox 执行器 + mock 生成器。

契约(LLM 需生成):
    def generate_pillars(agent_start, goal, num):
        # 返回 [(x,y), ...]  长度==num,pillar 圆柱中心坐标

物理合法性(physical verifier 用):在界内、互斥 keepout、离 start/goal 净空、
不平凡围死 goal。真正的 reset 合法性由 NavEnvAdapter 校验(会抛 ResamplingError)。
"""
import threading

import numpy as np

from nav import nav_env as NE

# 生成/校验用的几何约束(随小世界;与 nav_env 尺度绑定,物理净空按 goal/pillar/agent 尺寸)
BOUND = NE.OBSTACLE_REGION                       # pillar 允许坐标范围 [-BOUND,BOUND](=规范障碍区)
MIN_PAIR = NE.PILLAR_MIN_SEPARATION               # 实体不重叠 + 0.04 余量(0.34)
# goal 同时覆盖实体尺寸和 Safety-Gym placement keepout,再加 0.04 余量。
GOAL_CLEAR = max(
    NE.GOAL_SIZE + NE.PILLAR_SIZE,
    NE.GOAL_KEEPOUT + NE.PILLAR_KEEPOUT,
) + 0.04                                                       # 0.39
# 起点需要额外机动空间；由 nav_env 作为训练/validation 单一事实来源。
START_CLEAR = NE.START_PILLAR_CLEARANCE                          # 0.45
# 危险 token 黑名单:只拦"模块访问/进程/序列化"类。open(/eval(/exec( 不列入——
# 它们不在沙箱白名单 __builtins__ 里,运行期本就会 NameError;而 "open(" 会误杀
# 导航代码里合法的 is_open(/reopen(/注释里的 open( 等(实测 LLM 触发)。
_DANGEROUS = ("import os", "import sys", "subprocess", "socket", "shutil",
              "importlib", "pickle", "__import__", "eval(", "exec(")


def _safe_import(name, *args, **kwargs):
    """沙箱内受限 import:只允许 numpy(及其子模块),其余一律拒绝。

    让 LLM 合法的 `import numpy as np` 可用,同时挡住 os/sys 等模块导入。
    """
    root = str(name).split(".")[0]
    if root == "numpy":
        import numpy as _np
        return _np if name == "numpy" else __import__(name, *args, **kwargs)
    raise ImportError(f"import of '{name}' is not allowed in sandbox")


def make_is_safe(bound=BOUND, goal_clear=GOAL_CLEAR, start_clear=START_CLEAR):
    """生成注入沙箱的 is_safe(x,y,sx,sy,gx,gy):逐候选校验(界内 + 离 start/goal 净空)。

    注:pairwise keepout / 不围死 goal 由 validate_pillars 事后统一校验(is_safe 只看单点)。
    """
    def is_safe(x, y, sx, sy, gx, gy):
        x = float(x); y = float(y)
        if not (abs(x) <= bound and abs(y) <= bound):
            return False
        if (x - float(gx)) ** 2 + (y - float(gy)) ** 2 < goal_clear ** 2:
            return False
        if (x - float(sx)) ** 2 + (y - float(sy)) ** 2 < start_clear ** 2:
            return False
        return True
    return is_safe


def _to_xy(item):
    """把生成器返回项统一成 (x,y):支持 {'x','y',...} 字典或 (x,y) 序列。"""
    if isinstance(item, dict):
        return float(item["x"]), float(item["y"])
    return float(item[0]), float(item[1])


def validate_pillars(pillars, start, goal, num=None):
    """轻量物理校验,返回 (ok, reason)。真正 reset 合法性另由 adapter 校验。"""
    if not isinstance(pillars, (list, tuple)) or len(pillars) == 0:
        return False, "empty_or_not_list"
    if num is not None and len(pillars) != num:
        return False, f"num_mismatch({len(pillars)}!={num})"
    P = []
    for p in pillars:
        if len(p) != 2:
            return False, "point_not_xy"
        x, y = float(p[0]), float(p[1])
        if not (abs(x) <= BOUND and abs(y) <= BOUND):
            return False, "out_of_bounds"
        P.append((x, y))
    s = np.array(start, float); g = np.array(goal, float)
    for i, (x, y) in enumerate(P):
        if np.hypot(x - g[0], y - g[1]) < GOAL_CLEAR:
            return False, "too_close_goal"
        if np.hypot(x - s[0], y - s[1]) < START_CLEAR:
            return False, "too_close_start"
        for j in range(i + 1, len(P)):
            if np.hypot(x - P[j][0], y - P[j][1]) < MIN_PAIR:
                return False, "pillar_overlap"
    # 不平凡围死 goal:goal 周围一圈过密(近似:>=6 根在 goal 0.35 半径内;小世界缩小)
    near_goal = sum(1 for x, y in P if np.hypot(x - g[0], y - g[1]) < 0.35)
    if near_goal >= 6:
        return False, "encloses_goal"
    return True, "ok"


class SandboxNavExecutor:
    """执行 LLM 生成的 generate_pillars 代码(受限命名空间 + 基本安全过滤)。"""

    def __init__(self, timeout_sec=5):
        self.timeout_sec = int(timeout_sec)
        self.code = None
        self._fn = None

    def load_code(self, code_str):
        low = code_str.lower()
        for bad in _DANGEROUS:
            if bad in low:
                return False, f"dangerous_token:{bad}"
        if "def generate_pillars" not in code_str:
            return False, "missing_generate_pillars"
        safe_globals = {"__builtins__": {
            "range": range, "len": len, "float": float, "int": int, "abs": abs,
            "min": min, "max": max, "list": list, "tuple": tuple, "enumerate": enumerate,
            "round": round, "sum": sum, "zip": zip, "dict": dict, "sorted": sorted,
            "__import__": _safe_import,   # 只放行 import numpy(其余 ImportError)
        }, "np": np, "numpy": np, "is_safe": make_is_safe()}
        loc = {}
        try:
            exec(code_str, safe_globals, loc)
        except Exception as e:
            return False, f"exec_error:{e}"
        fn = loc.get("generate_pillars") or safe_globals.get("generate_pillars")
        if not callable(fn):
            return False, "generate_pillars_not_callable"
        self.code = code_str
        self._fn = fn
        return True, "ok"

    def generate(self, agent_start, goal, num):
        """执行 LLM 代码产布局。带 SIGALRM 超时,防止代码内无上限循环卡死训练/evolve。

        超时/异常由上层 try/except 兜底(视作该场景无效)。仅在主线程生效(nav 单线程训练)。
        """
        if self._fn is None:
            raise RuntimeError("no code loaded")
        import signal
        use_alarm = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
        old_handler = None
        if use_alarm:
            def _on_timeout(signum, frame):
                raise TimeoutError("generate_pillars exceeded timeout")
            old_handler = signal.signal(signal.SIGALRM, _on_timeout)
            signal.setitimer(signal.ITIMER_REAL, float(self.timeout_sec))
        try:
            out = self._fn(np.asarray(agent_start, float), np.asarray(goal, float), int(num))
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)
        return [_to_xy(item) for item in out]


class MockGenerator:
    """无 LLM 的确定性/参数化生成器,供 verifier/evolve 端到端测试。

    difficulty ∈ [0,1]:越大 pillar 越多、越靠 start->goal 连线(越难)。
    generate(start, goal, num=None) 条件于起点(先起点后生成器)。
    """

    def __init__(self, difficulty=0.5, seed=0, base_num=2):
        self.difficulty = float(np.clip(difficulty, 0, 1))
        self.rng = np.random.default_rng(seed)
        self.base_num = int(base_num)

    def generate(self, agent_start, goal, num=None):
        s = np.asarray(agent_start, float); g = np.asarray(goal, float)
        n = num if num is not None else self.base_num + int(round(self.difficulty * 3))
        n = max(1, n)
        mid = (s + g) / 2.0
        seg = g - s; L = np.linalg.norm(seg) + 1e-9; u = seg / L
        perp = np.array([-u[1], u[0]])
        pts = []
        tries = 0
        # 难度高 => 横向偏移小(更挡在连线上);难度低 => 偏移大(更易绕)
        lateral_scale = 0.6 * (1 - self.difficulty) + 0.05
        while len(pts) < n and tries < 200:
            tries += 1
            along = float(self.rng.uniform(-0.3, 0.3))
            lat = float(self.rng.uniform(-1, 1)) * lateral_scale
            p = mid + u * along + perp * lat
            p = np.clip(p, -BOUND, BOUND)
            cand = (round(float(p[0]), 3), round(float(p[1]), 3))
            if all(np.hypot(cand[0] - x, cand[1] - y) >= MIN_PAIR for x, y in pts):
                ok, _ = validate_pillars(pts + [cand], agent_start, goal)
                if ok:
                    pts.append(cand)
        if not pts:
            pts = [(round(float(mid[0]), 3), round(float(mid[1]), 3))]
        return pts
