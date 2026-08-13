"""导航课程(Stage 0/2):LLM `generate_pillars` 生成器 + 治锁 verifier + evolve。

模块:
- generator_source:generate_pillars 契约、SandboxNavExecutor(校验/执行 LLM 代码)、
  MockGenerator(无 LLM 的确定性生成器,供 verifier/evolve 测试)。
- prompt_builder:方向化 prompt(RELAX/HARDEN/PRESERVE)。
- acgs_api:LLM 调用封装(需 OPENAI_API_KEY;含 mock 路径)。
- verifier:单标量 boundary_count 治锁 verifier(复用 Push-T 判据逻辑)。
- evolve:Stage 2 evolve 主循环。
- stage0:LLM 生成初始 G_0 + 选择。
"""
