"""Stage 3 主对比 + 消融 驱动(Req 12.3/12.4)。

主对比(对齐社区口径):random / static / eureka(success 引导)/ 本方法(库信号引导)。
消融:训库/不训、单技能/挑技能、K∈{4,8,16}、rollout ratio。

现状:主对比的 eureka(success 引导)与 static/random 需各自的课程信号变体;
本方法/库无课程可复用 train_skill(+Stage2 整合)。本脚本 print 消融命令矩阵,便于排队。
"""
import argparse


def print_ablations(seed=0):
    # 均走 train_stage2 hydra 开关(与主对比同一入口)。消融默认固定 train_library=true;
    # 除"训库 vs 不训"外,进化固定关(evolve.enabled=false)以隔离被消融变量。
    base = f"python -m nav.train_stage2 provider=mock seed={seed}"
    noevo = "curriculum.evolve.enabled=false"
    print("# Stage 3 消融命令矩阵(seed 示例 %d;均走 train_stage2 hydra 开关):" % seed)
    print("## 训库 vs 不训(backbone):")
    print(f"{base} training.train_library=true  {noevo} tag=abl_lib_s{seed}")
    print(f"{base} training.train_library=false {noevo} tag=abl_single_s{seed}")
    print("## codebook:one-hot vs random(D1;random 可设 d_w):")
    print(f"{base} training.train_library=true {noevo} skill.codebook_type=one_hot tag=abl_cb_onehot_s{seed}")
    print(f"{base} training.train_library=true {noevo} skill.codebook_type=random skill.d_w=6 tag=abl_cb_random_s{seed}")
    print("## 显式 r_div 当前未接入 Stage-2；YAML 会拒绝非零 beta_div，避免产生伪消融。")
    print("## K∈{4,8,16}:")
    for K in (4, 8, 16):
        print(f"{base} training.train_library=true {noevo} skill.K={K} tag=abl_K{K}_s{seed}")
    print("## rollout ratio P(w_0):")
    for p in (0.5, 0.7):
        print(f"{base} training.train_library=true {noevo} skill.w0_rollout_ratio={p} tag=abl_pw0_{p}_s{seed}")
    print("## 单技能 vs 挑技能(best-of-K,仅 ablation):见 nav.eval 扩展(w_0 vs argmax_k)")
    print("## 主对比 random/static/eureka:需各自课程信号变体(eureka=success 引导),与本方法同 backbone。")
    print("# 每个 tag 训完用 nav.eval 在 between-best 上评 B/M/U/D;nav.compare_fourcell --aggregate 汇总。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print_ablations(args.seed)


if __name__ == "__main__":
    main()
