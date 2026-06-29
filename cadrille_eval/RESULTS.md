# SOTA Point-Cloud→B-Rep 评测：cadrille vs CAD-Recode（ABC）

按沈老师要求"系统评测 SOTA Point Cloud to BRep 方法"，在 **GeometryBench 的 ABC 测试集**上评测了两个 SOTA 方法（点云→CadQuery 代码→B-Rep）：

- **cadrille**（ICLR 2026，arXiv:2505.22914，多模态，RL/SOTA 权重 `maksimko123/cadrille-rl`）
- **CAD-Recode**（ICCV 2025，arXiv:2412.14042，cadrille 的前身，`filapro/cad-recode-v1.5`）

> 文献：[1] Kolodiazhnyi et al. *cadrille: Multi-modal CAD Reconstruction with Online RL.* ICLR 2026. arXiv:2505.22914.　[2] Rukhovich et al. *CAD-Recode: Reverse Engineering CAD Code from Point Clouds.* ICCV 2025. arXiv:2412.14042.

## 协议
- **测试集**：300 个 ABC 模型（STEP → 归一化单位立方体 STL；方法各自采 8192 点 → FPS 256 点输入）。与 GeometryBench 同源（ABC），刻意不用方法自带的 DeepCAD split——测"SOTA 在我们 benchmark 上的真实表现"。
- **流程**：点云 → 生成 CadQuery 代码 → `exec`（子进程+超时）→ OCCT 实体 → STEP/mesh → 指标。
- **指标**：代码可执行率、B-Rep 合法率（kernel 体检）、Chamfer / F-score@0.02（预测 vs GT，均归一化）。`raw` = 仅居中+缩放对齐；`icp` = 再加 ICP 旋转对齐（公平性检验）。
- 单张 RTX 4090。

## 结果（300 模型）

| 指标 | **cadrille-rl** | **CAD-Recode-v1.5** |
|---|---|---|
| 代码可执行率 | **100%** | 87.3% |
| B-Rep 合法率 | **99%** | 89.6% |
| median Chamfer | **0.037** | 0.043 |
| F-score@0.02 (raw) | **0.673** | 0.620 |
| F-score@0.02 (ICP) | 0.692 | 0.649 |

按难度（F-score@0.02 / median Chamfer）：

| 难度 | cadrille | CAD-Recode |
|---|---|---|
| simple | 0.77 / 0.024 | 0.76 / 0.025 |
| complex | **0.65** / 0.041 | 0.58 / 0.048 |

![GT vs cadrille 重建](results/fig_recon_gallery.png)
<sub>上排 GT，下排 cadrille 重建；左 3 = Chamfer 最小，右 3 = 最大。</sub>

## 结论
1. **cadrille 更可靠**：100% 可执行 / 99% 合法，远高于 CAD-Recode 的 87% / 90%。原因——cadrille 偏用高层 primitive（cylinder/box），代码稳健；CAD-Recode 偏 sketch-segment（画轮廓再拉伸），易产生自交/非法草图 → exec 失败。
2. **cadrille 精度也略高**（F 0.67 vs 0.62），差距主要在 complex（0.65 vs 0.58），simple 上基本打平。
3. **cadrille（更新的 RL 升级版）全面优于其前身 CAD-Recode**——benchmark 正确地把这个高下排了出来，本身就验证了评测的有效性。
4. **两个方法都在 complex 上明显退化**（难度效应跨方法一致）——说明这套"按难度分层"的评测确实抓到了真实的能力梯度。
5. **ICP 对齐只带来 ~3% 提升** → 两方法输出本就对齐良好，原始 Chamfer/F-score 公平、不是被旋转高估。

## 诚实说明
- CAD-Recode 的 Chamfer/F-score 是在它**成功 exec 的 ~260 个**上算的（38 个 exec 失败无 mesh），而 cadrille 是全 300 个——这对 CAD-Recode 略有利，所以真实差距可能更大一点。可执行率/合法率才是全 300 口径的公平对比。
- F-score@0.02 对采样密度敏感，此处统一用 n=4096 采样。

## 在 GeometryBench 中的位置
这是一个**重建评测**（点云→整模型 B-Rep），比 L1 识别高一档（接近方案的 L4 生成）。与 L1 共用同一 ABC 底座、同一 kernel-as-oracle 取标签/判合法的范式，体现"一套数据底座贯穿多级评测"。

## 复现
脚本见 `cadrille_eval/`：`prep_testset.py`、`run_full.sh`（cadrille）、`cadrecode_run.py` + `run_cadrecode_full.sh`（CAD-Recode）、`exec_to_brep.py`、`eval_recon2.py`（含 ICP + 难度拆分）、`viz_recon.py`。环境 `cadrille_setup.sh` / `cadrecode_setup.sh`（AutoDL 上需 `source /etc/network_turbo`）。
