# GeometryBench — SOTA Point-Cloud→B-Rep 评测（cadrille）

目标：按沈老师要求"系统评测 SOTA Point Cloud to BRep 方法"，跑通一个 SOTA 方法（**cadrille**，CAD-Recode 的多模态升级版，ICLR 2026），在 **ABC 测试集**上做点云→B-Rep 重建评测，产出可比较的指标表 + 图。够交差，且并入 GeometryBench 仓库作为"重建评测"一层。

## 方法选择
- **首选 cadrille**（`github.com/col14m/cadrille`，权重 `maksimko123/cadrille-rl`，RL/SOTA，点云+图）。
- **兜底 CAD-Recode**（`filapro/cad-recode-v1.5`，更简单、自带 demo、纯点云）——先用它 10 分钟 de-risk 工具链，再切 cadrille。
- 机制：Qwen2 + FourierPointEncoder → 生成 CadQuery 代码 → `exec` → CadQuery/OCCT 实体 → 导 STEP。

## 单机方案（关键简化）
brep 本身就是 4090 机器，之前用的是它的**无卡模式**（CPU 限速、不计 GPU）。**用 4090 开机** → 满血 4090 + 完整 CPU/内存，而且**之前的 conda 环境(`brep`，含 pythonocc)、ABC 数据(`/root/autodl-tmp/gbench_work/abc_data`)、脚本(`/root/gbench`) 都在持久盘上**。所以全程一台机：

- 复用已有 ABC 数据做测试集（**不用重下**）；复用 `brep` 环境做 CPU 备数据/算指标；
- 新建 `cadrille` conda 环境跑 GPU 推理；
- 开机即用，一个会话内 备数据→装环境→推理→评测 全做完，完事关机。GPU 计费时间最小化。

> 唯一操作点：**用 4090 重启 brep 后，AutoDL 的 SSH 端口通常会变**，需要把新的 `ssh -p <新端口> root@...` 给我，我更新 `brep` alias。

## 流水线（全在 4090-brep 一台机上）
1. 取测试模型：复用磁盘上已有 ABC STEP（挑 held-out ~300 个，与 L1 不重叠）；不够再 `fetch_abc.py` 补。
2. `prep_testset.py`：每个 STEP → 归一化到单位立方体的 **STL**（cadrille stock loader 直接吃 STL）+ 留同归一化 mesh 作 GT。
3. `cadrille_setup.sh` 装 `cadrille` 环境 + 下权重 → `test.py --split abc_test --mode pc` → 每模型一段 CadQuery 代码；开 STEP 导出 → 每模型一个 STEP/mesh。
4. `eval_recon.py`（用 `brep` 环境，CPU）：预测 vs GT → **Chamfer、F-score@τ、B-Rep 合法率(kernel)、代码可执行率、无效率**。
5. **产出**：结果表 + 几张图，写进 `GeometryBench/` 仓库（"SOTA 方法重建评测"小节）。

## 关键坑（来自 runbook，务必照做）
- **归一化**：STL 必须归一化到单位立方体 `[0,1]³`（中心 0.5、最长边 1）；喂错尺度会**静默出垃圾**。
- **不是位姿不变**：输入点云需轴对齐（ABC 件一般已对齐，核验即可）。
- **256 点**：模型只吃 256 点（stock loader 自动 8192→FPS 256）。
- **flash-attn**：硬编码，装不上就把 `test.py` 改成 `attn_implementation='sdpa'`。
- **transformers==4.50.3** 不能漂。
- `exec` 生成代码要**子进程 + 超时**（repo 里 evaluate.py 已有 3s 超时范式）。
- 输出几何在 **±100 量纲**，算 Chamfer 前把 pred 和 GT 都归一化到同一规范帧。

## 指标定义
- **Chamfer 距离 / F-score@τ**：预测重建表面 vs GT 表面（双向，归一化帧）。
- **B-Rep 合法率**：导出的 STEP 用 kernel 体检（watertight / 无自交 / 可执行）。
- **代码可执行率**：生成的 CadQuery 代码能否 `exec` 成功出实体。
- （可选）**按曲面类型的恢复情况**——衔接 L1。

## 默认决定（如需改告诉我）
- checkpoint：`maksimko123/cadrille-rl`（SOTA）。
- 测试集：ABC held-out ~300 个（与 L1 训练集不重叠；用 model_id 哈希分）。
- 数据源：ABC（与 GeometryBench 一致），不用 DeepCAD 自带 split——这样测的是"SOTA 方法在我们 benchmark 上的表现"，分布偏移本身就是有意思的结果。

## 文件
```
cadrille_eval/
  PLAN.md              本文件
  cadrille_setup.sh    [4090] 环境安装（minimal，sdpa 跳 flash-attn）
  prep_testset.py      [brep] ABC STEP → 归一化 STL + GT mesh
  run_cadrille.sh      [4090] 跑推理 + 开 STEP 导出
  eval_recon.py        [brep] 预测 vs GT → Chamfer/F-score/合法率
  DEVLOG.md            进度记录（仿 JOFD-SLAM 风格）
```
