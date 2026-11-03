# V3 300k 配对续训：中断后的恢复流程

本文件补充 `MASAC_V3_PAIRED_300K_20261103.md` 的中断处理。正式批次在一个新目录顺序运行六个训练历史/种子；每 50,000 步保存同一步的模型、Replay 与完整 manifest。**只有含 `complete.json` 且哈希核验通过的恢复点才可用**；仅有 10,000 步间隔模型 checkpoint 的目录不足以精确恢复。

1. 查看正式批次的 `error.json`、状态和日志，确认原训练进程已退出并释放全局训练锁；保留原结果目录及错误记录。
2. 找出中断分支最近的完整 `snapshots/<step>/complete.json`。将它作为 `--resume-snapshot` 传入续训器，以全新的 `--out` 与 `--log` 恢复该单一分支到 300,000 步。续训器核对原父批、恢复点文件哈希、恢复点所属配对运行的源码快照/协议及当前代码，不重新暖身；不得将模型单独的 checkpoint 和不同步的 Replay 拼接。
3. 对其他尚未开始的分支，用重复的 `--run-item curriculum:422`、`--run-item original:411` 等参数从其各自完整 100,000 步父状态在另一个新目录续跑；按冻结协议顺序执行所选择的项目。先核对原批次及所有恢复分片的 `audit.json`、300,000 步评估和对齐恢复点，已完成分支无需重跑。
4. 恢复点分支与指定分支都是显式分片：它们写 `recovery_completion.json` 与 `selected_runs`，不写表示完整六分支的 `completion.json`。不要把分片完成标志当作整批完成。最终比较须收集六个不同历史/种子的有效最终结果，核对各自来源、300,000 总步数、299,000 更新、地图、检查点与 Replay；保留所有失败，记录结果跨多个目录。该分片恢复流程不会自动汇合原批次的完成标志。

操作示意（目录和种子必须根据实际错误日志选择，命令中的目录均需尚不存在）：

```bash
cd /Users/a1/Desktop/research/MASAC
.venv/bin/python -u scripts/continue_masac_v3_paired.py \
  --protocol configs/masac_v3/paired_300k_protocol.json \
  --resume-snapshot results/PAIRED_RUN/curriculum_411/snapshots/150000 \
  --out results/RECOVERY_SHARD --log results/RECOVERY_SHARD.log

.venv/bin/python -u scripts/continue_masac_v3_paired.py \
  --protocol configs/masac_v3/paired_300k_protocol.json \
  --run-item curriculum:422 --run-item original:411 \
  --out results/PENDING_SHARD --log results/PENDING_SHARD.log
```

以上是手动恢复示意，不会自动启动。若中断发生在首次完整恢复点之前，当前种子只能从其已保存的 100,000 步父状态重跑；若源码已修改或父结果哈希不符，严格恢复会拒绝执行，需先调查来源并另行验证，不得直接忽略检查。
