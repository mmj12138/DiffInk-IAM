# IAM character JSON 接入

在项目根目录运行，使用当前已安装 torch、numpy、h5py、PyYAML 的 Python。

## 已完成的样本验证

`datas/iam_example/test.h5` 来自用户提供的 `a01-013z-01.xml.json`。
共 502 个点，21 个字符（含空格），collate 后为 `[1,504,5]`。
`smoke_report.json` 记录实际运行结果。

```bash
python scripts/smoke_iam.py --config datas/iam_example/vae_iam.yaml --checkpoint outputs/ink_vae_03_14_02_12/vae_epoch_100.pt
```

该命令加载现有 VAE 的 200 个轨迹权重张量，重新初始化 IAM OCR 和 writer 分类头，执行真实 TrainDataset/ValDataset、VAE、全部训练损失、backward、AdamW 更新及 DiT 前缀 mask 检查。更新只在内存中进行，不覆盖 checkpoint。省略 `--checkpoint` 可验证随机初始化 VAE。

这是单 batch 接入验证，未运行 DiT 前向、完整训练或生成质量评估。test 样本的内存更新只用于验证代码路径，不能作为训练或性能评估结果。

## 批量转换

传单个 JSON 或目录；目录递归读取全部 JSON。训练、验证、测试应分别转换，不能混用。

```bash
python scripts/convert_iam_json.py --input /path/to/train_json --output datas/iam_train --split train
python scripts/convert_iam_json.py --input /path/to/val_json --output datas/iam_val --split val --text-file datas/iam_train/text.json --writer-file datas/iam_train/writers.json
python scripts/convert_iam_json.py --input /path/to/test_json --output datas/iam_test --split test --text-file datas/iam_train/text.json --writer-file datas/iam_train/writers.json
```

输出目录须无同名产物，避免覆盖。转换器遇到文本不一致、无覆盖点、重叠范围、缺失 writer 或未知字符会报错，并指出源文件，不会静默丢弃。

- 每个 HDF5 group 是整行，包含 `point_seq`、`char_points_idx`、`line_text`、`writer_id`。
- `point_seq` 为 `[N,5]`：绝对 x/y 坐标和三个 one-hot 状态（连接下点、笔画结束、字符结束）。不是坐标差分。IAM y 向下，转换后 y 向上；整行均值居中，以原始 y 标准差等比例缩放，保持长宽比。仓库未提供原始预处理代码，该归一化是明确的新转换约定，不保证与作者预处理完全相同。
- `inkRanges` 两端都包含，按 `(stroke,index)` 定位；保留全部点，不重采样。使用 `gt_segmentation`，不以 CTC spike 替代字符边界。
- `char_points_idx` 是每个字符的累计点数（右开端点）；无轨迹空格沿用前一个端点，文本里的空格保留。
- 字符结束使用 `001`。原始跨字符连接在字符末端按仓库约定断开，不插入虚构空格笔画。
- 字典按 JSON key 顺序编码，CTC blank 单独占 0。额外保留 `、` 作为现有 ValDataset 追加的哨兵，不修改 IAM 原文。
- 训练加载器只接收 200..2000 点，验证加载器接收至少 200 点；`manifest.json` 给出训练长度合格数量。短行和长行不会为了通过过滤而自动补点或截断。
- 不同 split 必须共享训练词表；只有一个示例生成的词表只适用于该样本。正式训练可预先提供完整目标字符词表，避免验证/测试出现 OOV。

## 后续完整训练

把训练输出的 `vae_iam.yaml` 中 `train_file` 指向训练 HDF5，`val_file` 指向独立验证 HDF5，`text_file`/`writer_file` 均使用训练元数据。当前自动生成的 YAML 只填写本次转换所属 split 的路径，其他路径为 null，防止测试数据意外成为正式训练集。

原来的 `train_vae_ddp.py` 固定读取 `configs/vae_config.yaml`，且只支持 CUDA/NCCL；正式 GPU 训练前需将以上配置填入该文件。DiT 配置同样替换数据路径，VAE checkpoint 应指向 IAM 训练后的权重。原始 requirements 含 Linux CUDA 专用包，不能原样在 Mac 安装；本次 CPU VAE 验证只需要上述基础依赖和 matplotlib。

原 DiT 还依赖 x-transformers/einops/einx，且 TextEmbedding 内有硬编码 `.to('cuda')`。本次未改动 GPU 训练行为，也未声称完整 DiT 已在本机跑通。中文模型的 OCR/文本字典、writer 索引不能直接套在 IAM 上；重新初始化对应头或按字符映射迁移后继续训练。
