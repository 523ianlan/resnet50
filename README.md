# UFALP Pruning Pipeline

這個專案把模型剪枝與微調流程包成一條可重用的 pipeline。

目前支援：

- 用 CLI 直接選內建模型
- 用 Python API 傳入自己的 `PyTorch nn.Module`
- 設定剪枝率、剪枝上下界、allocation 超參數、微調 epoch
- 輸出剪枝前後與微調後的模型、報告與視覺化結果

## 1. 專案核心檔案

主流程需要的檔案如下：

- `main.py`
- `configs/config.py`
- `data/build.py`
- `data/transforms.py`
- `models/resnet_setup.py`
- `models/custom_layers.py`
- `models/utils.py`
- `pruning/`
- `utils/`

## 2. 內建模型

可直接從 CLI 選的模型：

- `resnet18`
- `resnet34`
- `resnet50`
- `simple_cnn`
- `mlp_small`
- `mlp_medium`

模型來源：

- `resnet18/resnet34/resnet50`
  - 來自 `torchvision.models`
- `simple_cnn`
  - 寫在 `models/resnet_setup.py` 的 `SimpleCNN`
- `mlp_small/mlp_medium`
  - 寫在 `models/resnet_setup.py` 的 `FlattenMLP`

### pretrained 支援

以下模型可以用 `--pretrained`：

- `resnet18`
- `resnet34`
- `resnet50`

not pretrained：

- `simple_cnn`
- `mlp_small`
- `mlp_medium`

## 3. PowerShell

```powershell
python main.py --model-name resnet50 --pretrained --target-compression 0.4 --fine-tune-epochs 10 --save-dir ./results/resnet50
```

or

```powershell
python main.py `
  --model-name resnet50 `
  --pretrained `
  --target-compression 0.4 `
  --fine-tune-epochs 10 `
  --save-dir ./results/resnet50
```

## 4. CLI 參數

### 模型與資料

- `--model-name`
  - 選擇內建模型
  - 例：`resnet50`, `resnet18`, `simple_cnn`, `mlp_small`
- `--pretrained`
  - 使用 pretrained 權重
- `--no-pretrained`
  - 不用 pretrained 權重
- `--train-root`
  - 訓練資料根目錄
- `--val-root`
  - 驗證資料根目錄
- `--batch-size`
  - batch size
- `--num-workers`
  - dataloader workers 數量
- `--device`
  - `cuda` 或 `cpu`

### Toy model 形狀

- `--input-channels`
  - 輸入通道數
- `--input-height`
  - 輸入高度
- `--input-width`
  - 輸入寬度
- `--num-classes`
  - 類別數
- `--mlp-hidden-dims`
  - `mlp_medium` 的 hidden dims，逗號分隔
  - 例：`1024,512`
- `--cnn-channels`
  - `simple_cnn` 的卷積通道數，逗號分隔
  - 例：`32,64,128`

## 5. 剪枝相關超參數

### 核心剪枝參數

- `--target-compression`
  - 目標壓縮率
  - 例：`0.4` 代表目標壓縮 40%
- `--min-rank`
  - 每層允許的最小 rank
- `--use-log-s`
  - 使用 log-singular 參數化
- `--no-log-s`
  - 不使用 log-singular 參數化

### 剪枝上下界

- `--pruning-clip-low`
  - 每層最小剪枝率
  - 例：`0.05` 代表每層至少剪 5%
- `--pruning-clip-high`
  - 每層最大剪枝率
  - 例：`0.95` 代表每層最多剪 95%

這兩個參數非常重要：

- `clip-low` 太大：即使總壓縮率很低，每層仍會被強制剪一些
- `clip-high` 太大：某些層可能被剪太狠，造成精度掉太多

### Allocation 與 binary search

- `--allocation-strategy`
  - layer budget allocation 策略
  - 可用：`binary_search`, `global_fisher`
- `--binary-search-iterations`
  - binary search 最大迭代次數
- `--binary-search-low`
  - binary search scale 下界
- `--binary-search-high`
  - binary search scale 上界
- `--binary-search-tolerance`
  - binary search 收斂容忍值

## 6. Stage 1 與 Stage 2 超參數

### Stage 1: Uncertainty

- `--mc-samples`
  - MC dropout 的 forward 次數
- `--mc-dropout-p`
  - dropout 機率
- `--calib-batches`
  - calibration batches 數量
  - `0` 代表用完整 calibration loader
- `--calib-split-ratio`
  - 從 training data 中切出 calibration subset 的比例
- `--calib-samples`
  - 直接指定 calibration sample 數量
- `--calib-seed`
  - calibration subset 的隨機種子
- `--calib-exclude-from-train`
  - 是否把 calibration subset 從 training set 排除
- `--calib-use-val-transform`
  - calibration subset 是否使用 validation transform
- `--uncertainty-alpha`
  - Stage 1 allocation 的權重強度
- `--uncertainty-metric`
  - 可用：`mu_over_var`, `mu`, `var`, `inv_var`
- `--uncertainty-log`
  - 對 Stage 1 分數做 `log1p` 穩定化
- `--no-uncertainty-log`
  - 不做 `log1p`
- `--uncertainty-clip-percentile`
  - Stage 1 分數 percentile clipping
- `--uncertainty-var-floor`
  - Stage 1 variance floor

### Stage 2: Fisher / component scoring

- `--fisher-batches`
  - 計算 Fisher 時使用的 batch 數量
- `--use-fisher-scores`
  - 啟用 Fisher component ranking
- `--no-fisher-scores`
  - 關閉 Fisher ranking
- `--fisher-first-order-weight`
  - first-order 權重
- `--fisher-second-order-weight`
  - second-order 權重
- `--stage2-score-metric`
  - 可用：`fisher`, `taylor`, `hessian`, `energy`, `magnitude`

## 7. 微調相關超參數

- `--fine-tune-epochs`
  - 微調 epoch 數
- `--base-lr`
  - 基本 learning rate
- `--fine-tune-lr`
  - 微調 learning rate
- `--fine-tune-weight-decay`
  - 微調 weight decay
- `--layer-decay`
  - layer-wise lr decay
- `--warmup-epochs`
  - warmup epoch 數
- `--freeze-epoch`
  - 幾個 epoch 後凍結 lr
- `--freeze-lr`
  - 凍結後的 lr
- `--mixed-precision`
  - 啟用 AMP

## 8. 快速測試與完整版測試

### 快速測試

- `--train-max-batches 2`
- `--eval-max-batches 10`
- `--calib-batches 1`
- `--fisher-batches 1`
- `--fine-tune-epochs 1`

example：

```powershell
python main.py `
  --model-name resnet18 `
  --pretrained `
  --target-compression 0.4 `
  --fine-tune-epochs 1 `
  --batch-size 32 `
  --train-max-batches 2 `
  --eval-max-batches 10 `
  --calib-batches 1 `
  --fisher-batches 1 `
  --save-dir ./tmp_resnet18
```

### 完整版測試

- `--eval-max-batches 0`
- `--train-max-batches 0`
- `--calib-batches` 不要太小
- `--fisher-batches` 不要太小
- `--fine-tune-epochs` 用正式值

example：

```powershell
python main.py `
  --model-name resnet50 `
  --pretrained `
  --target-compression 0.4 `
  --fine-tune-epochs 90 `
  --batch-size 256 `
  --calib-batches 5 `
  --fisher-batches 100 `
  --pruning-clip-low 0.05 `
  --pruning-clip-high 0.95 `
  --eval-max-batches 0 `
  --train-max-batches 0 `
  --save-dir ./final_results/resnet50_40pr
```

## 9. Baseline testing

```powershell
python main.py `
  --model-name resnet18 `
  --pretrained `
  --target-compression 0.0 `
  --fine-tune-epochs 0 `
  --eval-max-batches 0 `
  --save-dir ./baseline_check/resnet18
```

Using `target_compression = 0.0` will skip pruning pipeline, keep baseline。

## 10. Validation 抽樣相關參數

- `--eval-max-batches`
  - `0` 代表完整 validation
  - `>0` 代表只評估部分 batch
- `--shuffle-val`
  - 打亂 validation loader
- `--no-shuffle-val`
  - 不打亂 validation loader
- `--random-eval-subset`
  - 當 `eval-max-batches > 0` 時，先隨機抽 validation subset
- `--no-random-eval-subset`
  - 關閉隨機 subset

如果是 quick test，建議保留 `random-eval-subset`，避免只取到排序後前面幾個 class。

## 11. Using your own model

Using own `PyTorch nn.Module`，用 Python API 而不是 CLI。

入口函式：

- `prune_and_finetune_model(...)`

位置：

- `main.py`

### 自訂 MLP 範例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from main import prune_and_finetune_model


class MyMLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


model = MyMLP()
train_loader: DataLoader = ...
val_loader: DataLoader = ...

result = prune_and_finetune_model(
    pruning_ratio=0.4,
    fine_tune_epochs=10,
    model_type="mlp",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    calib_loader=None,
    save_dir="./teacher_runs/my_mlp",
    min_rank=4,
)

final_model = result["final_model"]
```

### 自訂 CNN 範例

```python
import torch
import torch.nn as nn
from main import prune_and_finetune_model


class MyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


model = MyCNN(num_classes=10)
train_loader = ...
val_loader = ...

result = prune_and_finetune_model(
    pruning_ratio=0.5,
    fine_tune_epochs=5,
    model_type="cnn",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    save_dir="./teacher_runs/my_cnn",
    min_rank=4,
    pruning_clip_low=0.0,
    pruning_clip_high=0.9,
)
```

### 自訂模型時注意事項

- `model_type="cnn"`：會剪 `Conv2d` 和 `Linear`
- `model_type="mlp"`：只剪 `Linear`
- 模型中必須真的有 `Conv2d` 或 `Linear`
- `train_loader` 和 `val_loader` 必須自己準備
- `calib_loader` 不傳時會自動使用 `train_loader`
- 如果模型很小，建議把 `min_rank` 調低，例如 `2` 或 `4`

## 12. 輸出內容

每次執行會在 `save_dir` 下建立對應資料夾，包含：

- `config.json`
- `*_before_finetune.pth`
- `*ft.pth`
- `complete_pruning_report.txt`
- `visualizations/`
- `finetunemodel/`