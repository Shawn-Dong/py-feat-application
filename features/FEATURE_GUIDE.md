# 🎯 Feature Extraction Guide for ML Training

## 📊 数据概览

你现在有 **3 个特征文件** 可以用于训练模型：

| 文件 | 粒度 | 样本数 | 特征数 | 适用模型 |
|------|------|--------|--------|----------|
| `combined_au_features.csv` | **Frame-level** (帧级别) | 25,920 帧 | 27 | LSTM, RNN, Transformer, CNN |
| `clip_features_simple.csv` | **Clip-level** (片段级别) | 42 clips | 138 | 传统ML (RF, XGBoost, SVM) |
| `clip_features_full.csv` | **Clip-level** (片段级别) | 42 clips | 562 | 传统ML + 深度学习 |

---

## 1️⃣ Frame-level Features (帧级别特征)
**文件**: `combined_au_features.csv`

### 原始特征 (27个)

#### Action Units (20个)
每个AU的激活强度，值域: [0.0, 1.0]

| AU | 描述 | 位置 |
|----|------|------|
| **AU01** | Inner Brow Raiser (内眉上抬) | 眉毛 |
| **AU02** | Outer Brow Raiser (外眉上抬) | 眉毛 |
| **AU04** | Brow Lowerer (皱眉) | 眉毛 |
| **AU05** | Upper Lid Raiser (上眼睑提升) | 眼睛 |
| **AU06** | Cheek Raiser (颧骨提升) | 脸颊 |
| **AU07** | Lid Tightener (眼睛眯起) | 眼睛 |
| **AU09** | Nose Wrinkler (皱鼻) | 鼻子 |
| **AU10** | Upper Lip Raiser (上唇上提) | 嘴部 |
| **AU11** | Nasolabial Deepener (鼻唇沟加深) | 嘴部 |
| **AU12** | Lip Corner Puller (嘴角上扬) | 嘴部 |
| **AU14** | Dimpler (酒窝) | 嘴部 |
| **AU15** | Lip Corner Depressor (嘴角下拉) | 嘴部 |
| **AU17** | Chin Raiser (下巴上提) | 下巴 |
| **AU20** | Lip Stretcher (嘴唇拉伸) | 嘴部 |
| **AU23** | Lip Tightener (嘴唇收紧) | 嘴部 |
| **AU24** | Lip Pressor (唇部压紧) | 嘴部 |
| **AU25** | Lips Part (嘴唇分开) | 嘴部 |
| **AU26** | Jaw Drop (下颌下降) | 下颌 |
| **AU28** | Lip Suck (吸嘴唇) | 嘴部 |
| **AU43** | Eyes Closed (闭眼) | 眼睛 |

#### Emotions (7个)
每个情绪的概率，值域: [0.0, 1.0]，总和≈1.0

- **anger** (愤怒)
- **disgust** (厌恶)
- **fear** (恐惧)
- **happiness** (快乐)
- **sadness** (悲伤)
- **surprise** (惊讶)
- **neutral** (中性)

### 使用场景
✅ 适合：**序列模型**
- LSTM, GRU, BiLSTM
- Transformer, Temporal Convolutional Networks (TCN)
- 1D CNN
- Attention-based models

📊 **数据格式**:
```python
# 形状: (25920, 27)
# 每一行 = 一帧的特征
# 可以reshape为: (42 clips, ~617 frames/clip, 27 features)
```

---

## 2️⃣ Clip-level Simple Features (片段级别简单特征)
**文件**: `clip_features_simple.csv`

### 特征结构 (138个)

对每个原始特征（20个AU + 7个emotion）提取 **5种统计量**:

| 统计量 | 描述 | 示例 |
|--------|------|------|
| **mean** | 平均值 | `AU01_mean` |
| **std** | 标准差 | `AU01_std` |
| **min** | 最小值 | `AU01_min` |
| **max** | 最大值 | `AU01_max` |
| **median** | 中位数 | `AU01_median` |

📐 **总特征数**: 27 features × 5 stats = **135 features** + 3 metadata = **138列**

### 示例特征列
```
AU01_mean, AU01_std, AU01_min, AU01_max, AU01_median
AU02_mean, AU02_std, AU02_min, AU02_max, AU02_median
...
anger_mean, anger_std, anger_min, anger_max, anger_median
happiness_mean, happiness_std, happiness_min, happiness_max, happiness_median
neutral_mean, neutral_std, neutral_min, neutral_max, neutral_median
```

### 使用场景
✅ 适合：**传统机器学习模型**
- Random Forest
- XGBoost, LightGBM, CatBoost
- SVM
- Logistic Regression
- KNN

💡 **优势**: 
- 快速训练
- 可解释性强
- 不需要大量数据

---

## 3️⃣ Clip-level Full Engineered Features (片段级别完整工程特征)
**文件**: `clip_features_full.csv`

### 特征分类 (562个)

#### A. 统计特征 (297个)
对每个原始特征提取 **11种统计量**:

| 特征类型 | 数量 | 描述 |
|----------|------|------|
| mean | 27 | 平均值 |
| median | 27 | 中位数 |
| std | 27 | 标准差 |
| min | 27 | 最小值 |
| max | 27 | 最大值 |
| range | 27 | 极差 (max - min) |
| q25 | 27 | 第25百分位数 |
| q75 | 27 | 第75百分位数 |
| iqr | 27 | 四分位距 (q75 - q25) |
| skew | 27 | 偏度 |
| kurtosis | 27 | 峰度 |

**示例**: `AU01_mean`, `AU01_std`, `AU01_skew`, `AU01_kurtosis`, `anger_median`, `happiness_iqr`

#### B. 时序特征 (216个)
对每个原始特征提取 **8种动态特征**:

| 特征类型 | 描述 |
|----------|------|
| **mean_change** | 一阶导数的均值 (速度) |
| **std_change** | 一阶导数的标准差 |
| **abs_change** | 绝对变化的均值 |
| **mean_accel** | 二阶导数的均值 (加速度) |
| **std_accel** | 二阶导数的标准差 |
| **num_peaks** | 峰值数量 |
| **peak_prominence_mean** | 峰值显著性均值 |
| **trend_slope** | 线性趋势斜率 |
| **trend_r2** | 趋势拟合度 (R²) |

**示例**: `AU01_mean_change`, `AU01_num_peaks`, `AU01_trend_slope`, `happiness_mean_accel`

💡 这些特征可以捕捉：
- 表情变化速度
- 表情变化平滑度
- 表情强度趋势（上升/下降）
- 表情波动频率

#### C. AU组合特征 (8个)
基于面部区域的组合特征:

| 特征 | 描述 |
|------|------|
| **upper_face_mean** | 上半脸AU平均激活 (AU01,02,04,05,06,07) |
| **upper_face_std** | 上半脸AU激活标准差 |
| **upper_face_max** | 上半脸AU最大激活 |
| **lower_face_mean** | 下半脸AU平均激活 (AU09-28) |
| **lower_face_std** | 下半脸AU激活标准差 |
| **lower_face_max** | 下半脸AU最大激活 |
| **num_high_aus** | 高激活AU数量 (>0.5) |
| **au_diversity** | AU多样性 (激活AU数量, >0.3) |

💡 这些特征可以区分：
- 眉眼表情 vs 嘴部表情
- 表情复杂度
- 微表情 vs 强烈表情

#### D. 情绪特征 (10个)
专门针对情绪的高级特征:

| 特征 | 描述 |
|------|------|
| **freq_anger_dominant** | 愤怒为主导情绪的帧占比 |
| **freq_disgust_dominant** | 厌恶为主导情绪的帧占比 |
| **freq_fear_dominant** | 恐惧为主导情绪的帧占比 |
| **freq_happiness_dominant** | 快乐为主导情绪的帧占比 |
| **freq_sadness_dominant** | 悲伤为主导情绪的帧占比 |
| **freq_surprise_dominant** | 惊讶为主导情绪的帧占比 |
| **freq_neutral_dominant** | 中性为主导情绪的帧占比 |
| **max_emotion_intensity** | 最大情绪强度均值 |
| **emotion_variability** | 情绪变化幅度 |
| **expressiveness** | 表达性 (非中性帧占比) |

💡 这些特征可以识别：
- 情绪稳定性
- 情绪多样性
- 表情丰富程度

### 使用场景
✅ 适合：**所有类型的模型**
- 传统ML: RF, XGBoost, SVM (选择重要特征)
- 深度学习: MLP, AutoEncoder
- 特征选择后用于任何模型

💡 **优势**: 
- 特征丰富，信息量大
- 包含时序动态信息
- 可以做特征选择/降维

---

## 🎯 建议的训练策略

### 方案1: 传统机器学习 (适合小数据集)
```python
# 使用 clip_features_simple.csv 或 clip_features_full.csv
X = df.drop(['sample_id', 'video_id', 'clip_id'], axis=1)
y = your_labels  # 你的目标变量

# 模型选择
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

model = RandomForestClassifier(n_estimators=100)
model = XGBClassifier(n_estimators=100)
```

**推荐特征组合**:
- 开始: `clip_features_simple.csv` (138 features)
- 进阶: `clip_features_full.csv` 的统计特征 (297 features)
- 高级: `clip_features_full.csv` 全部特征 + 特征选择

### 方案2: 深度序列模型 (适合捕捉时序信息)
```python
# 使用 combined_au_features.csv
# 将数据reshape为序列格式
# Shape: (num_clips, seq_length, num_features)

import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=27, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, 27)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])
```

### 方案3: 混合方法
```python
# 1. 用 combined_au_features.csv 训练LSTM提取序列特征
# 2. 将LSTM的hidden states作为新特征
# 3. 结合 clip_features_full.csv 的统计特征
# 4. 用XGBoost做最终预测
```

---

## 🔍 特征选择建议

### 如果数据量小 (< 1000 samples)
1. 使用 `clip_features_simple.csv` (138 features)
2. 或从 `clip_features_full.csv` 中选择最重要的特征

### 如果数据量中等 (1000-10000 samples)
1. 使用 `clip_features_full.csv` 全部特征
2. 做特征重要性分析
3. 保留 top 100-200 重要特征

### 如果数据量大 (> 10000 samples)
1. 可以使用 `combined_au_features.csv` 训练深度学习模型
2. 或者用全部工程特征 + 特征选择

---

## 📊 特征重要性分析示例

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 加载特征
df = pd.read_csv('features/clip_features_full.csv')
X = df.drop(['sample_id', 'video_id', 'clip_id'], axis=1)
y = your_labels

# 训练模型
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 获取特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# 查看top 20重要特征
print(feature_importance.head(20))

# 可视化
feature_importance.head(20).plot(x='feature', y='importance', kind='barh')
plt.show()
```

---

## 💡 最佳实践

### 1. **从简单开始**
- ✅ 先用 `clip_features_simple.csv`
- ✅ 建立baseline模型
- ✅ 理解数据分布

### 2. **逐步增加复杂度**
- ✅ 尝试 `clip_features_full.csv`
- ✅ 做特征选择
- ✅ 尝试不同的模型

### 3. **考虑时序信息**
- ✅ 如果任务需要捕捉动态变化，用 `combined_au_features.csv`
- ✅ 使用LSTM/Transformer等序列模型
- ✅ 可以结合静态特征和序列特征

### 4. **特征工程迭代**
- ✅ 分析哪些特征对你的任务最重要
- ✅ 创建领域特定的特征
- ✅ 做特征交互 (feature interactions)

---

## 📝 示例代码

### 加载和使用简单特征
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 加载数据
df = pd.read_csv('features/clip_features_simple.csv')

# 分离特征和标签
X = df.drop(['sample_id', 'video_id', 'clip_id'], axis=1)
y = your_labels  # 你需要提供标签

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 使用序列特征
```python
import pandas as pd
import numpy as np

# 加载frame-level数据
df = pd.read_csv('features/combined_au_features.csv')

# 提取特征列
feature_cols = [col for col in df.columns if col.startswith('AU') or 
                col in ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']]

# 为每个clip准备序列数据
sequences = []
labels = []

for sample_id in df['sample_id'].unique():
    clip_df = df[df['sample_id'] == sample_id]
    sequence = clip_df[feature_cols].values  # shape: (num_frames, 27)
    sequences.append(sequence)
    labels.append(your_label_for_this_clip)

# 转换为numpy数组（需要padding到相同长度）
from tensorflow.keras.preprocessing.sequence import pad_sequences
X = pad_sequences(sequences, padding='post', dtype='float32')
y = np.array(labels)

# 现在可以用于训练LSTM等模型
```

---

## 🎓 总结

| 特征文件 | 样本数 | 特征数 | 最佳用途 |
|----------|--------|--------|----------|
| **combined_au_features.csv** | 25,920 | 27 | 序列模型 (LSTM, Transformer) |
| **clip_features_simple.csv** | 42 | 138 | 快速原型 + 传统ML |
| **clip_features_full.csv** | 42 | 562 | 完整特征工程 + 高级模型 |

**建议workflow**:
1. 从 `clip_features_simple.csv` 开始建立baseline
2. 如果性能不够，尝试 `clip_features_full.csv`
3. 如果需要捕捉时序动态，使用 `combined_au_features.csv`
4. 根据特征重要性做特征选择
5. 尝试ensemble多个模型

祝训练顺利！🚀

