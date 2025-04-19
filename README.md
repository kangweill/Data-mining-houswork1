# Data-mining-houswork1
# 数据预处理与用户画像分析

## 功能概述
- **数据清洗**：去重/缺失值处理/异常值过滤（3σ原则）
- **画像分析**：年龄分布/收入统计/性别比例/国家分布
- **可视化**：核密度图/环形饼图/小提琴图自动生成

## 环境要求
- PySpark 3.3+
- Python 3.8+
- 依赖库：matplotlib, seaborn,pyspark,time

## 使用步骤
1. 将Parquet文件放入`./10G_data_new/`目录或30G
2. 运行Spark任务：
```bash
python ./homework1.py
