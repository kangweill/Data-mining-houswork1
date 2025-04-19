import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, mean, stddev, lit
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import seaborn as sns

# 初始化 SparkSession，调整内存配置
spark = SparkSession.builder \
    .appName("Data Preprocessing and User Profiling") \
    .config("spark.executor.memory", "10g") \
    .config("spark.driver.memory", "10g") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.python.worker.timeout", "600") \
    .config("spark.driver.pyspark.python.worker.timeout", "600") \
    .getOrCreate()

# 设置 Matplotlib 字体，解决中文显示问题
rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 数据目录路径
# data_dir = "./1G_data"
# data_dir = "./10G_data_new"
data_dir = "./30G_data_new"

# 初始化一个空的 DataFrame，用于合并所有文件的数据
all_data = None

# 遍历目录下的所有 Parquet 文件
for file_name in os.listdir(data_dir):
    if file_name.endswith(".parquet") and not file_name.startswith("processed_"):
        file_path = os.path.join(data_dir, file_name)
        print(f"\n正在处理文件: {file_name}")

        file_start_time = time.time()  # 每个文件处理开始时间

        # 读取 Parquet 文件
        df = spark.read.parquet(file_path)

        # 唯一值检查
        if "id" in df.columns:
            dup_count = df.groupBy("id").count().filter(col("count") > 1).count()
            if dup_count > 0:
                print(f"发现重复ID: {dup_count} 条，已删除")
                df = df.dropDuplicates(["id"])

        # 缺失值统计
        missing_values = df.select([(count(when(col(c).isNull(), c)) / df.count() * 100).alias(c) for c in df.columns])
        print("\n缺失值统计:")
        missing_values.show()

        # 删除缺失值过多的列
        columns_to_drop = [c for c in df.columns if missing_values.select(c).collect()[0][0] > 50]
        if columns_to_drop:
            print("删除缺失值过多的列:", columns_to_drop)
            df = df.drop(*columns_to_drop)

        # 删除包含缺失值的行
        rows_before = df.count()
        df = df.dropna()
        rows_after = df.count()
        print(f"删除包含缺失值的行: 删除了 {rows_before - rows_after} 行, 剩余 {rows_after} 行")

        # 类别型字段异常值检查
        if "gender" in df.columns:
            valid_genders = ["男", "女", "male", "female", "M", "F"]
            invalid_gender_count = df.filter(~col("gender").isin(valid_genders)).count()
            if invalid_gender_count > 0:
                print(f"发现非法性别数据: {invalid_gender_count} 条，已删除")
                df = df.filter(col("gender").isin(valid_genders))

        if "country" in df.columns:
            country_null_count = df.filter(col("country").isNull()).count()
            if country_null_count > 0:
                print(f"发现缺失国家信息: {country_null_count} 条，已删除")
                df = df.filter(col("country").isNotNull())

        # 数值型字段合理性检查
        if "age" in df.columns:
            invalid_age_count = df.filter((col("age") < 0) | (col("age") > 120)).count()
            df = df.filter((col("age") >= 0) & (col("age") <= 120))
            if invalid_age_count > 0:
                print(f"已删除不合理的年龄数据: {invalid_age_count} 条")
        if "income" in df.columns:
            invalid_income_count = df.filter(col("income") < 0).count()
            df = df.filter(col("income") >= 0)
            if invalid_income_count > 0:
                print(f"已删除不合理的收入数据: {invalid_income_count} 条")

        # 异常值统计与处理（3σ原则）
        print("\n异常值统计:")
        numeric_columns = [c for c, t in df.dtypes if t in ["int", "double"]]
        for col_name in numeric_columns:
            stats = df.select(mean(col(col_name)).alias("mean"), stddev(col(col_name)).alias("stddev")).collect()
            mean_val, stddev_val = stats[0]["mean"], stats[0]["stddev"]
            if stddev_val is not None and stddev_val > 0:
                lower_bound = mean_val - 3 * stddev_val
                upper_bound = mean_val + 3 * stddev_val
                outlier_count = df.filter((col(col_name) < lower_bound) | (col(col_name) > upper_bound)).count()
                if outlier_count > 0:
                    print(f"{col_name} 异常值数量: {outlier_count}，已删除")
                    df = df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))

        # 合并到总数据集
        if all_data is None:
            all_data = df
        else:
            all_data = all_data.union(df)
        
        file_end_time = time.time()  # 每个文件处理结束时间
        print(f"文件 {file_name} 处理完成，耗时: {file_end_time - file_start_time:.2f}秒")

# 统一统计和分析
print("\n所有文件的数据已合并，开始统一统计和分析...")

# 用户画像分析
print("\n开始用户画像分析...")

# 年龄分布
age_distribution = all_data.select(mean("age").alias("mean"), stddev("age").alias("stddev")).collect()
print("\n年龄分布:")
print(age_distribution)

# 收入分布
income_distribution = all_data.select(mean("income").alias("mean"), stddev("income").alias("stddev")).collect()
print("\n收入分布:")
print(income_distribution)

# 性别比例
gender_counts = all_data.groupBy("gender").count()
print("\n性别比例:")
gender_counts.show()

# 活跃用户比例
active_counts = all_data.groupBy("is_active").count()
print("\n活跃用户比例:")
active_counts.show()

# 国家分布
country_counts = all_data.groupBy("country").count()
print("\n国家分布:")
country_counts.show()

print("\n用户画像分析完成。\n")

# 探索性分析和可视化
print("\n开始探索性分析和可视化...")

# 创建输出目录
os.makedirs(f"{data_dir}/visualizations", exist_ok=True)

# 可视化 1：年龄分布核密度估计图
plt.figure(figsize=(10, 6))
age_sample = all_data.select("age").sample(withReplacement=False, fraction=0.1).toPandas()
sns.kdeplot(age_sample['age'], shade=True, color="skyblue")
plt.title("年龄分布核密度估计图")
plt.xlabel("年龄")
plt.ylabel("密度")
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig(f"{data_dir}/visualizations/age_distribution_kde.png")
plt.close()

# 可视化 2：性别比例环形图
plt.figure(figsize=(8, 8))
gender_counts = all_data.filter(col("gender").isNotNull()).groupBy("gender").count().toPandas()
labels = gender_counts['gender'].tolist()
sizes = gender_counts['count'].tolist()

# 将性别统一为中文显示
label_map = {'male': '男性', 'female': '女性', 'M': '男性', 'F': '女性'}
labels = [label_map.get(x, x) for x in labels]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
        colors=["#66b3ff","#ff9999"], 
        wedgeprops=dict(width=0.4, edgecolor='w'))
plt.title("性别比例环形图")
plt.savefig(f"{data_dir}/visualizations/gender_distribution_donut.png")
plt.close()

# 可视化 3：收入分布小提琴图
plt.figure(figsize=(10, 6))
income_sample = all_data.select("income").sample(withReplacement=False, fraction=0.1).toPandas()
sns.violinplot(y=income_sample['income'], inner="quartile", palette="Set2")
plt.title("收入分布小提琴图")
plt.ylabel("收入（单位：元）")
plt.grid(True, axis='y', linestyle="--", alpha=0.7)
plt.savefig(f"{data_dir}/visualizations/income_distribution_violin.png")
plt.close()

print("可视化图表已保存至 visualizations 目录")