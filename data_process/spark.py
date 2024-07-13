from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, struct
from pyspark.sql.types import StringType
import os
from multiprocessing import Pool
import multiprocessing

# 使用spark将
def process_jsonl_with_spark(input_file):
    # 创建SparkSession
    spark = SparkSession.builder \
        .appName("ReadProcessSaveJSONL") \
        .getOrCreate()

    # 定义自定义处理函数
    def add_suffix(data):
        if data is None:
            return "错误：输入数据为None"

        if not isinstance(data, list):
            return f"错误：输入数据不是列表，而是 {type(data)}"

        if len(data) == 0:
            return "错误：输入列表为空"

        contents = []
        for item in data:
            contents.append(item['内容'])
        combined_text = ' '.join(content.replace('\n', ' ') for content in contents)
        return combined_text

    # 注册UDF
    add_suffix_udf = udf(add_suffix, StringType())

    # 读取JSONL文件
    df = spark.read.json(input_file)

    # 应用自定义处理函数并创建新的DataFrame结构
    processed_df = df.withColumn("processed_paragraph", add_suffix_udf(col("段落"))) \
                     .select(struct(col("processed_paragraph").alias("text")).alias("_tmp")) \
                     .select("_tmp.*")

    # 生成输出文件名
    input_filename = os.path.basename(input_file)
    input_name, input_ext = os.path.splitext(input_filename)
    output_file = f"{input_name}_spark"

    # 将处理后的DataFrame保存为JSONL文件
    output_path = os.path.join(os.path.dirname(input_file), output_file)
    processed_df.coalesce(1).write.mode("overwrite").json(output_path)

    print(f"处理后的数据已保存到: {output_path}")

    # 停止SparkSession
    spark.stop()

    return output_path
def process_single_file(input_dir,file_name):
    input_file = os.path.join(input_dir, file_name)
    output_file = process_jsonl_with_spark(input_file)
    return f"输入文件: {file_name}, 输出文件: {output_file}"
# 使用示例
if __name__ == "__main__":

    input_dir="./data"
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    parallelism = 20
    if parallelism is None:
        parallelism = multiprocessing.cpu_count()

    # 创建一个进程池，进程数量为指定的并行度
    with Pool(processes=parallelism) as pool:
        # 并行调用process_single_file函数
        results = pool.starmap(process_single_file, [(input_dir, file) for file in jsonl_files])

    for result in results:
        print(result)