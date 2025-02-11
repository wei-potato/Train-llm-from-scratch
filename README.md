# Train-llm-from-scratch
从头开始训练一个LLM,模型大小为6B(可以通过配置参数根据自己的算力调节模型大小)，会使用deepspeed进行分布式训练
经过pretrain和sft
验证llm学习知识、理解语言、回答问题的能力
在每个步骤会有一个document解释代码和关键步骤，解析原理，方便学习
## 环境搭建
cuda 版本 12.1
依赖见requirements

## 分词器（Tokenizer）：
LLM分词器的构建方式有两种：
一种是自己构造词表并训练一个分词器custom tokenizers，自己训练一个分词器的代码在generate_tokenizer
另一种是选择开源模型训练好的分词器，例如ChatGLM2-6B，Llama2等。
本次使用ChatGLM2-6B的tokenizer

## 预训练

### 准备预训练数据 
预训练数据推荐
MNBVC  
地址：https://github.com/esbatmop/MNBVC  
数据集说明：里面有大佬整理的33T 预计达到GPT3.5的40T数据  
超大规模中文语料集，不但包括主流文化，也包括各个小众文化甚至火星文的数据。MNBVC数据集包括新闻、作文、小说、书籍、杂志、论文、台词、帖子、wiki、古诗、歌词、商品介绍、笑话、糗事、聊天记录等一切形式的纯文本中文数据。数据均来源于互联网收集，且在持续更新中。    
data_process/spark.py 使用spark将MNBVC数据处理成训练格式

WuDaoCorporaText  
地址：https://data.baai.ac.cn/details/WuDaoCorporaText  
数据集说明：WuDaoCorpora是北京智源人工智能研究院（智源研究院）构建的大规模、高质量数据集，用于支撑大模型训练研究。目前由文本、对话、图文对、视频文本对四部分组成，分别致力于构建微型语言世界、提炼对话核心规律、打破图文模态壁垒、建立视频文字关联，为大模型训练提供坚实的数据支撑。  

Awesome Chinese LLM  
地址：https://github.com/HqWu-HITCS/Awesome-Chinese-LLM  

SKYWORK  
数据集说明：天工开源的150B数据，质量很高  
地址：https://huggingface.co/datasets/Skywork/SkyPile-150B  
### 处理训练数据
说明：训练数据质量是影响模型性能最大的因素
#### 文本去重
不做会导致的后果：存在语义相似的训练数据会导致模型的生成重复，即重复生成同一个字
#### 其他数据处理工具
地址：https://github.com/aplmikex/deduplication_mnbvc  
作用： 语料去重


清除从不同来源提交给MNBVC项目的文件中，文件完全一致的。  
找出不同来源，不同渠道导致有细微差别的同一文件，并打上标签，如在不同盗版网站上的同一个小说。  
通过Word2Vec或Sentence2Vec将语料在句子层面进行向量聚类，找到重复率较高的句子，进行人工或深度学习进行分类，生成政治敏感，色情，广告等黑名单。  
清洗语料中较长的内部重复，或内部重复出现较多次的情况。    
网页语料清洗    
基于规则对commoncrawl数据做初步清洗，包括：  
过滤页眉页脚，标签栏等  
过滤乱码  
不相关的网页标识符  
过滤中文占比<70%的网页  
过滤中文字符少于10个的段落  
敏感词过滤带有脏话、色情、赌博等非法内容的页面  
从数据集中删除隐私内容，如身份证号、电话号码、qq号码、电子邮件地址  
截断段落的最后一句没有结束标点的句子  
繁体转简体  
删除了每个句子中的所有多余空格和标点  
用空格替换句子中连续空白字符（即选项卡、空格、不可见字符等）  
收集正负样本训练fasttext，对低质量文本进行分类，主要去除了赌博广告，色情广告等  
基于规则对wudao数据进行清洗，收集wudao正负样本作为低质量语料数据，目前优化中  
小说数据清洗，目前基本清洗完成  
基于规则进行清洗   
去除连载小说各小节前后与内容不相关的作者的碎碎念  
去除小说网站的插入广告  
去除特殊字符  
去除中文占比低的段落  
去除编码异常段落  

### SFT 数据推荐
alpaca-zh       
地址：https://huggingface.co/datasets/shibing624/alpaca-zh  
BelleGroup    
地址：https://huggingface.co/datasets/BelleGroup/train_1M_CN   
firefly      
地址：https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M    
COIG-CQIA    
地址：https://huggingface.co/datasets/m-a-p/COIG-CQIA  



## 详细文档
[预训练](documents/预训练原理.md)  
[全量微调](documents/SFT.md)
## 项目和书籍参考
https://github.com/DLLXW/baby-llama2-chinese  
https://github.com/LlamaFamily/Llama-Chinese  
https://github.com/SkyworkAI/Skywork
