import sentencepiece as spm

# 训练 SentencePiece 模型
spm.SentencePieceTrainer.train(
    input='wiki_zh_first_20.jsonl',  # 替换为你的训练数据文件路径
    model_prefix='tokenizer',    # 模型前缀
    vocab_size=13000,            # 词汇表大小
    model_type='bpe',            # 使用 BPE 模型
    pad_id=0,                    # Padding 的 ID
    unk_id=1,                    # Unknown 的 ID
    bos_id=2,                    # Begin of Sentence 的 ID
    eos_id=3,                    # End of Sentence 的 ID
    user_defined_symbols=["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]  # 自定义符号
)
