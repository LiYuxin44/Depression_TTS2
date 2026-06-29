"""
独立的情绪分析脚本（优化版 - 使用异步处理加速）
从训练集 filelist 加载文本，使用 deepseek 进行情绪分析，分离为 Healthy_Text_Pool 和 Depressed_Text_Pool
"""
import datetime as dt
import json
import re
import os
import random
import time
import asyncio
from pathlib import Path
from tqdm.auto import tqdm
from collections import Counter, defaultdict
import math
import csv

# 导入 OpenAI 客户端用于情绪分析（异步版本）
try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Error: openai library not installed. Please install it first.")
    exit(1)

# ───────────────────────────  配置参数  ───────────────────────────────────

# 训练集 filelist（包含音频路径和转录文本）
TRAIN_FILELIST = "/home/i-liyuxin/Depression_TTS/matcha/data/daic_filelist_train_22k.txt"

# 文本筛选参数（用于去掉太短的文本）
TEXT_MIN_CHARS = 20     # 最小字符数（清洗后计算）
TEXT_MIN_WORDS = 5      # 最少词数
TEXT_MAX_CHARS = 2000    # 最大字符数上限

# 情绪分析配置（优化版 - 增加并发以加速）
SENTIMENT_MODEL = "deepseek-r1-volce"  # 情绪分析模型
SENTIMENT_MAX_CONCURRENT = 100  # 最大并发数（异步处理，可设置更高）
SENTIMENT_MAX_RETRIES = 60  # 最大重试次数
SENTIMENT_INITIAL_DELAY = 1.0  # 初始延迟（秒）
SENTIMENT_BATCH_SIZE = 1000  # 批处理大小：每次处理一批，处理完再处理下一批（增大以提升效率）
SENTIMENT_SEMAPHORE_LIMIT = 100  # 信号量限制，控制同时进行的请求数

# OpenAI 客户端配置（用于情绪分析）
OPENAI_API_KEY = "ak-57d1efgh23i9jkl64mno32pqrs18tuv4k6"
OPENAI_BASE_URL = "https://models-proxy.stepfun-inc.com/v1"
OPENAI_TIMEOUT = 600

# 情绪分析文本池保存路径（原始池，保持不动，避免被本脚本覆盖）
SENTIMENT_POOL_DIR = "/data/depression_tts/synthese_data/sentiment_pool"
SENTIMENT_POOL_HEALTHY_FILE = os.path.join(SENTIMENT_POOL_DIR, "healthy_text_pool.json")
SENTIMENT_POOL_DEPRESSED_FILE = os.path.join(SENTIMENT_POOL_DIR, "depressed_text_pool.json")
SENTIMENT_POOL_METADATA_FILE = os.path.join(SENTIMENT_POOL_DIR, "metadata.json")

# 本脚本的统计与新文本池输出目录（不会覆盖原始文本池）
SENTIMENT_BIAS_DIR = os.path.join(SENTIMENT_POOL_DIR, "bias_analysis")
SENTIMENT_BIAS_HEALTHY_FILE = os.path.join(SENTIMENT_BIAS_DIR, "healthy_text_pool_bias.json")
SENTIMENT_BIAS_DEPRESSED_FILE = os.path.join(SENTIMENT_BIAS_DIR, "depressed_text_pool_bias.json")
SENTIMENT_BIAS_METADATA_FILE = os.path.join(SENTIMENT_BIAS_DIR, "metadata_bias.json")

# 受试者级别标签文件（PHQ8_Binary 作为是否抑郁标签）
SUBJECT_LABEL_CSV = "/home/i-liyuxin/test/data/Label/train_cleaned.csv"


# ───────────────────────────  辅助函数  ───────────────────────────────────

def clean_text(text: str) -> str:
    """清理文本"""
    text = re.sub(r'\s+', ' ', str(text)).strip()
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\'\"]', '', text)
    return text


def normalize_sentiment_label(sentiment: str) -> str:
    """将模型返回的情绪结果归一化为 positive/neutral/negative"""
    s = (sentiment or "").lower()
    if any(k in s for k in ["积极", "positive", "pos"]):
        return "positive"
    if any(k in s for k in ["中性", "neutral", "neu"]):
        return "neutral"
    if any(k in s for k in ["消极", "negative", "neg"]):
        return "negative"
    return "unknown"


def load_subject_labels(csv_path: str) -> dict:
    """从受试者级别 CSV 加载标签（Participant_ID -> healthy/depressed）
    
    使用 PHQ8_Binary 列：
        0 -> healthy
        1 -> depressed
    """
    labels = {}
    if not os.path.exists(csv_path):
        print(f"Warning: subject label csv not found: {csv_path}")
        return labels
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row.get("Participant_ID", "")).strip()
            if not pid:
                continue
            try:
                binary = int(row.get("PHQ8_Binary", 0))
            except Exception:
                binary = 0
            labels[pid] = "depressed" if binary == 1 else "healthy"
    print(f"Loaded {len(labels)} subject labels from {csv_path}")
    return labels


def parse_subject_id_from_path(path: str) -> str:
    """根据 wav 路径解析出 subject id（例如 303_21.wav -> 303）"""
    base = os.path.basename(path)
    name, _ext = os.path.splitext(base)
    # 取第一个下划线前的数字部分
    first = name.split("_", 1)[0]
    # 保证是纯数字
    digits = "".join(ch for ch in first if ch.isdigit())
    return digits or first


def infer_label_from_subject(path: str, subject_labels: dict) -> str:
    """使用受试者级别标签推断 utterance 的诊断标签"""
    sid = parse_subject_id_from_path(path)
    return subject_labels.get(sid, "unknown")


def count_words(text: str) -> int:
    """统计词数"""
    return len(re.findall(r'\b\w+\b', str(text)))


def is_text_valid(text: str) -> bool:
    """验证文本有效性"""
    s = clean_text(text)
    lower = s.lower()
    if lower.startswith('http') or lower.startswith('www.'):
        return False
    num_chars = len(s)
    num_words = count_words(s)
    return (num_chars >= TEXT_MIN_CHARS and num_chars <= TEXT_MAX_CHARS and num_words >= TEXT_MIN_WORDS)


async def request_sentiment_analysis_async(text: str, client: AsyncOpenAI,
                                           model: str = SENTIMENT_MODEL, 
                                           max_retries: int = SENTIMENT_MAX_RETRIES, 
                                           initial_delay: float = SENTIMENT_INITIAL_DELAY) -> str:
    """使用 deepseek 进行文本情绪分析（异步版本）"""
    prompt = f"""你是一名专业的文本情绪分析专家。请分析以下文本的情感倾向。

文本内容：
{text}

请判断这段文本的情感倾向，只回答以下三种之一（不要添加任何其他文字）：
1. 积极 - 文本表达积极、乐观、正面的情绪
2. 中性 - 文本表达中性、客观、无明显情绪倾向
3. 消极 - 文本表达消极、悲观、负面的情绪

请只回答：积极、中性 或 消极（三个字之一）"""
    
    for attempt in range(max_retries):
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            result = completion.choices[0].message.content or ""
            return result.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = min(initial_delay * attempt + random.uniform(0, 1), 30)
            await asyncio.sleep(delay)


async def process_single_text_for_sentiment_async(text_info: tuple, client: AsyncOpenAI, 
                                                  semaphore: asyncio.Semaphore) -> dict:
    """处理单个文本：进行情绪分析并分类（异步版本）
    
    Args:
        text_info: (text, line_idx, src_path_label) 元组
        client: AsyncOpenAI 客户端
        semaphore: 信号量，用于控制并发数
    
    Returns:
        dict: {"text": text, "sentiment": sentiment, "sentiment_label": str,
               "category": category, "src_label": src_label} 或 None
    """
    text, line_idx, src_label = text_info
    
    # 验证文本有效性
    if not is_text_valid(text):
        return None
    
    # 清理文本
    cleaned_text = clean_text(text)
    
    # 使用 deepseek 进行情绪分析（使用信号量控制并发）
    try:
        async with semaphore:  # 使用信号量控制并发数
            sentiment_result = await request_sentiment_analysis_async(cleaned_text, client)
        
        # 归一化情绪标签
        sentiment_label = normalize_sentiment_label(sentiment_result)
        
        # 判断情绪类别
        is_positive_or_neutral = sentiment_label in ("positive", "neutral")
        is_negative = sentiment_label == "negative"
        
        category = None
        if is_positive_or_neutral:
            category = "healthy"  # 积极/中性文本 -> Healthy_Text_Pool
        elif is_negative:
            category = "depressed"  # 消极文本 -> Depressed_Text_Pool
        
        if category:
            return {
                "text": cleaned_text,
                "sentiment": sentiment_result,
                "sentiment_label": sentiment_label,
                "category": category,
                "src_label": src_label
            }
        
        return {
            "text": cleaned_text,
            "sentiment": sentiment_result,
            "sentiment_label": sentiment_label,
            "category": None,
            "src_label": src_label
        }
        
    except Exception as e:
        if line_idx < 10:  # 只打印前10个错误
            print(f"  Warning: Error analyzing sentiment for line {line_idx}: {e}")
        return None


# ───────────────────────────  主函数  ───────────────────────────────────

async def analyze_sentiment_pool_async():
    """从训练集 filelist 加载文本，使用情绪分析（deepseek）分离为 Healthy_Text_Pool 和 Depressed_Text_Pool（异步版本）"""
    print("=== 情绪分析脚本（异步优化版）：从训练集 filelist 分析文本情绪 ===")
    
    # 检查是否存在本脚本生成的 bias_analysis 文本池（只对新的结果做是否重跑确认）
    if os.path.exists(SENTIMENT_BIAS_HEALTHY_FILE) and os.path.exists(SENTIMENT_BIAS_DEPRESSED_FILE):
        print(f"⚠️  发现已存在的 bias_analysis 文本池文件:")
        print(f"  Healthy (bias): {SENTIMENT_BIAS_HEALTHY_FILE}")
        print(f"  Depressed (bias): {SENTIMENT_BIAS_DEPRESSED_FILE}")
        response = input("是否重新分析并覆盖 bias_analysis 结果？(y/n): ").strip().lower()
        if response != 'y':
            print("已取消，退出。")
            return
    
    healthy_texts = []
    depressed_texts = []
    stat_records = []  # 收集所有结果用于统计

    # 加载受试者级别标签
    subject_labels = load_subject_labels(SUBJECT_LABEL_CSV)
    
    if not os.path.exists(TRAIN_FILELIST):
        raise FileNotFoundError(f"Training filelist not found: {TRAIN_FILELIST}")
    
    # 读取 filelist
    print(f"\n读取 filelist: {TRAIN_FILELIST}")
    with open(TRAIN_FILELIST, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"  Total lines in filelist: {len(lines)}")
    
    # 提取所有文本
    text_list = []
    for line_idx, line in enumerate(lines):
        try:
            line = line.strip()
            if not line:
                continue
            
            # 解析格式：path|text
            if '|' not in line:
                continue
            
            parts = line.split('|', 1)
            if len(parts) < 2:
                continue
            
            wav_path = parts[0].strip()
            text = parts[1].strip()
            src_label = infer_label_from_subject(wav_path, subject_labels)
            text_list.append((text, line_idx, src_label))
            
        except Exception as e:
            if line_idx < 10:
                print(f"  Warning: Error processing line {line_idx}: {e}")
            continue
    
    print(f"  Extracted {len(text_list)} valid texts")
    
    # 初始化异步 OpenAI 客户端
    print(f"\n初始化异步 OpenAI 客户端...")
    print(f"  Model: {SENTIMENT_MODEL}")
    print(f"  Max concurrent: {SENTIMENT_MAX_CONCURRENT}")
    print(f"  Semaphore limit: {SENTIMENT_SEMAPHORE_LIMIT}")
    print(f"  Batch size: {SENTIMENT_BATCH_SIZE}")
    
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=OPENAI_TIMEOUT)
        print("✓ AsyncOpenAI 客户端初始化成功")
    except Exception as e:
        print(f"✗ Failed to initialize AsyncOpenAI client: {e}")
        raise
    
    # 创建信号量以控制并发数
    semaphore = asyncio.Semaphore(SENTIMENT_SEMAPHORE_LIMIT)
    
    # 使用情绪分析分类文本（异步批处理模式）
    print(f"\n开始情绪分析（异步处理）...")
    total_batches = (len(text_list) + SENTIMENT_BATCH_SIZE - 1) // SENTIMENT_BATCH_SIZE
    
    print(f"  Processing {len(text_list)} texts in {total_batches} batches (batch_size={SENTIMENT_BATCH_SIZE})")
    
    # 创建进度条
    pbar = tqdm(total=len(text_list), desc="Processing texts", unit="text")
    
    # 分批处理，每次处理一批，处理完再处理下一批（减少内存占用）
    for batch_idx in range(total_batches):
        start_idx = batch_idx * SENTIMENT_BATCH_SIZE
        end_idx = min(start_idx + SENTIMENT_BATCH_SIZE, len(text_list))
        batch_texts = text_list[start_idx:end_idx]
        
        print(f"\n  Processing batch {batch_idx + 1}/{total_batches} (texts {start_idx + 1}-{end_idx})...")
        
        # 创建当前批次的所有任务
        tasks = [
            process_single_text_for_sentiment_async(text_info, client, semaphore)
            for text_info in batch_texts
        ]
        
        # 并发执行当前批次的所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for result in results:
            if isinstance(result, Exception):
                pbar.update(1)
                continue  # 跳过异常结果
            
            if result:
                stat_records.append(result)
                if result.get("category") == "healthy":
                    healthy_texts.append(result["text"])
                elif result.get("category") == "depressed":
                    depressed_texts.append(result["text"])
            
            pbar.update(1)
        
        # 批次完成后的统计
        print(f"    Batch {batch_idx + 1}/{total_batches} completed: healthy={len(healthy_texts)}, depressed={len(depressed_texts)}")
    
    pbar.close()
    
    # 最终统计信息
    print(f"\n✓ 分析完成:")
    print(f"    Total texts analyzed: {len(text_list)}")
    print(f"    Final: healthy={len(healthy_texts)}, depressed={len(depressed_texts)}")
    
    # 去重
    healthy_texts = list(dict.fromkeys(healthy_texts))  # 保持顺序的去重
    depressed_texts = list(dict.fromkeys(depressed_texts))
    
    print(f"\n去重后:")
    print(f"  Healthy_Text_Pool (积极/中性文本): {len(healthy_texts)} texts")
    print(f"  Depressed_Text_Pool (消极文本): {len(depressed_texts)} texts")
    
    if len(healthy_texts) == 0 or len(depressed_texts) == 0:
        print(f"⚠️  Warning: One or both text pools are empty!")
    
    # 统计与显著性分析
    print(f"\n📊 统计情绪分布与显著性检验...")
    total_analyzed = len(stat_records)
    sentiment_counts = Counter([r.get("sentiment_label", "unknown") for r in stat_records])
    sentiment_ratios = {
        k: (sentiment_counts[k] / total_analyzed) if total_analyzed else 0.0
        for k in sentiment_counts
    }
    
    # 标签（来源）× 情绪 分布表
    cross_table = defaultdict(Counter)
    for r in stat_records:
        src = r.get("src_label", "unknown") or "unknown"
        cross_table[src][r.get("sentiment_label", "unknown")] += 1
    
    # 卡方检验：仅比较 healthy vs depressed，维度为 (pos+neu) vs neg
    hp_posneu = cross_table["healthy"]["positive"] + cross_table["healthy"]["neutral"]
    hp_neg = cross_table["healthy"]["negative"]
    dp_posneu = cross_table["depressed"]["positive"] + cross_table["depressed"]["neutral"]
    dp_neg = cross_table["depressed"]["negative"]
    chi_result = chi_square_test_2x2((hp_posneu, hp_neg), (dp_posneu, dp_neg))
    
    stats_report = {
        "total_texts_analyzed": total_analyzed,
        "sentiment_counts": dict(sentiment_counts),
        "sentiment_ratios": sentiment_ratios,
        "cross_table_counts": {k: dict(v) for k, v in cross_table.items()},
        "healthy_posneu_vs_neg": {"pos_or_neu": hp_posneu, "neg": hp_neg},
        "depressed_posneu_vs_neg": {"pos_or_neu": dp_posneu, "neg": dp_neg},
        "chi_square": chi_result,
    }
    
    # 保存文本池与统计报告（写到 bias_analysis 目录，避免覆盖原有文本池）
    print(f"\n💾 Saving sentiment bias analysis to {SENTIMENT_BIAS_DIR}...")
    try:
        os.makedirs(SENTIMENT_BIAS_DIR, exist_ok=True)
        
        # 保存本次分析得到的文本池（不会覆盖原始池）
        with open(SENTIMENT_BIAS_HEALTHY_FILE, 'w', encoding='utf-8') as f:
            json.dump(healthy_texts, f, indent=2, ensure_ascii=False)
        with open(SENTIMENT_BIAS_DEPRESSED_FILE, 'w', encoding='utf-8') as f:
            json.dump(depressed_texts, f, indent=2, ensure_ascii=False)
        
        # 保存元数据
        metadata = {
            "timestamp": dt.datetime.now().isoformat(),
            "model": SENTIMENT_MODEL,
            "filelist_path": TRAIN_FILELIST,
            "healthy_text_count": len(healthy_texts),
            "depressed_text_count": len(depressed_texts),
            "total_texts_analyzed": len(text_list),
            "use_sentiment_analysis": True,
            "max_concurrent": SENTIMENT_MAX_CONCURRENT,
            "semaphore_limit": SENTIMENT_SEMAPHORE_LIMIT
        }
        with open(SENTIMENT_BIAS_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 保存统计报告
        stats_json_path = os.path.join(SENTIMENT_BIAS_DIR, "sentiment_stats.json")
        stats_csv_path = os.path.join(SENTIMENT_BIAS_DIR, "sentiment_stats.csv")
        
        with open(stats_json_path, 'w', encoding='utf-8') as f:
            json.dump(stats_report, f, indent=2, ensure_ascii=False)
        
        # CSV 报告：按来源标签行展示
        labels = sorted(cross_table.keys())
        with open(stats_csv_path, 'w', encoding='utf-8') as f:
            header = "label,positive,neutral,negative,unknown,total,pos_ratio,neg_ratio\n"
            f.write(header)
            for lbl in labels:
                pos = cross_table[lbl]["positive"]
                neu = cross_table[lbl]["neutral"]
                neg = cross_table[lbl]["negative"]
                unk = cross_table[lbl]["unknown"]
                total = pos + neu + neg + unk
                pos_ratio = (pos + neu) / total if total else 0.0
                neg_ratio = neg / total if total else 0.0
                f.write(f"{lbl},{pos},{neu},{neg},{unk},{total},{pos_ratio:.6f},{neg_ratio:.6f}\n")
            # 总计
            pos = sentiment_counts.get("positive", 0)
            neu = sentiment_counts.get("neutral", 0)
            neg = sentiment_counts.get("negative", 0)
            unk = sentiment_counts.get("unknown", 0)
            total = total_analyzed
            pos_ratio = (pos + neu) / total if total else 0.0
            neg_ratio = neg / total if total else 0.0
            f.write(f"TOTAL,{pos},{neu},{neg},{unk},{total},{pos_ratio:.6f},{neg_ratio:.6f}\n")
            # 卡方结果
            f.write(f"CHI_SQUARE,stat={chi_result['stat']:.6f},p_value={chi_result['p_value']:.6f},df={chi_result['df']},,,,\n")
        
        # 绘制散点图
        scatter_path = os.path.join(SENTIMENT_BIAS_DIR, "sentiment_scatter.png")
        plot_sentiment_scatter(stat_records, scatter_path)
        
        print(f"✓ Sentiment bias analysis saved successfully")
        print(f"  Healthy texts (bias): {SENTIMENT_BIAS_HEALTHY_FILE}")
        print(f"  Depressed texts (bias): {SENTIMENT_BIAS_DEPRESSED_FILE}")
        print(f"  Metadata (bias): {SENTIMENT_BIAS_METADATA_FILE}")
        print(f"  Stats JSON: {stats_json_path}")
        print(f"  Stats CSV: {stats_csv_path}")
        print(f"  Scatter PNG: {scatter_path}")
    except Exception as e:
        print(f"  ✗ Failed to save sentiment pools: {e}")
        raise
    
    print(f"\n{'='*60}")
    print(f"=== 情绪分析完成 ===")
    print(f"Healthy_Text_Pool: {len(healthy_texts)} texts")
    print(f"Depressed_Text_Pool: {len(depressed_texts)} texts")
    print(f"Sentiment counts: {dict(sentiment_counts)}")
    print(f"Chi-square (pos/neu vs neg, healthy vs depressed): stat={chi_result['stat']:.4f}, p={chi_result['p_value']:.4g}, df={chi_result['df']}")
    print(f"{'='*60}")


def chi_square_test_2x2(healthy_posneg: tuple, depressed_posneg: tuple):
    """对 2x2 列联表（healthy vs depressed, pos/neu vs neg）进行卡方检验（无外部依赖）
    
    Args:
        healthy_posneg: (pos_or_neu_count, neg_count)
        depressed_posneg: (pos_or_neu_count, neg_count)
    Returns:
        dict: {"stat": float, "p_value": float, "df": 1}
    """
    hp, hn = healthy_posneg
    dp, dn = depressed_posneg
    total = hp + hn + dp + dn
    if total == 0:
        return {"stat": 0.0, "p_value": 1.0, "df": 1}
    
    row_totals = [hp + hn, dp + dn]
    col_totals = [hp + dp, hn + dn]
    
    # 期望频数
    expected_hp = row_totals[0] * col_totals[0] / total if total else 0
    expected_hn = row_totals[0] * col_totals[1] / total if total else 0
    expected_dp = row_totals[1] * col_totals[0] / total if total else 0
    expected_dn = row_totals[1] * col_totals[1] / total if total else 0
    
    # 防止除零
    expected = [expected_hp, expected_hn, expected_dp, expected_dn]
    observed = [hp, hn, dp, dn]
    stat = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            stat += (o - e) ** 2 / e
    # 自由度 (r-1)*(c-1) = 1
    df = 1
    # 近似 p 值：使用正态近似的生存函数（简化版）
    # 这里用 math.erfc 近似卡方 df=1 的尾部概率
    # p = 2 * (1 - Phi(sqrt(stat)))，Phi ~ 0.5 * erfc(-x/sqrt(2))
    z = math.sqrt(max(stat, 0))
    p_value = math.erfc(z / math.sqrt(2))
    return {"stat": stat, "p_value": p_value, "df": df}


def plot_sentiment_scatter(stat_records, output_path: str):
    """绘制情绪极性 × 诊断标签的散点分布图
    
    横坐标：semantic polarity（positive, neutral, negative）
    纵坐标：diagnosis label（depressed, healthy）
    点颜色：depressed = 红色，healthy = 蓝色，其它/unknown = 灰色
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib 未安装，无法绘制散点图。")
        return
    
    if not stat_records:
        print("No records to plot, skip scatter plot.")
        return
    
    polarity_to_x = {
        "positive": 0,
        "neutral": 1,
        "negative": 2,
        "unknown": 3,
    }
    label_to_y = {
        "depressed": 0,
        "healthy": 1,
        "unknown": 0.5,
    }
    
    xs, ys, colors = [], [], []
    for r in stat_records:
        pol = r.get("sentiment_label", "unknown") or "unknown"
        lbl = r.get("src_label", "unknown") or "unknown"
        if pol not in polarity_to_x:
            pol = "unknown"
        if lbl not in label_to_y:
            lbl = "unknown"
        x = polarity_to_x[pol] + (random.random() - 0.5) * 0.15  # 轻微抖动
        y = label_to_y[lbl] + (random.random() - 0.5) * 0.05
        if lbl == "depressed":
            c = "red"
        elif lbl == "healthy":
            c = "blue"
        else:
            c = "gray"
        xs.append(x)
        ys.append(y)
        colors.append(c)
    
    plt.figure(figsize=(8, 4))
    plt.scatter(xs, ys, c=colors, alpha=0.6, s=8, edgecolors="none")
    
    plt.xticks(
        [0, 1, 2],
        ["positive", "neutral", "negative"],
        fontsize=10
    )
    plt.yticks(
        [0, 1],
        ["depressed", "healthy"],
        fontsize=10
    )
    plt.xlabel("Semantic polarity", fontsize=12)
    plt.ylabel("Diagnosis label", fontsize=12)
    plt.title("Utterance distribution by sentiment and diagnosis", fontsize=13)
    # 简单图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='depressed', markerfacecolor='red', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='healthy', markerfacecolor='blue', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='unknown/other', markerfacecolor='gray', markersize=6),
    ]
    plt.legend(handles=legend_elements, loc="best", fontsize=9)
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=200)
        print(f"✓ Sentiment scatter plot saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save scatter plot: {e}")
    finally:
        plt.close()


def analyze_sentiment_pool():
    """同步包装函数，用于运行异步主函数"""
    asyncio.run(analyze_sentiment_pool_async())


if __name__ == "__main__":
    analyze_sentiment_pool()

