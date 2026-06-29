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

# 情绪分析文本池保存路径
SENTIMENT_POOL_DIR = "/data/depression_tts/synthese_data/sentiment_pool"
SENTIMENT_POOL_HEALTHY_FILE = os.path.join(SENTIMENT_POOL_DIR, "healthy_text_pool.json")
SENTIMENT_POOL_DEPRESSED_FILE = os.path.join(SENTIMENT_POOL_DIR, "depressed_text_pool.json")
SENTIMENT_POOL_METADATA_FILE = os.path.join(SENTIMENT_POOL_DIR, "metadata.json")


# ───────────────────────────  辅助函数  ───────────────────────────────────

def clean_text(text: str) -> str:
    """清理文本"""
    text = re.sub(r'\s+', ' ', str(text)).strip()
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\'\"]', '', text)
    return text


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
        text_info: (text, line_idx) 元组
        client: AsyncOpenAI 客户端
        semaphore: 信号量，用于控制并发数
    
    Returns:
        dict: {"text": text, "sentiment": sentiment, "category": category} 或 None
    """
    text, line_idx = text_info
    
    # 验证文本有效性
    if not is_text_valid(text):
        return None
    
    # 清理文本
    cleaned_text = clean_text(text)
    
    # 使用 deepseek 进行情绪分析（使用信号量控制并发）
    try:
        async with semaphore:  # 使用信号量控制并发数
            sentiment_result = await request_sentiment_analysis_async(cleaned_text, client)
        
        # 判断情绪类别
        is_positive_or_neutral = ("积极" in sentiment_result or "中性" in sentiment_result or 
                                  "positive" in sentiment_result.lower() or "neutral" in sentiment_result.lower())
        is_negative = ("消极" in sentiment_result or "negative" in sentiment_result.lower())
        
        category = None
        if is_positive_or_neutral:
            category = "healthy"  # 积极/中性文本 -> Healthy_Text_Pool
        elif is_negative:
            category = "depressed"  # 消极文本 -> Depressed_Text_Pool
        
        if category:
            return {
                "text": cleaned_text,
                "sentiment": sentiment_result,
                "category": category
            }
        
        return None
        
    except Exception as e:
        if line_idx < 10:  # 只打印前10个错误
            print(f"  Warning: Error analyzing sentiment for line {line_idx}: {e}")
        return None


# ───────────────────────────  主函数  ───────────────────────────────────

async def analyze_sentiment_pool_async():
    """从训练集 filelist 加载文本，使用情绪分析（deepseek）分离为 Healthy_Text_Pool 和 Depressed_Text_Pool（异步版本）"""
    print("=== 情绪分析脚本（异步优化版）：从训练集 filelist 分析文本情绪 ===")
    
    # 检查是否存在已保存的文本池
    if os.path.exists(SENTIMENT_POOL_HEALTHY_FILE) and os.path.exists(SENTIMENT_POOL_DEPRESSED_FILE):
        print(f"⚠️  发现已存在的文本池文件:")
        print(f"  Healthy: {SENTIMENT_POOL_HEALTHY_FILE}")
        print(f"  Depressed: {SENTIMENT_POOL_DEPRESSED_FILE}")
        response = input("是否重新分析？(y/n): ").strip().lower()
        if response != 'y':
            print("已取消，退出。")
            return
    
    healthy_texts = []
    depressed_texts = []
    
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
            
            text = parts[1].strip()
            text_list.append((text, line_idx))
            
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
                continue  # 跳过异常结果
            
            if result:
                if result["category"] == "healthy":
                    healthy_texts.append(result["text"])
                elif result["category"] == "depressed":
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
    
    # 保存文本池
    print(f"\n💾 Saving sentiment pools to {SENTIMENT_POOL_DIR}...")
    try:
        os.makedirs(SENTIMENT_POOL_DIR, exist_ok=True)
        
        # 保存文本池
        with open(SENTIMENT_POOL_HEALTHY_FILE, 'w', encoding='utf-8') as f:
            json.dump(healthy_texts, f, indent=2, ensure_ascii=False)
        with open(SENTIMENT_POOL_DEPRESSED_FILE, 'w', encoding='utf-8') as f:
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
        with open(SENTIMENT_POOL_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Sentiment pools saved successfully")
        print(f"  Healthy texts: {SENTIMENT_POOL_HEALTHY_FILE}")
        print(f"  Depressed texts: {SENTIMENT_POOL_DEPRESSED_FILE}")
        print(f"  Metadata: {SENTIMENT_POOL_METADATA_FILE}")
    except Exception as e:
        print(f"  ✗ Failed to save sentiment pools: {e}")
        raise
    
    print(f"\n{'='*60}")
    print(f"=== 情绪分析完成 ===")
    print(f"Healthy_Text_Pool: {len(healthy_texts)} texts")
    print(f"Depressed_Text_Pool: {len(depressed_texts)} texts")
    print(f"{'='*60}")


def analyze_sentiment_pool():
    """同步包装函数，用于运行异步主函数"""
    asyncio.run(analyze_sentiment_pool_async())


if __name__ == "__main__":
    analyze_sentiment_pool()

