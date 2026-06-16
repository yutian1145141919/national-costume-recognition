"""
网络图片爬虫模块
从公开图片来源搜索并下载民族服饰图片
支持：Bing图片搜索、百度图片搜索
"""
import os
import time
import hashlib
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, urlparse

from tqdm import tqdm

# 浏览器UA，避免被反爬
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# 三类民族的搜索关键词配置
ETHNIC_KEYWORDS = {
    "zang": [
        "藏族服饰", "藏族传统服装", "藏族节日盛装",
        "Tibetan traditional costume", "Tibetan dress",
        "藏族服饰细节", "藏族头饰", "藏族刺绣",
        "藏袍", "藏族银饰", "藏族婚礼服饰"
    ],
    "qiang": [
        "羌族服饰", "羌族传统服装", "羌族刺绣",
        "Qiang ethnic costume", "Qiang traditional dress",
        "羌族头饰", "羌族节日服装", "羌族云云鞋",
        "羌族女性服饰", "羌族男子服饰"
    ],
    "yi": [
        "彝族服饰", "彝族传统服装", "彝族银饰",
        "Yi ethnic costume", "Yi traditional dress",
        "彝族头饰", "彝族擦尔瓦", "彝族百褶裙",
        "彝族火把节服饰", "彝族刺绣服装"
    ]
}

# 下载超时与重试配置
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒


def _download_image(url: str, save_path: str) -> bool:
    """下载单张图片，返回是否成功"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()

            # 检查 Content-Type 是否为图片
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                return False

            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True

        except requests.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return False


def _is_valid_image(path: str, min_size_kb: int = 2) -> bool:
    """检查下载的文件是否为有效图片（大小合理）"""
    try:
        file_size = os.path.getsize(path)
        if file_size < min_size_kb * 1024:
            return False
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def search_bing_images(keyword: str, max_results: int = 50) -> list[str]:
    """
    从 Bing 图片搜索获取图片 URL 列表

    参数:
        keyword: 搜索关键词
        max_results: 最大返回结果数

    返回:
        图片URL列表
    """
    image_urls = []
    encoded_query = quote(keyword)
    # Bing 图片搜索 API（非官方）
    base_url = f"https://www.bing.com/images/search?q={encoded_query}&first="

    count = 0
    for offset in range(0, max_results * 2, 35):  # Bing每页约35张
        if len(image_urls) >= max_results:
            break
        try:
            url = base_url + str(offset)
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()

            # 从页面提取图片URL（murl属性）
            html = resp.text
            # 简易解析：搜索 "murl":"..." 模式
            import re
            matches = re.findall(r'"murl"\s*:\s*"([^"]+)"', html)
            for m in matches:
                if m not in image_urls:
                    image_urls.append(m)
                    if len(image_urls) >= max_results:
                        break
            if not matches:
                break  # 无更多结果
        except requests.RequestException:
            break

    return image_urls


def search_baidu_images(keyword: str, max_results: int = 50) -> list[str]:
    """
    从百度图片搜索获取图片 URL 列表

    参数:
        keyword: 搜索关键词
        max_results: 最大返回结果数

    返回:
        图片URL列表
    """
    image_urls = []
    encoded_query = quote(keyword)
    base_url = "https://image.baidu.com/search/acjson"

    for pn in range(0, max_results, 30):
        if len(image_urls) >= max_results:
            break
        params = {
            "tn": "resultjson_com",
            "word": encoded_query,
            "pn": pn,
            "rn": min(30, max_results - len(image_urls)),
        }
        try:
            resp = requests.get(base_url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("data", []):
                if "thumbURL" in item and item["thumbURL"]:
                    # 优先使用原图链接
                    img_url = item.get("objURL", item.get("thumbURL"))
                    if img_url not in image_urls:
                        image_urls.append(img_url)
                        if len(image_urls) >= max_results:
                            break
        except (requests.RequestException, ValueError):
            break

    return image_urls


def download_images(
    keyword_list: list[str],
    save_dir: str,
    max_per_keyword: int = 20,
    max_workers: int = 6,
    source: str = "bing"
) -> int:
    """
    根据关键词列表下载图片

    参数:
        keyword_list: 搜索关键词列表
        save_dir: 保存目录
        max_per_keyword: 每个关键词最大下载数
        max_workers: 并发下载线程数
        source: 图片来源 "bing" 或 "baidu"

    返回:
        成功下载的图片数量
    """
    os.makedirs(save_dir, exist_ok=True)

    if source == "baidu":
        search_fn = search_baidu_images
    else:
        search_fn = search_bing_images

    # 收集所有待下载 URL
    all_urls = set()
    for kw in keyword_list:
        print(f"  搜索关键词: {kw}")
        try:
            urls = search_fn(kw, max_per_keyword)
            all_urls.update(urls)
        except Exception as e:
            print(f"  搜索失败: {e}")
            continue

    print(f"  共找到 {len(all_urls)} 个图片URL，开始下载...")

    # 并发下载
    success_count = 0
    existing_files = set(os.listdir(save_dir))

    tasks = []
    for url in all_urls:
        # 用URL的哈希作为文件名，避免重复
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            ext = ".jpg"
        fname = f"{url_hash}{ext}"
        if fname in existing_files:
            continue
        fpath = os.path.join(save_dir, fname)
        tasks.append((url, fpath))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_image, url, fpath): (url, fpath) for url, fpath in tasks}
        with tqdm(total=len(tasks), desc="  下载中") as pbar:
            for future in as_completed(futures):
                url, fpath = futures[future]
                try:
                    ok = future.result()
                    if ok and _is_valid_image(fpath):
                        success_count += 1
                    else:
                        # 删除无效文件
                        if os.path.exists(fpath):
                            os.remove(fpath)
                except Exception:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                pbar.update(1)

    print(f"  成功下载 {success_count} 张图片到 {save_dir}")
    return success_count


def run_collection(base_dir: str = "dataset/raw", target_count: int = 200):
    """
    执行完整的数据采集流程

    参数:
        base_dir: 数据集根目录
        target_count: 每类目标图片数量
    """
    base_path = Path(base_dir)

    for ethnicity, keywords in ETHNIC_KEYWORDS.items():
        save_dir = base_path / ethnicity
        current_count = len(list(save_dir.glob("*"))) if save_dir.exists() else 0
        needed = max(0, target_count - current_count)

        print(f"\n{'='*50}")
        print(f"采集 {ethnicity} 服饰图片（已有 {current_count} 张，目标 {target_count} 张）")
        print(f"{'='*50}")

        if needed <= 0:
            print(f"  已满足目标数量，跳过采集")
            continue

        # 每个关键词下载量 = 还需要 / 关键词数 + 一些冗余
        per_kw = max(10, needed // len(keywords) + 5)
        download_images(keywords, str(save_dir), max_per_keyword=per_kw)

        final_count = len(list(save_dir.glob("*")))
        print(f"  最终 {ethnicity} 图片数量: {final_count}")
        time.sleep(1)  # 请求间隔


if __name__ == "__main__":
    print("=" * 50)
    print("民族服饰图片采集工具")
    print("=" * 50)

    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "dataset" / "raw"

    import argparse
    parser = argparse.ArgumentParser(description="民族服饰图片采集")
    parser.add_argument("--target", type=int, default=200, help="每类目标图片数量")
    parser.add_argument("--source", choices=["bing", "baidu"], default="bing", help="图片来源")
    parser.add_argument("--out", type=str, default=str(raw_dir), help="输出目录")
    args = parser.parse_args()

    run_collection(base_dir=args.out, target_count=args.target)
    print("\n采集完成！")
