import os
import torch
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm
import sys

# 将根目录添加到Python路径，以便导入utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从utils中导入必要的模块和函数
from utils.aes_utils import Selector as AestheticsScorer
from utils.hps_utils import Selector as HpsScorer
from utils.imagereward_utils import Selector as ImageRewardScorer
from utils.pickscore_utils import PickScoreScorer

def get_prompt_from_filename(filename):
    """从文件名中提取提示词 (移除批次信息和文件后缀)"""
    parts = os.path.splitext(filename)[0].split('_Batch_')
    return parts[0]

def main():
    """
    主函数，用于加载模型、处理图片并保存评分结果。
    """
    # --- 配置区域 ---
    TEST_DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_dataset'))
    RESULTS_CSV_PATH = os.path.join(os.path.dirname(__file__), 'results.csv')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"使用设备: {DEVICE}")
    print(f"测试数据集路径: {TEST_DATASET_PATH}")
    print(f"结果将保存至: {RESULTS_CSV_PATH}")

    # --- 1. 加载所有评分模型 ---
    print("\n正在加载评分模型...")
    try:
        # 美学评分模型
        aes_model = AestheticsScorer(device=DEVICE)
        # HPS 模型
        hps_model = HpsScorer(device=DEVICE)
        # ImageReward 模型
        imagereward_model = ImageRewardScorer(device=DEVICE)
        # PickScore 模型
        pickscore_model = PickScoreScorer(device=DEVICE)
        print("所有模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保已根据项目README的要求下载并放置了所有必要的模型文件。")
        return

    # --- 2. 查找所有待处理的图片 ---
    image_paths = []
    for root, _, files in os.walk(TEST_DATASET_PATH):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"在 {TEST_DATASET_PATH} 中没有找到任何图片文件。请检查路径。")
        return
        
    print(f"\n找到 {len(image_paths)} 张图片。开始评分...")

    # --- 3. 遍历图片并评分 ---
    results = []
    # 使用tqdm创建进度条
    for image_path in tqdm(image_paths, desc="正在评分"):
        try:
            # 提取提示词
            prompt = get_prompt_from_filename(os.path.basename(image_path))
            
            # 打开图片
            image = Image.open(image_path).convert("RGB")

            # 计算所有分数
            aes_score_list = aes_model.score(image, prompt)
            aes_score = aes_score_list[0] if isinstance(aes_score_list, list) else aes_score_list

            # 为HPS模型传递路径，为其他模型传递Image对象
            hps_score_list = hps_model.score(image_path, prompt)
            hps_score = hps_score_list[0] if isinstance(hps_score_list, list) else hps_score_list

            imagereward_score_list = imagereward_model.score(image, prompt)
            imagereward_score = imagereward_score_list[0] if isinstance(imagereward_score_list, list) else imagereward_score_list
            
            pickscore_score_list = pickscore_model.score(image, prompt)
            pickscore_score = pickscore_score_list[0] if isinstance(pickscore_score_list, list) else pickscore_score_list

            results.append({
                "prompt": prompt,
                "image_path": os.path.relpath(image_path, os.path.dirname(RESULTS_CSV_PATH)),
                "aes_score": aes_score,
                "hps_score": hps_score,
                "imagereward_score": imagereward_score,
                "pickscore_score": pickscore_score
            })
        except Exception as e:
            print(f"\n处理图片 {image_path} 时出错: {e}")
            continue # 如果一张图片出错，记录错误并继续处理下一张

    # --- 4. 保存结果到CSV ---
    if not results:
        print("没有成功处理任何图片，不生成CSV文件。")
        return

    df = pd.DataFrame(results)
    # 根据提示词排序，方便查看
    df.sort_values(by="prompt", inplace=True)
    df.to_csv(RESULTS_CSV_PATH, index=False)

    print(f"\n评分完成！结果已保存到 {RESULTS_CSV_PATH}")
    print("你可以使用Excel或Pandas查看和分析结果。")

if __name__ == "__main__":
    main()
