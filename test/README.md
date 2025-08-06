# 图像评分测试框架

本文件夹包含用于对 `test_dataset` 中的图像进行评分的工具。

## 使用方法

1.  **安装依赖**:
    确保你已经安装了项目所需的所有依赖。如果没有，请返回项目根目录并运行：
    ```bash
    pip install -r requirements.txt
    ```

2.  **运行评分脚本**:
    在 `test` 文件夹内，运行以下命令来启动评分过程：
    ```bash
    python scoring_script.py
    ```
    脚本会自动处理位于 `../test_dataset` 目录下的所有图片。

3.  **查看结果**:
    脚本执行完毕后，所有的评分结果将被保存在 `results.csv` 文件中。

    该文件将包含以下列：
    - `prompt`: 从文件名中提取的提示词。
    - `image_path`: 图片的相对路径。
    - `aes_score`: 美学评分。
    - `hps_score`: 人类偏好评分 (HPS)。
    - `imagereward_score`: ImageReward 评分。
    - `pickscore_score`: PickScore 评分。
