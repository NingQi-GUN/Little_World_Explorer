🐾 小小探索家 · 星瞳AI

传智杯 AI Web 赛道项目：基于 YOLOv8 与智谱 GLM 的儿童自然科普互动 Web 应用。

✨ 核心功能
📸 拍照识物：调用浏览器摄像头，基于 YOLOv8n 模型进行本地目标检测。
📖 童话生成：接入智谱 GLM 大模型，将识别结果转化为儿童友好的科普故事。
🔊 语音播报：前端集成语音合成，让孩子“听”见探索结果。
🏆 图鉴收集：点亮机制与徽章系统，激发探索欲。
🛡️ 诚实判错：识别置信度低时明确提示，绝不 AI 幻觉糊弄孩子。

🛠️ 技术栈
前端：单文件 HTML (React UMD) + 原生 JavaScript + localStorage。
后端：Python Flask + Flask-CORS。
AI 模型：Ultralytics YOLOv8n + 智谱 GLM-4 API。

🚀 快速开始 (本地运行)
1. 克隆仓库并进入目录：
   git clone https://github.com/NingQi-GUN/Little_World_Explorer.git
   cd Little_World_Explorer
2. 安装 Python 依赖：
   pip install -r requirements.txt
3. 配置环境变量：
   复制 `.env.example` 并重命名为 `.env`，填入你的智谱 API Key：
   GLM_API_KEY=你的智谱API密钥
4. 启动后端服务：
   python app_v2.py
5. 打开浏览器访问：http://127.0.0.1:5000/ （请勿直接双击打开 HTML，必须通过 Flask 服务访问）。

⚠️ 项目边界与声明
模型限制：YOLOv8n 仅支持 COCO 80 类，真实拍摄效果最佳。
设备兼容：受浏览器安全策略限制，手机非 HTTPS 不支持实时预览。
隐私保护：数据仅保存在 localStorage，不上传服务器，保障隐私。
