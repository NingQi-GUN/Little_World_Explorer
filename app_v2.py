from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
from pathlib import Path
import os, re, json, uuid, requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
model = YOLO("yolov8n.pt")

KID_WHITELIST = {
    'bird': '小鸟', 'cat': '小猫', 'dog': '小狗', 'horse': '马', 'sheep': '绵羊',
    'cow': '奶牛', 'elephant': '大象', 'bear': '熊', 'zebra': '斑马', 'giraffe': '长颈鹿',
    'person': '人物', 'bicycle': '自行车', 'car': '汽车', 'motorcycle': '摩托车',
    'airplane': '飞机', 'bus': '公交车', 'train': '火车', 'truck': '卡车', 'boat': '船',
    'traffic light': '红绿灯', 'fire hydrant': '消防栓', 'stop sign': '停车标志',
    'parking meter': '停车计时器', 'bench': '长椅', 'backpack': '背包', 'umbrella': '雨伞',
    'handbag': '手提包', 'tie': '领带', 'suitcase': '行李箱', 'frisbee': '飞盘',
    'skis': '滑雪板', 'snowboard': '单板滑雪板', 'sports ball': '运动球', 'kite': '风筝',
    'baseball bat': '棒球棒', 'baseball glove': '棒球手套', 'skateboard': '滑板',
    'surfboard': '冲浪板', 'tennis racket': '网球拍', 'bottle': '瓶子', 'wine glass': '酒杯',
    'cup': '杯子', 'fork': '叉子', 'knife': '刀', 'spoon': '勺子', 'bowl': '碗',
    'banana': '香蕉', 'apple': '苹果', 'sandwich': '三明治', 'orange': '橙子',
    'broccoli': '西兰花', 'carrot': '胡萝卜', 'hot dog': '热狗', 'pizza': '披萨',
    'donut': '甜甜圈', 'cake': '蛋糕', 'chair': '椅子', 'couch': '沙发',
    'potted plant': '盆栽', 'bed': '床', 'dining table': '餐桌', 'toilet': '马桶',
    'tv': '电视', 'laptop': '笔记本电脑', 'mouse': '鼠标', 'remote': '遥控器',
    'keyboard': '键盘', 'cell phone': '手机', 'microwave': '微波炉', 'oven': '烤箱',
    'toaster': '烤面包机', 'sink': '水槽', 'refrigerator': '冰箱', 'book': '书本',
    'clock': '时钟', 'vase': '花瓶', 'scissors': '剪刀', 'teddy bear': '泰迪熊',
    'hair drier': '吹风机', 'toothbrush': '牙刷'
}

ALL_CN_LABELS = sorted(set(KID_WHITELIST.values()))


def get_glm_story(label_cn, category_cn=None):
    fallback = {
        "title": "找到了" + label_cn + "！",
        "paragraphs": ["你找到了" + label_cn + "，真棒！"],
        "funFacts": []
    }
    api_key = os.getenv("GLM_API_KEY")
    if not api_key or api_key == "先空着":
        fallback["paragraphs"] = ["你找到了" + label_cn + "，真棒！（提示：请在 .env 填入 GLM_API_KEY）"]
        return fallback

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    prompt = (
        "你是温柔亲切的少儿自然科普向导。今天小朋友的寻宝任务是【" + label_cn + "】"
        "（属于" + (category_cn or "物品") + "类）。请对4-8岁小朋友说一段话："
        "先用惊喜语气恭喜找到它，再讲故事般介绍它的样子和有趣知识，最后问一个互动小问题。\n"
        "严格只输出JSON（不要markdown、不要任何解释文字），格式如下：\n"
        '{"title":"简短标题不超过12字","paragraphs":["第一段不超过60字","第二段不超过60字"],'
        '"funFacts":["趣味小知识1","趣味小知识2"]}\n'
        "所有字段必填，总字数不超过150字。"
    )
    payload = {
        "model": "glm-4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        story = json.loads(content)
        if not isinstance(story, dict):
            raise ValueError("glm returned non-dict")
        story.setdefault("title", "找到了" + label_cn + "！")
        story.setdefault("paragraphs", [])
        story.setdefault("funFacts", [])
        if not isinstance(story["paragraphs"], list) or not story["paragraphs"]:
            story["paragraphs"] = ["你找到了" + label_cn + "。"]
        if not isinstance(story["funFacts"], list):
            story["funFacts"] = []
        return story
    except Exception as e:
        app.logger.warning("GLM 失败或解析失败: " + str(e))
        fallback["paragraphs"] = ["这是" + label_cn + "。（AI 解说暂时不可用，请检查 API Key 或网络）"]
        return fallback


@app.route("/api/recognize", methods=["POST"])
@app.route("/predict", methods=["POST"])
def recognize():
    target_name = (request.form.get("targetName") or "").strip()
    target_id = (request.form.get("targetId") or "").strip()
    category_name = (request.form.get("categoryName") or "").strip()

    file = None
    for k in ("image", "file"):
        if k in request.files:
            file = request.files[k]
            break
    if not file or not file.filename:
        return jsonify({
            "success": False, "matched": False,
            "error": "没有收到图片字段", "message": "没有收到图片字段"
        }), 400

    save_path = UPLOAD_DIR / (uuid.uuid4().hex + ".jpg")
    try:
        file.save(str(save_path))

        try:
            results = model(str(save_path), conf=0.35)
        except Exception as e:
            app.logger.error("YOLO 推理失败: " + str(e))
            return jsonify({
                "success": False, "matched": False,
                "error": "图像识别服务异常", "message": "图像识别服务异常"
            }), 500

        objects = []
        labels_in_order = []
        for box in results[0].boxes:
            raw = model.names[int(box.cls)]
            if raw in KID_WHITELIST:
                cn = KID_WHITELIST[raw]
                labels_in_order.append(cn)
                objects.append({"label": cn, "confidence": round(float(box.conf), 3)})

        unique_objects = []
        seen_labels = set()
        for obj in objects:
            if obj["label"] not in seen_labels:
                unique_objects.append(obj)
                seen_labels.add(obj["label"])

        matched = bool(target_name) and (target_name in seen_labels)
        if matched:
            detected_label = target_name
            confidence = next(
                (o["confidence"] for o in unique_objects if o["label"] == target_name),
                0.0
            )
        else:
            if unique_objects:
                detected_label = unique_objects[0]["label"]
                confidence = unique_objects[0]["confidence"]
            else:
                detected_label = "神秘的东西"
                confidence = 0.0

        if matched:
            story = get_glm_story(target_name, category_name)
            comment = story["title"] + "。" + "".join(story["paragraphs"])
        else:
            story = None
            if target_name:
                comment = "你看见了" + detected_label + "，不是这次要找的" + target_name + "哦，再试一次吧！"
            else:
                comment = "你看见了" + detected_label + "。"

        return jsonify({
            "success": True,
            "objects": unique_objects,
            "comment": comment,
            "matched": matched,
            "detectedLabel": detected_label,
            "confidence": round(confidence, 3),
            "story": story,
            "narrationUrl": None
        })

    finally:
        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/api/health", methods=["GET"])
def health():
    key = os.getenv("GLM_API_KEY")
    return jsonify({
        "ok": True,
        "model": "yolov8n.pt",
        "glm_ready": bool(key) and key != "先空着",
        "targets_count": len(ALL_CN_LABELS)
    })


@app.route("/api/targets", methods=["GET"])
def list_targets():
    return jsonify({"targets": ALL_CN_LABELS})

@app.route("/")
def index():
    return send_file("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)