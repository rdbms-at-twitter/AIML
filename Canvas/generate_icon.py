r"""
画像からアイコンを生成するスクリプト（水彩画風）
使用モデル: Amazon Nova Canvas (amazon.nova-canvas-v1:0)

方式: IMAGE_VARIATION で水彩画風にスタイル変換

使い方:
  python generate_icon.py <入力画像パス> [出力先パス]

例:
  python generate_icon.py photo.png
  python generate_icon.py C:\images\logo.jpg C:\output\icon.png
"""

import sys
import io
import base64
import json
import logging
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "amazon.nova-canvas-v1:0"

# --- アイコン生成の設定 ---
ICON_PROMPT = (
    "beautiful watercolor painting portrait, soft brush strokes, "
    "gentle pastel colors, artistic hand-painted style, "
    "warm and friendly, light watercolor wash background, "
    "delicate and elegant"
)
NEGATIVE_PROMPT = (
    "photo-realistic, 3d render, horror, scary, dark, ugly, "
    "distorted face, extra limbs, blurry, low quality, text"
)
SIMILARITY_STRENGTH = 0.75  # 0.2(自由) ~ 1.0(忠実)
OUTPUT_SIZE = 1024
CFG_SCALE = 8.0
MIN_DIMENSION = 320
MAX_PIXELS = 4_194_304  # Nova Canvas の入力画像最大ピクセル数


def invoke_nova_canvas(body: dict) -> bytes:
    """Nova Canvas を呼び出して生成画像のバイト列を返す"""
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        config=Config(read_timeout=300),
    )

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response["body"].read())

    if response_body.get("error"):
        raise RuntimeError(f"Nova Canvas error: {response_body['error']}")

    return base64.b64decode(response_body["images"][0])


def prepare_image(path: Path) -> str:
    """画像を読み込み → 正方形クロップ → リサイズ → base64"""
    img = Image.open(path)
    logger.info(f"   元サイズ: {img.width}x{img.height}px")

    # RGBに変換
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # 正方形にクロップ
    w, h = img.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    logger.info(f"   クロップ: {img.width}x{img.height}px (正方形)")

    # 最大ピクセル数を超える場合は縮小
    pixels = img.width * img.height
    if pixels > MAX_PIXELS:
        import math
        scale = math.sqrt(MAX_PIXELS / pixels)
        new_size = int(img.width * scale)
        # 16の倍数に切り捨て（Nova Canvas推奨）
        new_size = (new_size // 16) * 16
        img = img.resize((new_size, new_size), Image.LANCZOS)
        logger.info(f"   縮小: {new_size}x{new_size}px (ピクセル上限対応)")

    # 最小サイズ確保
    if img.width < MIN_DIMENSION:
        img = img.resize((MIN_DIMENSION, MIN_DIMENSION), Image.LANCZOS)
        logger.info(f"   リサイズ: {MIN_DIMENSION}x{MIN_DIMENSION}px")

    # base64エンコード
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_icon(image_b64: str) -> bytes:
    """IMAGE_VARIATION で水彩画風アイコン生成"""
    logger.info("アイコン生成中 (水彩画風)...")
    body = {
        "taskType": "IMAGE_VARIATION",
        "imageVariationParams": {
            "text": ICON_PROMPT,
            "negativeText": NEGATIVE_PROMPT,
            "images": [image_b64],
            "similarityStrength": SIMILARITY_STRENGTH,
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": OUTPUT_SIZE,
            "width": OUTPUT_SIZE,
            "cfgScale": CFG_SCALE,
        },
    }
    return invoke_nova_canvas(body)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_icon.py <入力画像パス> [出力先パス]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # 出力パス
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.parent / f"{input_path.stem}_icon.png"

    # 画像読み込み・前処理
    logger.info(f"入力画像: {input_path}")
    image_b64 = prepare_image(input_path)

    # アイコン生成
    icon_bytes = generate_icon(image_b64)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(icon_bytes)

    logger.info(f"✅ アイコン生成完了: {output_path}")
    logger.info(f"   サイズ: {OUTPUT_SIZE}x{OUTPUT_SIZE}px")
    logger.info(f"   スタイル: 水彩画風")
    logger.info(f"   忠実度: {SIMILARITY_STRENGTH}")


if __name__ == "__main__":
    main()

