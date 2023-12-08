import copy
import os
import struct

import cv2
import numpy as np
import supervision as sv
import pandas as pd
import torch
import torchvision
from PIL.Image import Image
import sqlite3

from local_groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "E:\\weights\\groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "E:\\weights\\sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                             model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=DEVICE)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Predict classes and hyper-param for GroundingDINO
# SOURCE_IMAGE_PATH = "./assets/my21.jpg"
SOURCE_IMAGE_PATH = "F:\图像处理\人物训练用图\正常hanna\原图"
# CLASSES = ["The running dog"]
BOX_THRESHOLD = 0.50
CLASSES = ["Face"]
TEXT_THRESHOLD = 0.50
NMS_THRESHOLD = 0.8


def indexOfMaxConfidence(confidences):
    maxC = -1
    maxI = -1
    i = 0
    for c in confidences:
        if c > 0.5 and c > maxC:
            maxC = c
            maxI = i
        i += 1
    return maxI


def face_only(sourceDir, imgDir, imgFile):
    # data1 = isExist(conn, imgDir, imgFile)
    # if data1.size > 0:
    #     return

    filename = imgFile[0:imgFile.rfind('.')]
    dirPath = os.path.join("faces/", imgDir)
    boxDirPath = os.path.join("faces_box/", imgDir)
    # if not os.path.exists(dirPath):
    #     os.makedirs(dirPath)
    #     os.makedirs(boxDirPath)
    image = cv2.imread(os.path.join(sourceDir, imgFile))
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
    ).numpy().tolist()
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    print(f"After NMS: {len(detections.xyxy)} boxes")
    index = indexOfMaxConfidence(detections.confidence)
    if index < 0:
        insert(conn, (imgDir, imgFile, 0, "False"))
        return
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    segment_image = copy.copy(image)
    masked_image = mask_image(segment_image, detections.mask[index])

    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
        os.makedirs(boxDirPath)
    filepath = os.path.join(dirPath, filename + '.png')
    boxFilepath = os.path.join(boxDirPath, imgFile)
    save_masked_image(masked_image, filepath)
    cropBox = detections.xyxy[index]
    print(cropBox)
    cropImage = image[int(cropBox[1]):int(cropBox[3]), int(cropBox[0]): int(cropBox[2])]
    cv2.imwrite(boxFilepath, cropImage)
    insert(conn, (imgDir, imgFile, detections.confidence[index], "True"))


def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


def apply_mask(image, mask, alpha_channel=True):  # 应用并且响应mask
    if alpha_channel:
        alpha = np.zeros_like(image[..., 0])  # 制作掩体
        alpha[mask == 1] = 255  # 兴趣地方标记为1，且为白色
        image = cv2.merge((image[..., 0], image[..., 1], image[..., 2], alpha))  # 融合图像
    else:
        image = np.where(mask[..., None] == 1, image, 0)
    return image


def mask_image(image, mask, crop_mode_=True):  # 保存掩盖部分的图像（感兴趣的图像）
    if crop_mode_:
        y, x = np.where(mask)
        y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        masked_image = apply_mask(cropped_image, cropped_mask)
    else:
        masked_image = apply_mask(image, mask)

    return masked_image


def save_masked_image(image, filepath):
    if image.shape[-1] == 4:
        cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(filepath, image)
    print(f"Saved as {filepath}")


# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# 打开或者创建数据库
def openDB(dbFile):
    if os.path.isfile(dbFile):
        conn = sqlite3.connect(dbFile)
    else:
        conn = sqlite3.connect(dbFile)
        cursor = conn.cursor()
        cursor.execute('''
			CREATE TABLE IMAGES
			(ID INTEGER PRIMARY KEY AUTOINCREMENT,
			GOODSID TEXT NOT NULL,
			NAME TEXT NOT NULL,
			SCORE FLOAT,
			EXPORT TEXT);
			''')
        cursor.close()
        conn.commit()
    return conn


# 插入数据库
def insert(conn, parameters):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO IMAGES (GOODSID,NAME,SCORE,EXPORT) VALUES (?,?,?,?)", parameters)
    cursor.close()
    conn.commit()


# 关闭数据库
def closeDB(conn):
    conn.close()


# 导出数据库保存到excel
def export(conn, resultFile):
    # print(struct.unpack('f',nani[3]))
    data = pd.read_sql_query("SELECT * FROM IMAGES WHERE SCORE > 0", conn)
    data.columns = ['ID', '商品id', '名字', '评分', '是否抠图']
    data['评分'] = data['评分'].apply(lambda x: struct.unpack('f', x)[0])
    data.to_excel(resultFile)


# 导出数据库保存到excel
def isExist(conn, imgDir, imgFile):
    data = pd.read_sql_query("SELECT GOODSID,NAME FROM IMAGES WHERE GOODSID='{}' AND NAME='{}'".format(imgDir, imgFile),
                             conn)
    return data


# dbName = "images.db"
# conn = openDB(dbName)
# export(conn, "images.xlsx")
# closeDB(conn)

for dir in os.listdir(SOURCE_IMAGE_PATH):
    sourceDir = os.path.join(SOURCE_IMAGE_PATH, dir)
    if os.path.isfile(sourceDir):
        continue
    for filename in os.listdir(sourceDir):
        face_only(sourceDir, dir, filename)
# export(conn, "images.xlsx")
# closeDB(conn)
