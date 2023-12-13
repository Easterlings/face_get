#基于GroundingDINO从图片中批量提取面部，用于lora训练
#结果保存两份，一份到lora训练文件夹，一份到本项目文件夹
#需要运行于显存大于4GB的机器上
#使用时只需将装有图片的文件夹放在imgs/img目录下，然后运行copy_face.py即可
#结果可以在imgs/faces/内看到
import os
import cv2
import torch
import torchvision

from local_groundingdino.util.inference import Model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "/data/web/saiwei-wardrobe-image-process//models/grounding-dino/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/data/web/saiwei-wardrobe-image-process/models/grounding-dino/groundingdino_swint_ogc.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                             model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=DEVICE)

SOURCE_IMAGE_PATH = "./imgs/img"
BOX_THRESHOLD = 0.20
CLASSES = ["head"]
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.3


def indexOfMaxConfidence(confidences):
    maxC = -1
    maxI = -1
    i = 0
    for c in confidences:
        if c > 0.1 and c > maxC:
            maxC = c
            maxI = i
        i += 1
    return maxI


def face_only(sourceDir, imgDir, imgFile):
    DirPath = os.path.join("./imgs/faces", imgDir)
    TRDirPath = os.path.join(f"/data/lkw/train_resources", imgDir)
    image = cv2.imread(os.path.join(sourceDir, imgFile))
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
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
        return

    if not os.path.exists(TRDirPath):
        os.makedirs(TRDirPath)
    if not os.path.exists(DirPath):
        os.makedirs(DirPath)
    localFilepath = os.path.join(DirPath, imgFile)
    boxFilepath = os.path.join(TRDirPath, imgFile)

    cropBox = detections.xyxy[index]

    cropBox = square(cropBox)#调整为方形

    cropImage = image[int(cropBox[1]):int(cropBox[3]), int(cropBox[0]): int(cropBox[2])]#y1 y2 x1 x2
    resizeImage = cv2.resize(cropImage, (1024, 1024))
    cv2.imwrite(localFilepath, resizeImage)
    cv2.imwrite(boxFilepath, resizeImage)

def square(cropBox):
    xlength= int(cropBox[2])-int(cropBox[0])
    ylength= int(cropBox[3])-int(cropBox[1])
    if(ylength>xlength):
        n = (ylength-xlength)/2
        cropBox[2]+=n
        cropBox[0]-=n
    elif(ylength<xlength):
        n = (xlength-ylength)/2
        cropBox[3]+=n
        cropBox[1]-=n
    cropBox = [0 if num < 0 else num for num in cropBox]
    return cropBox


for dir in os.listdir(SOURCE_IMAGE_PATH):
    sourceDir = os.path.join(SOURCE_IMAGE_PATH, dir)
    if os.path.isfile(sourceDir):
        continue
    for filename in os.listdir(sourceDir):
        print(filename)
        face_only(sourceDir, dir, filename)
