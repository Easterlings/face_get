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
from rembg_api_group import send_images_to_api
import random
from config import GROUNDING_DINO_CONFIG_PATH,GROUNDING_DINO_CHECKPOINT_PATH,SOURCE_IMAGE_PATH,RESULT_IMAGE_PATH,TRAIN_RESOURCES_PATH
from config import BOX_THRESHOLD,CLASSES,TEXT_THRESHOLD,NMS_THRESHOLD
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                             model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=DEVICE)


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
    TRDirPath = os.path.join(TRAIN_RESOURCES_PATH, imgDir)
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
    cropBox = addN(cropBox,200)
    # cropBox = addN(cropBox,random.randint(0, 4)*100)
    # print("cropBox:",cropBox)
    cropBox = fitin(cropBox,image)
    cropImage = image[int(cropBox[1]):int(cropBox[3]), int(cropBox[0]): int(cropBox[2])]#y1 y2 x1 x2
    resizeImage = cv2.resize(cropImage, (1024, 1024))
    cv2.imwrite(localFilepath, resizeImage)
    # cv2.imwrite(boxFilepath, resizeImage)

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
    return cropBox

def addN(cropBox,n = 100):
    cropBox[0]-=n
    cropBox[1]-=n
    cropBox[2]+=n
    cropBox[3]+=n
    return cropBox

#cropBox需要保持在画框内,暂不考虑方框边长大于画面的情况
def fitin(cropBox,image):
    height, width, _ = image.shape
    if cropBox[0]<0:
        cropBox[2]-=cropBox[0]
        cropBox[0]=0
    if cropBox[1]<0:
        cropBox[3]-=cropBox[1]
        cropBox[1]=0
    if cropBox[2]>width:
        cropBox[0]=cropBox[0]-cropBox[2]+width
        cropBox[2]=width
    if cropBox[3]>height:
        cropBox[1]=cropBox[1]-cropBox[3]+height
        cropBox[3]=height

    return cropBox

for dir in os.listdir(SOURCE_IMAGE_PATH):
    sourceDir = os.path.join(SOURCE_IMAGE_PATH, dir)
    if os.path.isfile(sourceDir):
        continue
    for filename in os.listdir(sourceDir):
        print(filename)
        face_only(sourceDir, dir, filename)

    image_folder = os.path.join(RESULT_IMAGE_PATH, dir)
    output_folder = os.path.join(RESULT_IMAGE_PATH, f"{dir}_nobg")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    send_images_to_api(image_folder, output_folder)