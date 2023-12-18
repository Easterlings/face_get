## face get

基于GroundingDINO从图片中批量提取面部，用于lora训练。结果保存两份，一份到lora训练文件夹，一份到本项目文件夹

### 使用方法
首先下载GroundingDINO及所需模型文件，详细参考[GroundingDINO项目介绍](https://github.com/IDEA-Research/GroundingDINO)
修改example.env的名称为.env，将其中的各项参数配置为实际路径
使用时只需将装有图片的文件夹放在SOURCE_IMAGE_PATH所指向的目录下，然后运行copy_face.py即可
结果可以在RESULT_IMAGE_PATH或TRAIN_RESOURCES_PATH内看到

需要运行于显存大于4GB的机器上
