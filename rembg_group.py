import os
from glob import glob
from PIL import Image
from rembg import remove

def rem_bg(image_folder, output_folder):
    # 获取图片文件列表
    image_files = glob(os.path.join(image_folder, '*.jpg'))  # 可以根据实际情况修改文件扩展名

    # 遍历图片文件并发送POST请求
    for image_file in image_files:
        try:
            no_bg_image = remove(Image.open(image_file)).convert('RGB')
                # 构建保存图片的路径
            output_path = os.path.join(output_folder, os.path.basename(image_file))
            no_bg_image.save(output_path)
            print(f"Image {image_file} sent and saved successfully to {output_path}.")

        except Exception as e:
            print(f"Error processing image {image_file}: {str(e)}")

