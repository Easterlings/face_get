import os
import requests
from glob import glob
API_URL = "http://192.168.200.143:5000/api/remove"

def send_images_to_api(image_folder, output_folder):
    # 获取图片文件列表
    image_files = glob(os.path.join(image_folder, '*.jpg'))  # 可以根据实际情况修改文件扩展名

    # 遍历图片文件并发送POST请求
    for image_file in image_files:
        try:
            with open(image_file, 'rb') as file:
                files = {'file': (os.path.basename(image_file), file, 'image/jpeg')}
                response = requests.post(API_URL, files=files)

            # 处理API响应
            if response.status_code == 200:
                # 从API响应中获取图片数据
                image_data = response.content
                
                # 构建保存图片的路径
                output_path = os.path.join(output_folder, os.path.basename(image_file))
                
                # 保存图片到指定路径
                with open(output_path, 'wb') as output_file:
                    output_file.write(image_data)

                print(f"Image {image_file} sent and saved successfully to {output_path}.")
            else:
                print(f"Failed to send image {image_file}. API response: {response.text}")

        except Exception as e:
            print(f"Error processing image {image_file}: {str(e)}")

if __name__ == "__main__":
    # 替换为你的API地址、图片文件夹路径和保存图片的文件夹路径

    image_folder = "E:/sw5493/Documents/0/18.赛维衣橱/face_get/imgs/faces/marco"
    output_folder = "E:/sw5493/Documents/0/18.赛维衣橱/face_get/imgs/faces/nobg_marco2"

    send_images_to_api(image_folder, output_folder)
