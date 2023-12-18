import os
abs_env_path = os.path.abspath('.env')
# 加载环境变量
if os.path.exists(abs_env_path):
    with open(abs_env_path, 'r') as env_file:
        for line in env_file:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value
else:
    raise Exception(f".env file not exists")

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.environ.get('GROUNDING_DINO_CONFIG_PATH')
GROUNDING_DINO_CHECKPOINT_PATH = os.environ.get('GROUNDING_DINO_CHECKPOINT_PATH')

SOURCE_IMAGE_PATH = os.environ.get('SOURCE_IMAGE_PATH')
RESULT_IMAGE_PATH = os.environ.get('RESULT_IMAGE_PATH')
TRAIN_RESOURCES_PATH = os.environ.get('TRAIN_RESOURCES_PATH')

BOX_THRESHOLD = 0.20
CLASSES = ["head"]
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.3