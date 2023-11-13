import os
import gdown


if __name__ == '__main__':
    url = 'https://drive.google.com/uc?id=1212m_9I84b8xqJeVgndcaCI4nWoYoLX_'
    output_dir = 'saved/models/final'
    os.makedirs(output_dir, exist_ok=True)
    output = 'saved/models/final/weights.pth'
    gdown.download(url, output)
