import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 替换为您的实际文件路径
# file_path = '/inspire/hdd/project/video-generation/chenxie-25019/hyr/F5-TTS/ckpts/DTM_F5TTS_Base_vocos_pinyin_WenetSpeech4TTS_Premium/samples/update_40000_gen.wav'

file_path = '/inspire/hdd/project/video-generation/chenxie-25019/hyr/F5-TTS/tests/infer_cli_basic.wav'

try:
    # 1. 加载音频文件
    # sr=None表示使用文件的原始采样率
    y, sr = librosa.load(file_path, sr=None) 

    # 2. 计算梅尔频谱图 (默认n_fft=2048, hop_length=512)
    # n_mels通常取80或128
    D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=32)

    # 3. 转换为分贝 (dB)
    D_db = librosa.power_to_db(D, ref=np.max)

    # 4. 绘制图形
    plt.figure(figsize=(10, 4))
    # specshow函数用于显示频谱图，y_axis='mel'表示纵轴是梅尔刻度
    librosa.display.specshow(D_db, 
                             sr=sr, 
                             x_axis='time', 
                             y_axis='mel', 
                             cmap='magma') # 可以选择不同的颜色映射，如'viridis', 'magma', 'jet'等
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig('2.png')

except Exception as e:
    print(f"处理文件时发生错误: {e}")