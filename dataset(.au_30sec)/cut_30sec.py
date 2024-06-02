from pydub import AudioSegment
import shutil
import os

def split_audio_files(input_folder, output_folder):
    # 檢查輸出資料夾是否存在，若存在則先刪除
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # 創建新的輸出資料夾
    os.makedirs(output_folder)

    # 獲取輸入資料夾中的所有.au檔案
    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.au')]

    count = 0

    # 遍歷每個音樂檔案並分割
    for audio_file in audio_files:
        input_file = os.path.join(input_folder, audio_file)
        audio = AudioSegment.from_file(input_file)

        # 計算要分割的段數
        split_length_ms = 30 * 1000  # 30 秒轉換成毫秒
        num_segments = len(audio) // split_length_ms

        # 分割音樂檔案
        for i in range(num_segments):
            start_time = i * split_length_ms
            end_time = (i + 1) * split_length_ms
            segment = audio[start_time:end_time]
            output_file = os.path.join(output_folder, f"blues.{str(count).zfill(6)}.au")
            segment.export(output_file, format="au")
            count += 1

    print("分割完成。")

# 取得當前檔案的目錄路徑
current_directory = os.path.dirname(os.path.abspath(__file__))

# 設定輸入資料夾及輸出資料夾的相對路徑
input_folder = os.path.join(current_directory, "input_folder")
output_folder = os.path.join(current_directory, "thirty")

# 執行分割音樂檔案
split_audio_files(input_folder, output_folder)
