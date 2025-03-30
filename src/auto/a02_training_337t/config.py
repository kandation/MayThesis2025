from pathlib import Path

# ค่าคงที่และการตั้งค่าต่างๆ
class Config:
    INPUT_DIR = Path.cwd() / "cleanned_datasets"  # ไดเรกทอรีไฟล์ CSV
    OUTPUT_DIR = Path.cwd() / "outputs"          # ไดเรกทอรีผลลัพธ์
    SEED_VALUE = 42                              # ค่า seed
    BATCH_SIZE = 1024                            # ขนาด batch
    EPOCHS = 50                                  # จำนวน epochs
    SPLITS = 5                                   # จำนวน splits
    SEQ_LENGTH = 24                              # ความยาว sequence
    PRED_LENGTH = 24                             # ความยาวการพยากรณ์
    FONT_NAME = "TH SarabunPSK"                  # ชื่อฟอนต์

    # การตั้งค่าไฟล์ชื่อภาษาอังกฤษ
    FILE_NAME_MAPPING = {
        "(37t)สถานีอุตุนิยมวิทยาลำปาง": "Lampang_Meteorological_Station",
        "(37t)สถานีตรวจวัดคุณภาพอากาศลำปาง": "Lampang_Air_Quality_Station",
        "(38t)สถานีตรวจวัดคุณภาพอากาศเชียงใหม่": "Chiang_Mai_Air_Quality_Station",
        "(39t)สถานีตรวจวัดคุณภาพอากาศเชียงราย": "Chiang_Rai_Air_Quality_Station",
        "(40t)สถานีตรวจวัดคุณภาพอากาศแม่ฮ่องสอน": "Mae_Hong_Son_Air_Quality_Station",
        "(37t)ศาลหลักเมือง(ปิดสถานี)": "San_Lak_Mueang_Closed_Station",
    }