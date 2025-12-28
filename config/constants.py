# Constantes do projeto
FOLDER_PATH = "FORTH_TRACE_DATASET/part"
DEVICES = [1, 2, 3, 4, 5]
DEVICE_LABELS = ["Pulso esquerdo", "Pulso direito", "Peito", "Perna superior direita", "Perna inferior esquerda"]

ACTIVITY_IDS = range(1, 17, 1)
ACTIVITY_LABELS = [
    "Stand", "Sit", "Sit and Talk", "Walk", "Walk and Talk", 
    "Climb Stair (up/down)", "Climb Stair (up/down) and talk", "Stand-> Sit",
    "Sit-> Stand", "Stand-> Sit and talk", "Sit->Stand and talk", "Stand-> walk",
    "Walk-> stand", "Stand -> climb stairs (up/down), stand -> climb stairs (up/down) and talk",
    "Climb stairs (up/down) -> walk", "Climb stairs (up/down) and talk -> walk and talk"
]


ACRONYM_LABELS = [
    "STD",      # Stand
    "SIT",      # Sit
    "SIT-T",    # Sit and Talk
    "WLK",      # Walk
    "WLK-T",    # Walk and Talk
    "STR",      # Stairs
    "STR-T",    # Stairs and Talk
    "STD->SIT",
    "SIT->STD",
    "STD->SIT-T",
    "SIT->STD-T",
    "STD->WLK",
    "WLK->STD",
    "STD->STR",
    "STR->WLK",
    "STR-T->WLK-T"
]

COLUMNS_ACCELEROMETER = ["accelerometer_x", "accelerometer_y", "accelerometer_z"]
COLUMNS_GYROSCOPE = ["gyroscope_x", "gyroscope_y", "gyroscope_z"]
COLUMNS_MAGNETOMETER = ["magnetometer_x", "magnetometer_y", "magnetometer_z"]



COLUMNS_MODULE = ["magnetometer_module", "gyroscope_module", "accelerometer_module"]
ACC_GYRO_COLUMN="acc_gyro_array"
ACC_ARRAY_COLUMN="acc_array"
GYRO_ARRAY_COLUMN="gyro_array"