import os
import pandas as pd
from config.constants import FOLDER_PATH, DEVICES

def load_individual(participant_id):
    """Carrega dados de um indivíduo específico"""
    ind_data = pd.DataFrame([])
    for device_id in DEVICES:
        df = pd.read_csv(f"{FOLDER_PATH}{participant_id}/part{participant_id}dev{device_id}.csv")
        
        columns = [
            "device_id", "accelerometer_x", "accelerometer_y", "accelerometer_z",
            "gyroscope_x", "gyroscope_y", "gyroscope_z", "magnetometer_x", 
            "magnetometer_y", "magnetometer_z", "timestamp", "actitivy_label"
        ]
        
        df = pd.DataFrame(df.to_numpy(), columns=columns)
        df["participant_id"] = participant_id
        ind_data = pd.concat([ind_data, df])
    
    return ind_data

def load_complete_dataset(number_of_participants=14):
    """Carrega o dataset completo"""
    dataset = pd.DataFrame([])
    for i in range(number_of_participants + 1):
        ind_data = load_individual(i)
        dataset = pd.concat([ind_data, dataset])
    
    return dataset


# =============================================================================
# NOVA FUNÇÃO DE ANÁLISE DE TEMPO E GAPS
# =============================================================================

def format_ms(ms):
    """Converte ms para formato 'MMm SSs'"""
    seconds = ms / 1000
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

import pandas as pd
import os

def analyze_dataset_quality(number_of_participants=14, gap_threshold_ms=1000):
    """
    Analisa o tempo de início, procura falhas (gaps) e conta o número de registos.
    Retorna um DataFrame com o relatório.
    """
    report = []
    
    print(f"A analisar qualidade dos dados (Gaps > {gap_threshold_ms}ms)...")

    for p_id in range(number_of_participants + 1):
        for d_id in DEVICES:
            # Ajusta o path conforme a tua estrutura de pastas
            file_path = f"{FOLDER_PATH}part{p_id}/part{p_id}dev{d_id}.csv" 
            
            if not os.path.exists(file_path):
                continue
            
            try:
                # Ler apenas a coluna de timestamp (index 10) para ser rápido
                df_raw = pd.read_csv(file_path, header=None, usecols=[10])
                
                # --- NOVO: Contar número de linhas ---
                num_records = len(df_raw)
                
                time_col = df_raw.iloc[:, 0].sort_values()
                
                # 1. Tempos
                start_ms = time_col.min()
                end_ms = time_col.max()
                duration = end_ms - start_ms
                
                # 2. Gaps
                deltas = time_col.diff()
                gaps_count = (deltas > gap_threshold_ms).sum()
                max_gap = deltas.max() if not deltas.empty else 0
                
                # 3. Frequência Estimada
                valid_deltas = deltas[deltas <= gap_threshold_ms]
                avg_freq = 1000 / valid_deltas.mean() if valid_deltas.mean() > 0 else 0

                report.append({
                    "Participant": p_id,
                    "Device": d_id,
                    "Num Records": num_records,  # <--- Adicionado aqui
                    "Start Time": format_ms(start_ms),
                    "Duration": format_ms(duration),
                    "Gaps Detected": gaps_count,
                    "Max Gap (s)": round(max_gap/1000, 2),
                    "Est. Freq (Hz)": round(avg_freq, 1)
                })
                
            except Exception as e:
                print(f"Erro ao ler P{p_id} D{d_id}: {e}")

    return pd.DataFrame(report)