import pandas as pd
import tsfresh
from tsfresh.feature_extraction.feature_calculators import set_property
from tsfresh.feature_extraction import feature_calculators
from tsfresh.utilities.string_manipulation import convert_to_output_format
import numpy as np
from config.constants import ACC_ARRAY_COLUMN, ACC_GYRO_COLUMN, COLUMNS_ACCELEROMETER, COLUMNS_GYROSCOPE, COLUMNS_MAGNETOMETER,DEVICES, GYRO_ARRAY_COLUMN
from tsfresh import extract_features


################################Physicial features#######################################
# =============================================================================
# 1. INTENSIDADE DE MOVIMENTO (MI)
# =============================================================================
@set_property("fctype", "combiner")
def mi_intensity(acc_array,param):
    arr = np.vstack(acc_array).astype(float)  # transforma em shape (N,3)
    magn = np.linalg.norm(arr, axis=1)
    avg_intensity = float(np.mean(magn)) if magn.size else 0.0
    variance_intensity = float(np.var(magn)) if magn.size else 0.0
    return [("avg",avg_intensity),("var",variance_intensity)]

# =============================================================================
# 2. SMA (Signal Magnitude Area)
# =============================================================================

@set_property("fctype", "simple")
def sma(acc_array):
    arr = np.vstack(acc_array).astype(float)
    sma_vals = np.sum(np.abs(arr), axis=1)
    return float(np.mean(sma_vals)) if sma_vals.size else 0.0


# =============================================================================
# 3. VALORES PRÓPRIOS EVA
# =============================================================================
@set_property("fctype", "combiner")
def EVA(acc_array,param):
    arr = np.vstack(acc_array).astype(float)
    if arr.shape[0] < 2:
        return [("ver", 0.0),("head", 0.0)]
    cov = np.cov(arr, rowvar=False)
    vals = np.linalg.eigvalsh(cov)
    vals_sorted = np.sort(vals)
    return [("ver", float(vals_sorted[0])), ("head", float(vals_sorted[-1]))]


# =============================================================================
# 4. CAGH – Correlação gravidade (aprox eixo X) × heading (norma YZ)
# =============================================================================

@set_property("fctype", "simple")
def cagh(acc_array):
    arr = np.vstack(acc_array).astype(float)
    if arr.shape[1] < 3:
        return 0.0
    ax = arr[:, 0]              # direção da gravidade
    heading = np.linalg.norm(arr[:, 1:3], axis=1)
    if len(heading) < 2:
        return 0.0
    # derivada discreta da heading
    heading_deriv = np.diff(heading)
    ax_trunc = ax[1:]
    if np.std(ax_trunc) == 0 or np.std(heading_deriv) == 0:
        return 0.0
    return float(np.corrcoef(ax_trunc, heading_deriv)[0, 1])


# =============================================================================
# 5. AVH – Velocidade média ao longo da direção de avanço (heading)
# =============================================================================

@set_property("fctype", "simple")
def avh(acc_array):
    arr = np.vstack(acc_array).astype(float)
    if arr.shape[1] < 3:
        return 0.0
    heading_acc = np.linalg.norm(arr[:, 1:3], axis=1)
    # integração sem dt → acumulado discreto (unidades arbitrárias)
    velocity = np.cumsum(heading_acc)

    return float(np.mean(np.abs(velocity))) if velocity.size else 0.0


# =============================================================================
# 6. AVG – Velocidade média ao longo da direção da gravidade (aprox eixo X)
# =============================================================================

@set_property("fctype", "simple")
def avg(acc_array):
    arr = np.vstack(acc_array).astype(float)
    ax = arr[:, 0]
    velocity = np.cumsum(ax)
    return float(np.mean(np.abs(velocity))) if velocity.size else 0.0


# =============================================================================
# 7. ARATG – Ângulos médios de rotação (usa gyro_x como rotação no eixo gravidade)
# =============================================================================

@set_property("fctype", "simple")
def aratg(gyro_array):
    arr = np.vstack(gyro_array).astype(float)
    gx = arr[:, 0]  # rotação em torno da gravidade
    angle_increment = np.abs(gx)  # sem dt → soma discreta
    return float(np.mean(angle_increment)) if angle_increment.size else 0.0


# =============================================================================
# 8. DF – Dominant Frequency (por eixo individual)
# =============================================================================

@set_property("fctype", "simple")
def dominant_frequency_axis(signal_1d):
    arr = np.asarray(signal_1d, dtype=float)
    if arr.size < 2:
        return 0.0
    arr = arr - np.mean(arr)
    fft_vals = np.fft.rfft(arr)
    power = np.abs(fft_vals)**2
    power[0] = 0  # excluir DC
    idx = np.argmax(power)
    return float(idx)  # índice do pico (freq ≈ idx se dt=1)


# =============================================================================
# 9. ENERGY – Energia por eixo
# =============================================================================

@set_property("fctype", "simple")
def energy_axis(signal_1d):
    s = np.asarray(signal_1d, dtype=float)
    if s.size < 2:
        return 0.0
    s = s - np.mean(s)
    fft_vals = np.fft.rfft(s)
    power = np.abs(fft_vals)**2
    power[0] = 0
    return float(np.sum(power))


# =============================================================================
# 10. AAE – Energia média dos 3 eixos do acelerómetro
# =============================================================================

@set_property("fctype", "simple")
def aae(acc_array):
    arr = np.vstack(acc_array).astype(float)
    energy = np.sum(arr**2, axis=1)  # ax² + ay² + az²
    return float(np.mean(energy)) if energy.size else 0.0


# =============================================================================
# 11. ARE – Energia média dos 3 eixos do giroscópio
# =============================================================================
@set_property("fctype", "simple")
def are(gyro_array):
    arr = np.vstack(gyro_array).astype(float)
    energy = np.sum(arr**2, axis=1)

    return float(np.mean(energy)) if energy.size else 0.0

#####################Features estatisticas######################################

@set_property("fctype", "simple")
def mean_crossing_rate(x):
    """
    Número de cruzamentos da série com a sua média (Mean Crossing Rate).
    """
    arr = np.asarray(x, dtype=float)
    mean_val = np.mean(arr)
    # vetor booleano indicando se cada ponto está acima ou abaixo da média
    above = arr > mean_val
    # cruzamentos acontecem quando o estado muda de True → False ou False → True
    crossings = np.sum(above[:-1] != above[1:])
    return int(crossings)


@set_property("fctype", "simple")
def spectral_entropy(x: pd.Series) -> float:
    """
    Calcula a entropia espectral da magnitude 3D (acc_array).
    """
    return 1
@set_property("fctype", "simple")
def interq_range(x):
    """
    Amplitude Interquartil (IQR = P75 - P25)
    """
    arr = np.asarray(x, dtype=float)

    if arr.size == 0:
        return 0.0

    q75 = np.percentile(arr, 75)
    q25 = np.percentile(arr, 25)
    return float(q75 - q25)


@set_property("fctype", "simple")
def averaged_derivative(x: pd.Series,**kwargs) -> float:
    """
    Derivada média absoluta de um sinal 1D.
    Usada em reconhecimento de fala e escrita.
    """
    arr = np.asarray(x, dtype=float)

    if arr.size < 2:
        return 0.0

    # primeira derivada: |x[i] - x[i-1]|
    derivative = np.abs(np.diff(arr))

    return float(np.mean(derivative))



@set_property("fctype", "combiner")
def pairwise_correlation(x, param):
    arr = np.array(x.tolist())  # N x 6
    acc = arr[:, :3]
    gyro = arr[:, 3:]
    
    features = []
    
    # Acc interno
    corr_acc = np.corrcoef(acc, rowvar=False)
    features.append(('corr_acc_xy', corr_acc[0,1]))
    features.append(('corr_acc_xz', corr_acc[0,2]))
    features.append(('corr_acc_yz', corr_acc[1,2]))
    
    # Gyro interno
    corr_gyro = np.corrcoef(gyro, rowvar=False)
    features.append(('corr_gyro_xy', corr_gyro[0,1]))
    features.append(('corr_gyro_xz', corr_gyro[0,2]))
    features.append(('corr_gyro_yz', corr_gyro[1,2]))
    
    # Acc vs Gyro
    axes = ["x","y","z"]
    for i, a in enumerate(axes):
        for j, g in enumerate(axes):
            features.append((f'acc_{a}_gyro_{g}', np.corrcoef(acc[:,i], gyro[:,j])[0,1]))    
    return [(name, val) for name,val in features]



setattr(tsfresh.feature_extraction.feature_calculators, 'mean_crossing_rate', mean_crossing_rate)
setattr(tsfresh.feature_extraction.feature_calculators, 'averaged_derivative', averaged_derivative)
setattr(tsfresh.feature_extraction.feature_calculators, 'spectral_entropy', spectral_entropy)
setattr(tsfresh.feature_extraction.feature_calculators,"pairwise_correlation",pairwise_correlation)
setattr(tsfresh.feature_extraction.feature_calculators,"interq_range",interq_range)

setattr(tsfresh.feature_extraction.feature_calculators,"mi_intensity", mi_intensity)
setattr(tsfresh.feature_extraction.feature_calculators,"sma", sma)
setattr(tsfresh.feature_extraction.feature_calculators,"EVA", EVA)
setattr(tsfresh.feature_extraction.feature_calculators,"cagh",cagh)
setattr(tsfresh.feature_extraction.feature_calculators,"avh",avh)
setattr(tsfresh.feature_extraction.feature_calculators,"aae",aae)
setattr(tsfresh.feature_extraction.feature_calculators,"avg",avg)
setattr(tsfresh.feature_extraction.feature_calculators,"aratg",aratg)
setattr(tsfresh.feature_extraction.feature_calculators,"are",are)
setattr(tsfresh.feature_extraction.feature_calculators,"dominant_frequency_axis",dominant_frequency_axis)
setattr(tsfresh.feature_extraction.feature_calculators,"energy_axis",energy_axis)


def create_sliding_windows(df, window_size, step_size):
    windows = []

    for device_id in DEVICES:
        df_sensor = df[df['device_id'] == device_id].copy()
        window_id = 0
        
        for start in range(0, len(df_sensor) - window_size + 1, step_size):
            end = start + window_size
            window = df_sensor.iloc[start:end].copy()
            window["window_id"] = window_id
            window["device_id"] = device_id
            
            # Arrays para funções combiner
            window[ACC_ARRAY_COLUMN] = window.apply(lambda row: np.array([
                row['accelerometer_x'],
                row['accelerometer_y'],
                row['accelerometer_z']
            ]), axis=1)
            
            window[GYRO_ARRAY_COLUMN] = window.apply(lambda row: np.array([
                row['gyroscope_x'],
                row['gyroscope_y'],
                row['gyroscope_z']
            ]), axis=1)
            window[ACC_GYRO_COLUMN] = window.apply(lambda row: np.array([
                row['accelerometer_x'],
                row['accelerometer_y'],
                row['accelerometer_z'],
                row['gyroscope_x'],
                row['gyroscope_y'],
                row['gyroscope_z']
            ]), axis=1)
            
            windows.append(window)
            window_id += 1
    
    return pd.concat(windows, ignore_index=True)



def to_long_format(df):
    df = df.copy()
    df['time'] = df.groupby(['device_id','window_id']).cumcount()
    # id único combinando sensor + janela
    df['id'] = df['device_id'].astype(str) + "_" + df['window_id'].astype(str)
    
    long_df = df.melt(
        id_vars=['id', 'time'],
        value_vars=COLUMNS_ACCELEROMETER + COLUMNS_GYROSCOPE + COLUMNS_MAGNETOMETER + [ACC_GYRO_COLUMN, ACC_ARRAY_COLUMN, GYRO_ARRAY_COLUMN],
        var_name='kind',
        value_name='value'
    )
    
    return long_df


def df_to_tsfresh_format(df):
    long_df = pd.DataFrame()
    
    for col in df.columns:
        if col in ["window_id", "time", "device_id", "id"]:
            continue
        
        tmp = pd.DataFrame({
            "id": df["id"],
            "time": df["time"],
            "kind": col,  # acc_x, acc_y, etc.
            "value": df[col]
        })
        
        long_df = pd.concat([long_df, tmp], ignore_index=True)

    return long_df


def extract_features_sensors(dataframe, window_size, step_size):
    # 1) Criar janelas
    df_windowed = create_sliding_windows(dataframe, window_size, step_size)

    # 2) Selecionar colunas necessárias
    df = df_windowed[
        COLUMNS_ACCELEROMETER +
        COLUMNS_GYROSCOPE +
        COLUMNS_MAGNETOMETER +
        ["device_id","window_id", ACC_GYRO_COLUMN, ACC_ARRAY_COLUMN, GYRO_ARRAY_COLUMN]
    ]
    long_df = to_long_format(df)
    

    stat_features = {
        "mean": None,
        "median": None,
        "standard_deviation": None,
        "variance": None,
        "root_mean_square": None,
        "skewness": None,
        "kurtosis": None,
        "number_crossing_m": [{"m": 0}],   # zero-crossing
        "mean_crossing_rate": None,
        "averaged_derivative": None,
        "spectral_entropy": None,
        "interq_range": None,
    }

    # aplicar estatísticas a todos os eixos individuais
    kind_to_fc_parameters = {k: stat_features.copy()
                             for k in COLUMNS_ACCELEROMETER + COLUMNS_GYROSCOPE}

    # features combinadas acc+gyro
    kind_to_fc_parameters[ACC_GYRO_COLUMN] = {
        "pairwise_correlation": None
    }

    # ---------------------------------------------------------
    # 5. FEATURE SET – FÍSICAS
    # ---------------------------------------------------------
    physical_features = {
        "dominant_frequency_axis": None,
        "energy_axis": None
    }

    # adicionar físicas aos eixos individuais
    for k in COLUMNS_ACCELEROMETER + COLUMNS_GYROSCOPE:
        kind_to_fc_parameters[k].update(physical_features)

    # 5.1 — features específicas para acc_array (3D)
    kind_to_fc_parameters[ACC_ARRAY_COLUMN] = {
        "mi_intensity": None,
        "sma": None,
        "EVA": None,
        "cagh": None,
        "avh": None,
        "aae": None,
    }

    # 5.2 — gyroscope_x recebe AVG
    if "gyroscope_x" in kind_to_fc_parameters:
        kind_to_fc_parameters["gyroscope_x"].update({"avg": None})

    # 5.3 — gyro_array recebe ARE e ARATG
    kind_to_fc_parameters[GYRO_ARRAY_COLUMN] = {
        "aratg": None,
        "are": None
    }

    # ---------------------------------------------------------
    # 6. EXTRAÇÃO FINAL DE FEATURES
    # ---------------------------------------------------------
    print("Extracting features...")
    features = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        kind_to_fc_parameters=kind_to_fc_parameters,
        n_jobs=0
    )
    return features