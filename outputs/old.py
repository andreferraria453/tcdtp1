def create_sliding_windows(dataframe, window_ms, step_ms, sampling_rate):
    # 1. Validação Básica
    if dataframe.empty or 'timestamp' not in dataframe.columns:
        return pd.DataFrame()

    df = dataframe.copy()
    
    # Detetar coluna de label
    if 'activity_label' in df.columns: lbl_col = 'activity_label'
    elif 'actitivy_label' in df.columns: lbl_col = 'actitivy_label'
    else: lbl_col = 'activity'
    
    # 2. SINCRONIZAÇÃO
    period_ms = int(1000 / sampling_rate)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['grid_time'] = df['datetime'].dt.round(f'{period_ms}ms')

    total_devices = df['device_id'].nunique()
    time_counts = df.groupby('grid_time')['device_id'].nunique()
    valid_times = time_counts[time_counts == total_devices].index
    
    # Manter apenas tempos onde TODOS os devices têm dados
    df_clean = df[df['grid_time'].isin(valid_times)].sort_values(['grid_time', 'device_id'])
    
    if df_clean.empty:
        return pd.DataFrame()

    # 3. JANELAMENTO
    unique_times = df_clean['grid_time'].unique()
    
    # Converter ms para número de amostras
    samples_per_window = int(window_ms / period_ms)
    samples_step = int(step_ms / period_ms)
    
    # Segurança: garantir que step não é zero
    if samples_step < 1: samples_step = 1

    windows_list = []
    w_id = 0
    
    expected_duration = np.timedelta64(int(window_ms), 'ms')
    tolerance = np.timedelta64(100, 'ms') 

    for i in range(0, len(unique_times) - samples_per_window + 1, samples_step):
        start_time = unique_times[i]
        end_time = unique_times[i + samples_per_window - 1]
        
        # Verificar Gaps
        actual_duration = end_time - start_time
        if actual_duration > (expected_duration + tolerance):
            continue 

        # Extrair dados
        window_mask = (df_clean['grid_time'] >= start_time) & (df_clean['grid_time'] <= end_time)
        chunk = df_clean[window_mask].copy()

        # Verificar Pureza da Label
        if chunk[lbl_col].nunique() > 1:
            continue
            
        chunk['window_id'] = w_id
        windows_list.append(chunk)
        w_id += 1

    if not windows_list:
        return pd.DataFrame()

    # 4. Finalizar
    df_final = pd.concat(windows_list, ignore_index=True)
    df_final = df_final.drop(columns=['datetime', 'grid_time'])

    # Recriar Arrays
    acc_cols = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z']
    gyro_cols = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    mag_cols = ['magnetometer_x', 'magnetometer_y', 'magnetometer_z']
    
    if all(c in df_final.columns for c in acc_cols):
        df_final['acc_array'] = df_final[acc_cols].values.tolist()
    if all(c in df_final.columns for c in gyro_cols):
        df_final['gyro_array'] = df_final[gyro_cols].values.tolist()
    if all(c in df_final.columns for c in mag_cols):
        df_final['mag_array'] = df_final[mag_cols].values.tolist()
    if all(c in df_final.columns for c in acc_cols + gyro_cols):
        df_final['acc_gyro_array'] = df_final[acc_cols + gyro_cols].values.tolist()

    return df_final