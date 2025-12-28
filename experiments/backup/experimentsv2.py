import copy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from itertools import product
from skrebate import ReliefF
from metrics.metrics import compute_accuracy, compute_f1_score, compute_precision, compute_recall
from data.data_splitting import split_data_kfold

def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": compute_accuracy(y_true, y_pred),
        "Recall": compute_recall(y_true, y_pred, average='macro'),
        "Precision": compute_precision(y_true, y_pred, average='macro'),
        "F1": compute_f1_score(y_true, y_pred, average='macro')
    }

def split_subjects_kfold(subjects, n_splits=10, random_state=42):
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    np.random.seed(random_state)
    shuffled = unique_subjects.copy()
    np.random.shuffle(shuffled)
    
    folds = []
    fold_size = n_subjects // n_splits
    
    for i in range(n_splits):
        start_idx = i * fold_size
        if i == n_splits - 1:
            test_subjects = shuffled[start_idx:]
        else:
            test_subjects = shuffled[start_idx:start_idx + fold_size]
        
        train_subjects = np.setdiff1d(shuffled, test_subjects)
        folds.append((train_subjects, test_subjects))
    
    return folds

# --- CLASSE MODIFICADA ---

class Experiment:
    def __init__(self, X, y, models, model_parameters, labels=None, subjects=None, scaler=None):
        """
        Adicionado parametro: scaler (ex: StandardScaler())
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.models = models
        self.model_parameters = model_parameters
        self.labels = labels if labels is not None else np.unique(y)
        self.random_state = 10
        self.subjects = np.array(subjects) if subjects is not None else None
        self.subject_aware_mode = (self.subjects is not None)
        self.scaler = scaler  # Novo atributo para guardar o scaler original

        if self.subject_aware_mode:
            unique_subjects = np.unique(self.subjects)
            print(f"{len(unique_subjects)} participantes detectados: {unique_subjects}")
        
        if self.scaler:
            print(f"Scaler detetado: {type(self.scaler).__name__} (será ajustado apenas ao treino em cada split).")

    def _generate_configs(self, grid):
        keys = list(grid.keys())
        values = list(grid.values())
        for combo in product(*values):
            yield dict(zip(keys, combo))

    # -------------------------------------------------------
    # NOVO: Método auxiliar para aplicar scaling
    # -------------------------------------------------------
    def _scale_data(self, X_train, *X_others):
        """
        Ajusta (fit) no X_train e transforma X_train e as outras matrizes (val, test).
        Retorna as matrizes transformadas na mesma ordem.
        """
        if self.scaler is None:
            # Se não houver scaler, retorna como entrou
            return (X_train, *X_others)
        
        # É crucial usar deepcopy para não "sujar" o scaler original com fits anteriores
        sc = copy.deepcopy(self.scaler)
        
        # Fit apenas no TREINO
        X_train_scaled = sc.fit_transform(X_train)
        
        # Transform nos restantes (Validação, Teste, etc)
        X_others_scaled = [sc.transform(x) for x in X_others]
        
        return (X_train_scaled, *X_others_scaled)

    # -------------------------------------------------------
    # Split manual por participantes
    # -------------------------------------------------------
    def split_tvt_manual(self, train_subjects, val_subjects, test_subjects):
        if not self.subject_aware_mode:
            raise ValueError("split_tvt_manual() requer subjects no construtor!")
        
        train_mask = np.isin(self.subjects, train_subjects)
        val_mask = np.isin(self.subjects, val_subjects)
        test_mask = np.isin(self.subjects, test_subjects)
        
        X_train, y_train = self.X[train_mask], self.y[train_mask]
        X_val, y_val = self.X[val_mask], self.y[val_mask]
        X_test, y_test = self.X[test_mask], self.y[test_mask]
        
        # O Scaling não é feito aqui, mas sim dentro dos métodos run_*, 
        # pois aqui só retornamos os dados crus divididos.
        
        print(f"Split manual aplicado:")
        print(f"  Treino: {len(train_subjects)} suj -> {len(X_train)} amostras")
        print(f"  Val:    {len(val_subjects)} suj -> {len(X_val)} amostras")
        print(f"  Teste:  {len(test_subjects)} suj -> {len(X_test)} amostras\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    # -------------------------------------------------------
    # Split TVT automático
    # -------------------------------------------------------
    def split_tvt(self, val_size=0.2, test_size=0.2, random_state=0):
        if self.subject_aware_mode:
            unique_subjects = np.unique(self.subjects)
            np.random.seed(random_state)
            np.random.shuffle(unique_subjects)
            
            n_test = max(1, int(len(unique_subjects) * test_size))
            n_val = max(1, int(len(unique_subjects) * val_size))
            
            test_subjects = unique_subjects[:n_test]
            val_subjects = unique_subjects[n_test:n_test+n_val]
            train_subjects = unique_subjects[n_test+n_val:]
            
            return self.split_tvt_manual(train_subjects, val_subjects, test_subjects)
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                self.X, self.y, test_size=test_size, stratify=self.y, random_state=random_state
            )
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
            )
            return X_train, X_val, X_test, y_train, y_val, y_test

    # -------------------------------------------------------
    # Run Train-Only
    # -------------------------------------------------------
    def run_train_only(self):
        results = []
        cms = []
        
        # Scaling no dataset inteiro (já que é train-only, não há fuga para teste)
        if self.scaler:
            sc = copy.deepcopy(self.scaler)
            X_processed = sc.fit_transform(self.X)
        else:
            X_processed = self.X

        for name, model in self.models.items():
            param_grid = self.model_parameters.get(name, {})
            for params in self._generate_configs(param_grid):
                model.set_params(**params)
                model.fit(X_processed, self.y)
                pred = model.predict(X_processed)

                metrics = compute_metrics(self.y, pred)
                cm = confusion_matrix(self.y, pred, labels=self.labels)
                results.append({"Model": name, **params, **metrics})
                cms.append(cm)

        return pd.DataFrame(results), cms, np.sum(cms, axis=0)

    # -------------------------------------------------------
    # Run Train-Test (COM SCALING)
    # -------------------------------------------------------
    def run_train_test(self, test_size=0.3, random_state=None):
        if random_state is None: random_state = self.random_state
            
        if self.subject_aware_mode:
            unique_subjects = np.unique(self.subjects)
            np.random.seed(random_state)
            np.random.shuffle(unique_subjects)
            n_test = max(1, int(len(unique_subjects) * test_size))
            test_subjects = unique_subjects[:n_test]
            train_subjects = unique_subjects[n_test:]
            
            train_mask = np.isin(self.subjects, train_subjects)
            test_mask = np.isin(self.subjects, test_subjects)
            X_train, y_train = self.X[train_mask], self.y[train_mask]
            X_test, y_test = self.X[test_mask], self.y[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, stratify=self.y, random_state=random_state
            )

        # --- APLICAR SCALER ---
        # Fit no Train, Transform no Train e Test
        X_train, X_test = self._scale_data(X_train, X_test)

        results = []
        cms = []
        for name, model in self.models.items():
            param_grid = self.model_parameters.get(name, {})
            for params in self._generate_configs(param_grid):
                model.set_params(**params)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                metrics = compute_metrics(y_test, pred)
                cm = confusion_matrix(y_test, pred, labels=self.labels)
                results.append({"Model": name, **params, **metrics})
                cms.append(cm)

        return pd.DataFrame(results), cms, np.sum(cms, axis=0)

    # -------------------------------------------------------
    # Cross-Validation (COM SCALING POR FOLD)
    # -------------------------------------------------------
    def run_cross_validation(self, n_splits=10, n_repeats=10):
            all_results = []
            cm_stats = {}
            for name, model in self.models.items():
                param_grid = self.model_parameters.get(name, {})
                
                for params in self._generate_configs(param_grid):
                    
                    # --- RESET CRÍTICO ---
                    # Estas listas reiniciam para CADA combinação de parâmetros.
                    # Isto garante que não misturas dados do k=1 com o k=3.
                    fold_metrics = {"Accuracy": [], "Recall": [], "Precision": [], "F1": []}
                    fold_cms = []

                    for repeat in range(n_repeats):
                        # Escolha do Splitter (Subject ou Data)
                        if self.subject_aware_mode:
                            folds = split_subjects_kfold(self.subjects, n_splits=n_splits, random_state=self.random_state + repeat)
                        else:
                            folds = split_data_kfold(self.X, n_splits=n_splits, random=repeat)

                        for fold_idx, (train_idx, test_idx) in enumerate(folds):
                            # 1. Aplicar Split (Máscaras ou Índices)
                            if self.subject_aware_mode:
                                train_mask = np.isin(self.subjects, train_idx)
                                test_mask = np.isin(self.subjects, test_idx)
                                X_train, X_test = self.X[train_mask], self.X[test_mask]
                                y_train, y_test = self.y[train_mask], self.y[test_mask]
                            else:
                                X_train, X_test = self.X[train_idx], self.X[test_idx]
                                y_train, y_test = self.y[train_idx], self.y[test_idx]
                            
                            # 2. Aplicar Scaler (Fit no Treino, Transform no Treino e Teste)
                            X_train, X_test = self._scale_data(X_train, X_test)

                            # 3. Treino e Predição
                            model.set_params(**params)
                            model.fit(X_train, y_train)
                            pred = model.predict(X_test)

                            # 4. Guardar Métricas e CM deste Fold
                            m = compute_metrics(y_test, pred)
                            cm = confusion_matrix(y_test, pred, labels=self.labels)

                            for k in fold_metrics: fold_metrics[k].append(m[k])
                            fold_cms.append(cm) # Adiciona a matriz deste fold à lista

                    # --- AGREGAÇÃO POR MODELO/PARAMETRO ---
                    
                    # Guarda as métricas médias no DataFrame
                    config_result = {
                        "Model": name, **params,
                        "Acc_Mean": np.mean(fold_metrics["Accuracy"]), "Acc_Std": np.std(fold_metrics["Accuracy"]),
                        "Prec_Mean": np.mean(fold_metrics["Precision"]), "Prec_Std": np.std(fold_metrics["Precision"]),
                        "Rec_Mean": np.mean(fold_metrics["Recall"]), "Rec_Std": np.std(fold_metrics["Recall"]),
                        "F1_Mean": np.mean(fold_metrics["F1"]), "F1_Std": np.std(fold_metrics["F1"])
                    }
                    all_results.append(config_result)

                    # Guarda as Matrizes de Confusão no Dicionário
                    cms_array = np.array(fold_cms) # Transforma lista em array 3D (n_folds, 7, 7)
                    
                    config_id = f"{name}_{str(params)}"
                    
                    cm_stats[config_id] = {
                        # A média das células (útil para ver tendências)
                        "mean": np.mean(cms_array, axis=0),
                        
                        # O desvio padrão (útil para ver estabilidade)
                        "std": np.std(cms_array, axis=0),
                        
                        # O TOTAL (Soma de todos os folds). ESTA É A QUE TU QUERES VER.
                        "total": np.sum(cms_array, axis=0),
                        
                        # AS MATRIZES INDIVIDUAIS (Para veres fold a fold se quiseres)
                        "folds": cms_array
                    }

            df_results = pd.DataFrame(all_results)
            
            # Retorna apenas o DataFrame e o Dicionário completo.
            # Não retornamos "total" solto para evitar confusões.
            return df_results, cm_stats
    # -------------------------------------------------------
    # TVT com feature selection (COM SCALING)
    # -------------------------------------------------------
    def run_tvt_with_feature_selection(self, val_size=0.2, test_size=0.2, 
                                       feature_ranking=None, train_subjects=None,
                                       val_subjects=None, test_subjects=None):
        
        # 1. Split dos dados
        if train_subjects is not None:
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt_manual(
                train_subjects, val_subjects, test_subjects
            )
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt(
                val_size=val_size, test_size=test_size
            )
        
        # --- APLICAR SCALER ---
        # Crucial: Scale ANTES do ReliefF
        # Fit no Train, Transform em Train, Val e Test
        X_train, X_val, X_test = self._scale_data(X_train, X_val, X_test)

        # 2. Feature ranking (agora com dados escalados)
        if feature_ranking is None:
            print("⚙️  Calculando feature ranking com ReliefF (dados escalados)...")
            relief = ReliefF(n_neighbors=100, n_jobs=-1)
            relief.fit(X_train, y_train)
            ranking = np.argsort(-relief.feature_importances_)
            print("✓ ReliefF concluído\n")
        else:
            print("✓ Usando feature ranking pré-computado\n")
            ranking = feature_ranking
        
        num_features = X_train.shape[1]
        models_results = []
        
        for name, model_original in self.models.items():
            print(f"Modelo: {name}")
            model_results = {}
            param_grid = self.model_parameters.get(name, {})

            # A. Feature Selection Loop (dados já estão escalados)
            validation_f1s_model = []
            for k in range(1, num_features + 1):
                model = copy.deepcopy(model_original)
                selected_features = ranking[:k]
                
                model.fit(X_train[:, selected_features], y_train)
                y_val_pred = model.predict(X_val[:, selected_features])
                f1_val = compute_metrics(y_val, y_val_pred)["F1"]
                
                validation_f1s_model.append({
                    "model": name, "F1": f1_val, "k_features": k, "selected_features": selected_features.tolist()
                })
            
            model_results["validation"] = validation_f1s_model
            
            # Melhor k
            f1_vals = [entry['F1'] for entry in validation_f1s_model]
            best_k = validation_f1s_model[np.argmax(f1_vals)]['k_features']
            best_features_indices = ranking[:best_k]
            
            # B. Param Optimization Loop
            best_params = None
            best_f1_param = -1
            
            for params in self._generate_configs(param_grid):
                model = copy.deepcopy(model_original)
                model.set_params(**params)
                model.fit(X_train[:, best_features_indices], y_train)
                y_val_pred = model.predict(X_val[:, best_features_indices])
                f1 = compute_metrics(y_val, y_val_pred)["F1"]
                
                if f1 > best_f1_param:
                    best_f1_param = f1
                    best_params = params
            
            models_results.append(model_results)
            
            # C. Final Evaluation
            # Juntamos Train + Val para o fit final
            X_trainval = np.vstack([X_train, X_val])
            y_trainval = np.hstack([y_train, y_val])
            
            model.set_params(**best_params)
            model.fit(X_trainval[:, best_features_indices], y_trainval)
            y_test_pred = model.predict(X_test[:, best_features_indices])
            f1_test = compute_metrics(y_test, y_test_pred)["F1"]
            
            model_results["test"] = {
                "model": name, "F1": f1_test, "k_features": best_k, "params": best_params
            }
            print(f"  ✓ Teste final: F1={f1_test:.4f}\n")

        return models_results

    # -------------------------------------------------------
    # Cross-Validation com Feature Selection (COM SCALING DUPLO)
    # -------------------------------------------------------
    def run_cross_with_validation_feature_selection(self, number_of_folds=10, 
                                                    n_repeats=1, test_size=0.3,
                                                    feature_ranking=None):
        all_results = []
        
        for repeat in range(n_repeats):
            print(f"\n[Repetição {repeat+1}/{n_repeats}]")
            current_random_state = self.random_state + repeat
            
            if self.subject_aware_mode:
                folds = split_subjects_kfold(self.subjects, n_splits=number_of_folds, random_state=current_random_state)
            else:
                folds = split_data_kfold(self.X, n_splits=number_of_folds, random=current_random_state)
            
            for fold_id, fold_data in enumerate(folds):
                print(f"  > Fold {fold_id+1}/{number_of_folds}")
                
                # --- 1. Split Outer (Treino vs Teste) ---
                if self.subject_aware_mode:
                    train_subjects, test_subjects = fold_data
                    train_mask = np.isin(self.subjects, train_subjects)
                    test_mask = np.isin(self.subjects, test_subjects)
                    X_train, X_test = self.X[train_mask], self.X[test_mask]
                    y_train, y_test = self.y[train_mask], self.y[test_mask]
                    
                    # Para o split interno (TrS vs Val)
                    unique_train = np.unique(self.subjects[train_mask])
                    np.random.seed(current_random_state)
                    np.random.shuffle(unique_train)
                    n_val = max(1, int(len(unique_train) * test_size))
                    val_subjects_inner = unique_train[:n_val]
                    trs_subjects_inner = unique_train[n_val:]
                    
                    trs_mask = np.isin(self.subjects[train_mask], trs_subjects_inner)
                    val_mask = np.isin(self.subjects[train_mask], val_subjects_inner)
                    
                    X_TrS = X_train[trs_mask]
                    y_TrS = y_train[trs_mask]
                    X_V = X_train[val_mask]
                    y_V = y_train[val_mask]

                else:
                    train_idx, test_idx = fold_data
                    X_train, X_test = self.X[train_idx], self.X[test_idx]
                    y_train, y_test = self.y[train_idx], self.y[test_idx]
                    
                    X_TrS, X_V, y_TrS, y_V = train_test_split(
                        X_train, y_train, test_size=test_size, stratify=y_train, random_state=current_random_state
                    )

                # --- 2. Scaling (Critico: Precisa de ser feito em 2 fases) ---
                
                # Fase A: Scaling Interno (para validação de features/params)
                # O Scaler aprende com TrS (sub-treino) e aplica em TrS e Validação
                X_TrS_sc, X_V_sc = self._scale_data(X_TrS, X_V)
                
                # Fase B: Scaling Externo (para teste final do fold)
                # O Scaler aprende com Train COMPLETO (TrS+V) e aplica em Train e Test
                X_train_sc, X_test_sc = self._scale_data(X_train, X_test)

                # Feature ranking (usando dados internos escalados)
                if feature_ranking is None:
                    relief = ReliefF(n_neighbors=100)
                    relief.fit(X_TrS_sc, y_TrS)
                    ranking = np.argsort(-relief.feature_importances_)
                else:
                    ranking = feature_ranking
                
                num_features = X_train.shape[1]
                fold_results = {"repetition": repeat, "fold_id": fold_id, "models": []}
                
                for name, model_original in self.models.items():
                    param_grid = self.model_parameters.get(name, {})
                    chaves = list(param_grid.keys())
                    defaults = model_original.get_params()
                    default_params = {k: defaults.get(k) for k in chaves}
                    
                    scores_validacao_features = []
                    best_k = 1
                    best_f1_k = -1
                    
                    # Seleção de Features (Usa X_TrS_sc e X_V_sc)
                    for k in range(1, num_features + 1):
                        model = copy.deepcopy(model_original)
                        model.set_params(**default_params)
                        sel_feat = ranking[:k]
                        
                        model.fit(X_TrS_sc[:, sel_feat], y_TrS)
                        y_pred_val = model.predict(X_V_sc[:, sel_feat])
                        f1 = compute_metrics(y_V, y_pred_val)["F1"]
                        
                        scores_validacao_features.append({"k": k, "F1": f1})
                        if f1 > best_f1_k:
                            best_f1_k = f1
                            best_k = k
                    
                    best_feats = ranking[:best_k]
                    
                    # Otimização Parametros (Usa X_TrS_sc e X_V_sc com best_feats)
                    best_params = default_params
                    best_f1_p = best_f1_k
                    scores_validacao_params = []

                    for params in self._generate_configs(param_grid):
                        model = copy.deepcopy(model_original)
                        model.set_params(**params)
                        
                        model.fit(X_TrS_sc[:, best_feats], y_TrS)
                        y_pred_val = model.predict(X_V_sc[:, best_feats])
                        f1 = compute_metrics(y_V, y_pred_val)["F1"]
                        
                        scores_validacao_params.append({"params": params, "F1": f1})
                        if f1 > best_f1_p:
                            best_f1_p = f1
                            best_params = params
                    
                    # Avaliação Final do Fold (Usa X_train_sc e X_test_sc)
                    # Aqui treinamos com TODO o treino (TrS + V) e testamos no Test
                    final_model = copy.deepcopy(model_original)
                    final_model.set_params(**best_params)
                    
                    final_model.fit(X_train_sc[:, best_feats], y_train)
                    y_pred_test = final_model.predict(X_test_sc[:, best_feats])
                    
                    metrics = compute_metrics(y_test, y_pred_test)
                    cm = confusion_matrix(y_test, y_pred_test, labels=self.labels)
                    
                    fold_results["models"].append({
                        "model_name": name,
                        "best_k": best_k,
                        "best_params": best_params,
                        "test_metrics": metrics,
                        "cm": cm
                    })
                
                all_results.append(fold_results)
                
        return all_results