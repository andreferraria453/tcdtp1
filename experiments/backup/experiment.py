import copy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from metrics.metrics import compute_accuracy, compute_f1_score, compute_precision, compute_recall
from itertools import product
from skrebate import ReliefF
from data.data_splitting import split_data_kfold

def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": compute_accuracy(y_true, y_pred),
        "Recall": compute_recall(y_true, y_pred, average='weighted'),
        "Precision": compute_precision(y_true, y_pred, average='weighted'),
        "F1": compute_f1_score(y_true, y_pred, average='weighted')
    }


def split_subjects_kfold(subjects, n_splits=10, random_state=42):
    """
    Divide PARTICIPANTES em K folds (não amostras).
    
    Returns:
    --------
    List de (train_subjects, test_subjects) para cada fold
    """
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    
    np.random.seed(random_state)
    shuffled = unique_subjects.copy()
    np.random.shuffle(shuffled)
    
    # Divide participantes em K grupos
    folds = []
    fold_size = n_subjects // n_splits
    
    for i in range(n_splits):
        start_idx = i * fold_size
        if i == n_splits - 1:  # último fold pega o resto
            test_subjects = shuffled[start_idx:]
        else:
            test_subjects = shuffled[start_idx:start_idx + fold_size]
        
        train_subjects = np.setdiff1d(shuffled, test_subjects)
        folds.append((train_subjects, test_subjects))
    
    return folds


def create_feature_ranking(X, y, n_neighbors=100, subjects=None, train_subjects=None):
    """
    Cria ranking de features usando ReliefF.
    Opcionalmente pode treinar apenas com subset de participantes.
    
    Parameters:
    -----------
    X : array
        Features completas
    y : array
        Labels
    n_neighbors : int
        Parâmetro do ReliefF
    subjects : array, optional
        IDs dos participantes
    train_subjects : array, optional
        IDs dos participantes para treinar ReliefF (se None, usa todos)
    
    Returns:
    --------
    ranking : array
        Índices das features ordenadas por importância (maior -> menor)
    feature_importances : array
        Scores de importância de cada feature
    """
    if train_subjects is not None and subjects is not None:
        # Treina ReliefF apenas com participantes específicos
        mask = np.isin(subjects, train_subjects)
        X_train = X[mask]
        y_train = y[mask]
    else:
        X_train = X
        y_train = y
    
    relief = ReliefF(n_neighbors=n_neighbors, n_jobs=-1)
    relief.fit(X_train, y_train)
    ranking = np.argsort(-relief.feature_importances_)
    
    return ranking, relief.feature_importances_


class Experiment:
    """
    Classe de experimentos com suporte a subject-aware.
    Mudanças principais:
    - Métodos de split dividem PARTICIPANTES (não amostras)
    - Suporte a feature ranking pré-computado (evita recalcular ReliefF)
    """

    def __init__(self, X, y, models, model_parameters, labels=None, subjects=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.models = models
        self.model_parameters = model_parameters
        self.labels = labels if labels is not None else np.unique(y)
        self.random_state = 10
        self.subjects = np.array(subjects) if subjects is not None else None
        self.subject_aware_mode = (self.subjects is not None)
        
        if self.subject_aware_mode:
            unique_subjects = np.unique(self.subjects)
            print(f"{len(unique_subjects)} participantes detectados: {unique_subjects}")

    def _generate_configs(self, grid):
        keys = list(grid.keys())
        values = list(grid.values())
        for combo in product(*values):
            yield dict(zip(keys, combo))

    # -------------------------------------------------------
    # NOVO: Split manual por participantes para TVT
    # -------------------------------------------------------
    def split_tvt_manual(self, train_subjects, val_subjects, test_subjects):
        """
        Divide dados em Train/Val/Test usando listas MANUAIS de participantes.
        
        Parameters:
        -----------
        train_subjects : list/array
            IDs dos participantes para treino
        val_subjects : list/array
            IDs dos participantes para validação
        test_subjects : list/array
            IDs dos participantes para teste
        
        Returns:
        --------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        if not self.subject_aware_mode:
            raise ValueError("split_tvt_manual() requer subjects no construtor!")
        
        train_mask = np.isin(self.subjects, train_subjects)
        val_mask = np.isin(self.subjects, val_subjects)
        test_mask = np.isin(self.subjects, test_subjects)
        
        X_train, y_train = self.X[train_mask], self.y[train_mask]
        X_val, y_val = self.X[val_mask], self.y[val_mask]
        X_test, y_test = self.X[test_mask], self.y[test_mask]
        
        print(f"Split manual aplicado:")
        print(f"  Treino: {len(train_subjects)} participantes → {len(X_train)} amostras")
        print(f"  Val:    {len(val_subjects)} participantes → {len(X_val)} amostras")
        print(f"  Teste:  {len(test_subjects)} participantes → {len(X_test)} amostras\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    # -------------------------------------------------------
    # Split TVT automático (original adaptado)
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
            # Comportamento original
            X_temp, X_test, y_temp, y_test = train_test_split(
                self.X, self.y,
                test_size=test_size,
                stratify=self.y,
                random_state=random_state
            )

            val_ratio = val_size / (1 - test_size)

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio,
                stratify=y_temp,
                random_state=random_state
            )

            return X_train, X_val, X_test, y_train, y_val, y_test

    # -------------------------------------------------------
    # Train-Only (mantido original)
    # -------------------------------------------------------
    def run_train_only(self):
        results = []
        cms = []

        for name, model in self.models.items():
            param_grid = self.model_parameters.get(name, {})

            for params in self._generate_configs(param_grid):
                model.set_params(**params)
                model.fit(self.X, self.y)
                pred = model.predict(self.X)

                metrics = compute_metrics(self.y, pred)
                cm = confusion_matrix(self.y, pred, labels=self.labels)

                results.append({"Model": name, **params, **metrics})
                cms.append(cm)

        cm_total = np.sum(cms, axis=0)
        return pd.DataFrame(results), cms, cm_total

    # -------------------------------------------------------
    # Train/Test com split por participantes
    # -------------------------------------------------------
    def run_train_test(self, test_size=0.3, random_state=None):
        if random_state is None:
            random_state = self.random_state
            
        if self.subject_aware_mode:
            # Divide participantes
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
            
            print(f"Split por participantes: {len(train_subjects)} treino, {len(test_subjects)} teste")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y,
                test_size=test_size,
                stratify=self.y,
                random_state=random_state
            )

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

        cm_total = np.sum(cms, axis=0)
        return pd.DataFrame(results), cms, cm_total

    # -------------------------------------------------------
    # Cross-Validation com K-Fold manual de participantes
    # -------------------------------------------------------
    def run_cross_validation(self, n_splits=10, n_repeats=10):
            """
            Executa 10x10 CV (ou n_splits x n_repeats).
            Retorna:
                - df_results: DataFrame com Média e Std de cada métrica.
                - cm_stats: Dicionário com a Matriz Média e Matriz Std por modelo/config.
                - cm_total: Soma absoluta de todas as matrizes (opcional).
            """
            all_results = []
            cm_stats = {}

            for name, model in self.models.items():
                param_grid = self.model_parameters.get(name, {})

                for params in self._generate_configs(param_grid):
                    # Listas para guardar os 100 valores individuais
                    fold_metrics = {"Accuracy": [], "Recall": [], "Precision": [], "F1": []}
                    fold_cms = []

                    for repeat in range(n_repeats):
                        # Definir o split (Subject-aware ou Standard)
                        if self.subject_aware_mode:
                            folds = split_subjects_kfold(
                                self.subjects, 
                                n_splits=n_splits, 
                                random_state=self.random_state + repeat
                            )
                        else:
                            # Assume-se que existe uma função split_data_kfold para modo normal
                            folds = split_data_kfold(self.X, n_splits=n_splits, random=repeat)

                        for fold_idx, (train_idx, test_idx) in enumerate(folds):
                            # No modo subject_aware, train_idx e test_idx são LISTAS DE SUJEITOS
                            if self.subject_aware_mode:
                                train_mask = np.isin(self.subjects, train_idx)
                                test_mask = np.isin(self.subjects, test_idx)
                                X_train, X_test = self.X[train_mask], self.X[test_mask]
                                y_train, y_test = self.y[train_mask], self.y[test_mask]
                            else:
                                X_train, X_test = self.X[train_idx], self.X[test_idx]
                                y_train, y_test = self.y[train_idx], self.y[test_idx]
                            
                            # Treino e Predição
                            model.set_params(**params)
                            model.fit(X_train, y_train)
                            pred = model.predict(X_test)

                            # Métricas do Fold Atual
                            m = compute_metrics(y_test, pred)
                            cm = confusion_matrix(y_test, pred, labels=self.labels)

                            for metric_name in fold_metrics:
                                fold_metrics[metric_name].append(m[metric_name])
                            fold_cms.append(cm)

                    # --- PÓS-PROCESSAMENTO DOS 100 FOLDS ---
                    
                    # 1. Gerar linha de resultados com Média e Std
                    config_result = {
                        "Model": name,
                        **params,
                        "Acc_Mean": np.mean(fold_metrics["Accuracy"]),
                        "Acc_Std":  np.std(fold_metrics["Accuracy"]),
                        "Prec_Mean": np.mean(fold_metrics["Precision"]),
                        "Prec_Std":  np.std(fold_metrics["Precision"]),
                        "Rec_Mean":  np.mean(fold_metrics["Recall"]),
                        "Rec_Std":   np.std(fold_metrics["Recall"]),
                        "F1_Mean":   np.mean(fold_metrics["F1"]),
                        "F1_Std":    np.std(fold_metrics["F1"])
                    }
                    all_results.append(config_result)

                    # 2. Guardar estatísticas da Matriz de Confusão (Média e Std por célula)
                    cms_array = np.array(fold_cms) # Shape: (100, n_classes, n_classes)
                    config_id = f"{name}_{str(params)}"
                    cm_stats[config_id] = {
                        "mean": np.mean(cms_array, axis=0),
                        "std":  np.std(cms_array, axis=0),
                        "total": np.sum(cms_array, axis=0)
                    }

            df_results = pd.DataFrame(all_results)
            
            # Para manter compatibilidade com o teu retorno original (cm_total)
            # Escolhemos a cm_total do último modelo processado ou a soma de todos
            first_config = list(cm_stats.keys())[0]
            cm_total_sum = cm_stats[first_config]["total"] 

            return df_results, cm_stats, cm_total_sum

    # -------------------------------------------------------
    # TVT com feature selection usando ranking PRÉ-COMPUTADO
    # -------------------------------------------------------
    def run_tvt_with_feature_selection(self, val_size=0.2, test_size=0.2, 
                                       feature_ranking=None, train_subjects=None,
                                       val_subjects=None, test_subjects=None):
        """
        TVT com feature selection.
        
        Parameters:
        -----------
        val_size, test_size : float
            Proporções (usadas se train/val/test_subjects não forem fornecidos)
        feature_ranking : array, optional
            Ranking pré-computado de features (índices ordenados por importância).
            Se None, calcula ReliefF internamente.
        train_subjects, val_subjects, test_subjects : array, optional
            IDs dos participantes para cada split (modo manual).
            Se fornecidos, ignora val_size e test_size.
        """
        # 1. Split dos dados
        if train_subjects is not None and val_subjects is not None and test_subjects is not None:
            # Split manual
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt_manual(
                train_subjects, val_subjects, test_subjects
            )
        else:
            # Split automático
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt(
                val_size=val_size, test_size=test_size
            )
        
        # 2. Feature ranking
        if feature_ranking is None:
            print("⚙️  Calculando feature ranking com ReliefF...")
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

            # A. Feature Selection
            validation_f1s_model = []
            print("  → Testando número de features...")
            for k in range(1, num_features + 1):
                model = copy.deepcopy(model_original)
                selected_features = ranking[:k]
                
                model.fit(X_train[:, selected_features], y_train)
                y_val_pred = model.predict(X_val[:, selected_features])
                f1_val = compute_metrics(y_val, y_val_pred)["F1"]
                
                validation_f1s_model.append({
                    "model": name,
                    "F1": f1_val,
                    "k_features": k,
                    "selected_features": selected_features.tolist()
                })
            
            model_results["validation"] = validation_f1s_model
            
            # Melhor k
            f1_vals = [entry['F1'] for entry in validation_f1s_model]
            ks = [entry['k_features'] for entry in validation_f1s_model]
            max_index = np.argmax(f1_vals)
            best_k = ks[max_index]
            best_f1 = f1_vals[max_index]
            best_features_indices = ranking[:best_k]
            
            print(f"  ✓ Melhor k={best_k} (F1={best_f1:.4f})")
            
            # B. Parameter Optimization
            best_params = None
            best_f1_param = -1
            print("  → Otimizando parâmetros...")
            
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
            model.set_params(**best_params)
            
            print(f"  ✓ Melhores params: {best_params} (F1={best_f1_param:.4f})")

            # C. Final Evaluation
            X_trainval = np.vstack([X_train, X_val])
            y_trainval = np.hstack([y_train, y_val])
            
            model.fit(X_trainval[:, best_features_indices], y_trainval)
            y_test_pred = model.predict(X_test[:, best_features_indices])
            f1_test = compute_metrics(y_test, y_test_pred)["F1"]
            
            model_results["test"] = {
                "model": name,
                "F1": f1_test,
                "k_features": best_k,
                "params": best_params
            }
            
            print(f"  ✓ Teste final: F1={f1_test:.4f}\n")

        return models_results

    # -------------------------------------------------------
    # Cross-validation com feature selection e ranking PRÉ-COMPUTADO
    # -------------------------------------------------------
    def run_cross_with_validation_feature_selection(self, number_of_folds=10, 
                                                    n_repeats=1, test_size=0.3,
                                                    feature_ranking=None):
        """
        Cross-validation com feature selection.
        
        Parameters:
        -----------
        number_of_folds : int
            Número de folds
        n_repeats : int
            Número de repetições
        test_size : float
            Proporção de validação dentro de cada fold
        feature_ranking : array, optional
            Ranking pré-computado de features. Se None, calcula ReliefF em cada fold.
        """
        all_results = []
        
        for repeat in range(n_repeats):
            print(f"\n[Repetição {repeat+1}/{n_repeats}]")
            current_random_state = self.random_state + repeat
            
            # Cria folds dividindo PARTICIPANTES (se subject-aware)
            if self.subject_aware_mode:
                folds = split_subjects_kfold(
                    self.subjects, 
                    n_splits=number_of_folds, 
                    random_state=current_random_state
                )
            else:
                from data.data_splitting import split_data_kfold
                folds = split_data_kfold(self.X, n_splits=number_of_folds, random=current_random_state)
            
            for fold_id, fold_data in enumerate(folds):
                print(f"  > Fold {fold_id+1}/{number_of_folds}")
                
                if self.subject_aware_mode:
                    train_subjects, test_subjects = fold_data
                    train_mask = np.isin(self.subjects, train_subjects)
                    test_mask = np.isin(self.subjects, test_subjects)
                    
                    X_train, X_test = self.X[train_mask], self.X[test_mask]
                    y_train, y_test = self.y[train_mask], self.y[test_mask]
                else:
                    train_idx, test_idx = fold_data
                    X_train, X_test = self.X[train_idx], self.X[test_idx]
                    y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                # Split treino em TrS e V
                if self.subject_aware_mode:
                    # Divide participantes de treino
                    unique_train = np.unique(self.subjects[train_mask])
                    np.random.seed(current_random_state)
                    np.random.shuffle(unique_train)
                    
                    n_val = max(1, int(len(unique_train) * test_size))
                    val_subjects = unique_train[:n_val]
                    trs_subjects = unique_train[n_val:]
                    
                    trs_mask = np.isin(self.subjects[train_mask], trs_subjects)
                    val_mask = np.isin(self.subjects[train_mask], val_subjects)
                    
                    X_TrS = X_train[trs_mask]
                    y_TrS = y_train[trs_mask]
                    X_V = X_train[val_mask]
                    y_V = y_train[val_mask]
                else:
                    X_TrS, X_V, y_TrS, y_V = train_test_split(
                        X_train, y_train,
                        test_size=test_size,
                        stratify=y_train,
                        random_state=current_random_state
                    )
                
                # Feature ranking
                if feature_ranking is None:
                    relief = ReliefF(n_neighbors=100)
                    relief.fit(X_TrS, y_TrS)
                    ranking = np.argsort(-relief.feature_importances_)
                else:
                    ranking = feature_ranking
                
                num_features = X_train.shape[1]

                fold_results = {
                    "repetition": repeat,
                    "fold_id": fold_id,
                    "models": []
                }
                
                for name, model_original in self.models.items():
                    param_grid = self.model_parameters.get(name, {})
                    
                    chaves_de_interesse = list(param_grid.keys())
                    todos_os_defaults = model_original.get_params()
                    default_params_filtrados = {
                        key: todos_os_defaults.get(key)
                        for key in chaves_de_interesse
                    }
                    
                    scores_validacao_features = []
                    scores_validacao_parametros = []
                    
                    best_k_features = 1
                    best_f1_k = -1
                    
                    # Feature Selection
                    for k in range(1, num_features + 1):
                        model = copy.deepcopy(model_original)
                        model.set_params(**default_params_filtrados)
                        
                        selected_features = ranking[:k]
                        
                        model.fit(X_TrS[:, selected_features], y_TrS)
                        y_val_pred = model.predict(X_V[:, selected_features])
                        f1_val = compute_metrics(y_V, y_val_pred)["F1"]
                        
                        scores_validacao_features.append({
                            "k_features": k, "F1": f1_val, "params": default_params_filtrados
                        })
                        
                        if f1_val > best_f1_k:
                            best_f1_k = f1_val
                            best_k_features = k
                    
                    best_features_indices = ranking[:best_k_features]
                    
                    # Parameter Optimization
                    best_params = default_params_filtrados
                    best_f1_param = best_f1_k
                    
                    for params in self._generate_configs(param_grid):
                        model = copy.deepcopy(model_original)
                        model.set_params(**params)
                        
                        model.fit(X_TrS[:, best_features_indices], y_TrS)
                        y_val_pred = model.predict(X_V[:, best_features_indices])
                        f1_val = compute_metrics(y_V, y_val_pred)["F1"]
                        
                        scores_validacao_parametros.append({
                            "k_features": best_k_features, "F1": f1_val, "params": params
                        })

                        if f1_val > best_f1_param:
                            best_f1_param = f1_val
                            best_params = params
                    
                    # Final Evaluation
                    model = copy.deepcopy(model_original)
                    model.set_params(**best_params)
                    
                    model.fit(X_train[:, best_features_indices], y_train)
                    
                    y_pred = model.predict(X_test[:, best_features_indices])
                    metrics = compute_metrics(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred, labels=self.labels)

                    fold_results["models"].append({
                        "model_name": name,
                        "validation_features_scores": scores_validacao_features,
                        "validation_params_scores": scores_validacao_parametros,
                        "best_k_features": best_k_features,
                        "best_params": best_params,
                        "test_metrics": metrics,
                        "test_confusion_matrix": cm
                    })
                    
                all_results.append(fold_results)
                
        return all_results


    # Outros métodos mantidos sem alteração
    def run_train_test_validation(self, test_size=0.2, val_size=0.2):
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt(
                val_size=val_size, test_size=test_size
            )

        validation_history = []
        test_results = []
        cms = []

        for name, model in self.models.items():
            param_grid = self.model_parameters.get(name, {})
            
            best_f1_score = -1.0
            best_params = {}
            for params in self._generate_configs(param_grid):
                model.set_params(**params)
                model.fit(X_train, y_train)
                
                y_pred_val = model.predict(X_val)
                f1_val = compute_metrics(y_val, y_pred_val)["F1"]
                
                row_history = {
                    "Model": name,
                    "Params": str(params),
                    "val_F1": f1_val
                }
                validation_history.append(row_history)

                if f1_val > best_f1_score:
                    best_f1_score = f1_val
                    best_params = params

            model.set_params(**best_params)
            model.fit(X_train, y_train)
            
            y_pred_test = model.predict(X_test)
            f1_test = compute_metrics(y_test, y_pred_test)["F1"]
            
            cm = confusion_matrix(y_test, y_pred_test, labels=self.labels)
            cms.append(cm)

            row_final = {
                "Model": name,
                "Best_Params": str(best_params),
                "Val_Best_F1": best_f1_score,
                "Test_F1": f1_test
            }
            
            test_results.append(row_final)

        return pd.DataFrame(validation_history), pd.DataFrame(test_results), cms





"""
    def run_cross_validation_feature_selection(self, number_of_folds,test_size=0.3):
        results = []   # para guardar resultado final de cada fold
        folds = split_data_kfold(self.X, n_splits=number_of_folds, random=self.random_state)
        #For each testing fold (Te) 
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"\n=== Fold {fold_id+1}/{number_of_folds} ===")       
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # === INNER SPLIT: cria TrS e V ===
            X_TrS, X_V, y_TrS, y_V = train_test_split(
                X_train, y_train,
                test_size=test_size,         
                stratify=y_train,     
                random_state=self.random_state)
            # === 2.2. FEATURE RANKING ===
            relief = ReliefF(n_neighbors=100)
            relief.fit(X_TrS, y_TrS)
            ranking = np.argsort(-relief.feature_importances_)
            num_features = X_train.shape[1]


            ###for each model
            fold_results = {"fold": fold_id, "models": []}
            # === 2.3.1.  PARA CADA MODELO model selection, parameter optimisation and feature selection ===
            for name, model_original in self.models.items():
                print(f"Processing model: {name}")
                 # -------- 2.3.1.2 FEATURE SELECTION VIA VALIDATION --------
                validation_scores = []
                ####default parameters
                model_results = {"model": name, "validation": []}
                #train model
                for k in range(1, num_features + 1):
                    ###parametros default
                    model = copy.deepcopy(model_original)
                    param_grid = self.model_parameters.get(name, {})
                    model.set_params(**param_grid)


                    selected_features = ranking[:k]
  
                    model.fit(X_TrS[:, selected_features],y_TrS)
                    y_val_pred = model.predict(X_V[:, selected_features])
                    f1_val=compute_metrics(y_V, y_val_pred)["F1"]
                    validation_scores.append({
                                    "model": name,
                                    "F1": f1_val,
                                    "k_features": k,
                                    "selected_features": selected_features.tolist()
                                    })   
                    model_results["validation"]=validation_scores

                    #Get the feature set that maximises the score in V
                    # Extrair o melhor score F1 e número de features necessárias
                    f1_vals = [entry['F1'] for entry in validation_scores]
                    ks = [entry['k_features'] for entry in validation_scores]
                    max_index = np.argmax(f1_vals) #first index with best result
                    best_k = ks[max_index]       # feature set that maximizes F1

                    best_f1_param=-1
                    best_params=None
                    ###valuate the use of different parameter values (e.g., k = 1, 3, 5, 7, …)
                    for params in self._generate_configs(param_grid):
                        model=copy.deepcopy(model_original)
                        #Evaluate the use of different parameter values on validation set
                        model.set_params(**params)
                        ##treinar usando as melhores features de validação
                        model.fit(X_TrS[:, best_features_indices], y_TrS) ##TODO needs fixing
                        ##VALIDATION
                        Y_V_pred = model.predict(X_V)
                        f1 = compute_metrics(y_V, Y_V_pred)["F1"]
                        if f1 > best_f1_param:
                            best_f1_param = f1
                            best_params = params

                        model=copy.deepcopy(model_original)        
                        model.set_params(**best_params)
                        model.fit(X_train,y_train)
                        y_pred = model.predict(X_test)
                        f1 = compute_metrics(y_test, y_pred)["F1"]
                        model_results["test"]={
                                "model": name,
                                "F1": f1,
                                "k_features": best_k,
                                "params": best_params}
                        fold_results["model_results"].append(model_results) 
"""