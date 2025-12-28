import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import product
from skrebate import ReliefF


from metrics.metrics import compute_accuracy, compute_f1_score, compute_precision, compute_recall, compute_confusion_matrix
from data.data_splitting import split_data_kfold

# Função auxiliar de métricas (caso não esteja importada)
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": compute_accuracy(y_true, y_pred),
        "Recall": compute_recall(y_true, y_pred, average='macro'),
        "Precision": compute_precision(y_true, y_pred, average='macro'),
        "F1": compute_f1_score(y_true, y_pred, average='macro')
    }

# Função auxiliar de split por sujeitos (caso não esteja importada)
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

# ==============================================================================
# CLASSE EXPERIMENT (COMPLETA E ATUALIZADA)
# ==============================================================================

class Experiment:
    def __init__(self, X, y, models, model_parameters, labels=None, subjects=None, scaler=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.models = models
        self.model_parameters = model_parameters
        self.labels = labels if labels is not None else np.unique(y)
        self.random_state = 10
        self.subjects = np.array(subjects) if subjects is not None else None
        self.subject_aware_mode = (self.subjects is not None)
        self.scaler = scaler  # Scaler (ex: StandardScaler())

        if self.subject_aware_mode:
            unique_subjects = np.unique(self.subjects)
            print(f"{len(unique_subjects)} participantes detectados.")
        
        if self.scaler:
            print(f"Scaler detetado: {type(self.scaler).__name__}")

    # -------------------------------------------------------
    # HELPERS DE SUPORTE
    # -------------------------------------------------------
    def _generate_configs(self, grid):
        keys = list(grid.keys())
        values = list(grid.values())
        for combo in product(*values):
            yield dict(zip(keys, combo))

    def _scale_data(self, X_train, *X_others):
        """Aplica fit no treino e transform nos outros para evitar data leakage."""
        if self.scaler is None:
            return (X_train, *X_others)
        
        sc = copy.deepcopy(self.scaler)
        X_train_scaled = sc.fit_transform(X_train)
        X_others_scaled = [sc.transform(x) for x in X_others]
        return (X_train_scaled, *X_others_scaled)

    def _plot_history(self, history, title):
        """Gera gráficos de Accuracy e Loss se houver histórico."""
        if not history: return
        epochs_range = range(1, len(history['train_acc']) + 1)
        
        plt.figure(figsize=(12, 4))
        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_acc'], label='Train Acc')
        plt.plot(epochs_range, history['val_acc'], label='Val Acc', linestyle="--")
        plt.title(f'Accuracy: {title}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Loss Plot
        if history['loss']:
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, history['loss'], label='Loss', color='red')
            plt.title('Loss')
            plt.xlabel('Epochs')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------
    # CORE: TREINO HÍBRIDO (FIT ou PARTIAL_FIT)
    # -------------------------------------------------------
    def _train_eval_model(self, model, X_train, y_train, X_val, y_val, epochs=None):
        if epochs is None or not hasattr(model, "partial_fit"):
            model.fit(X_train, y_train)
            if X_val is not None and y_val is not None:
                pred_val = model.predict(X_val)
                metrics = compute_metrics(y_val, pred_val)
                return metrics["F1"], model, None 
            return 0.0, model, None

        all_classes = self.labels
        best_val_f1 = -1
        best_model_state = None
        history = {'train_acc': [], 'val_acc': [], 'loss': []}
        
        for _ in range(epochs):
            model.partial_fit(X_train, y_train, classes=all_classes)
            
            pred_val = model.predict(X_val)
            val_f1 = compute_metrics(y_val, pred_val)["F1"]
            
            # Guardar histórico
            history['train_acc'].append(model.score(X_train, y_train))
            history['val_acc'].append(model.score(X_val, y_val))
            if hasattr(model, 'loss_'): history['loss'].append(model.loss_)

            # Checkpoint (Guardar melhor modelo)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(model)

        return best_val_f1, best_model_state, history

    # -------------------------------------------------------
    # SPLITS 
    # -------------------------------------------------------
    def split_tvt_manual(self, train_subjects, val_subjects, test_subjects):
        if not self.subject_aware_mode:
            raise ValueError("Requer subjects!")
        
        train_mask = np.isin(self.subjects, train_subjects)
        val_mask = np.isin(self.subjects, val_subjects)
        test_mask = np.isin(self.subjects, test_subjects)
        
        X_train, y_train = self.X[train_mask], self.y[train_mask]
        X_val, y_val = self.X[val_mask], self.y[val_mask]
        X_test, y_test = self.X[test_mask], self.y[test_mask]
        return X_train, X_val, X_test, y_train, y_val, y_test

    def split_tvt(self, val_size=0.2, test_size=0.2, random_state=0):
        if self.subject_aware_mode:
            unique = np.unique(self.subjects)
            np.random.seed(random_state)
            np.random.shuffle(unique)
            n_test = max(1, int(len(unique) * test_size))
            n_val = max(1, int(len(unique) * val_size))
            test_sub = unique[:n_test]
            val_sub = unique[n_test:n_test+n_val]
            train_sub = unique[n_test+n_val:]
            return self.split_tvt_manual(train_sub, val_sub, test_sub)
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y, random_state=random_state)
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state)
            return X_train, X_val, X_test, y_train, y_val, y_test

    def run_train_only(self):
        results = []
        cms = []
        if self.scaler:
            sc = copy.deepcopy(self.scaler)
            X_proc = sc.fit_transform(self.X)
        else:
            X_proc = self.X

        for name, model in self.models.items():
            param_grid = self.model_parameters.get(name, {})
            for params in self._generate_configs(param_grid):
                model.set_params(**params)
                model.fit(X_proc, self.y)
                pred = model.predict(X_proc)
                metrics = compute_metrics(self.y, pred)
                cm = compute_confusion_matrix(self.y, pred, labels=self.labels)
                results.append({"Model": name, **params, **metrics})
                cms.append(cm)
        return pd.DataFrame(results), cms, np.sum(cms, axis=0)


    def run_train_test(self, test_size=0.3, val_size=0.2, random_state=None, epochs=None):

        if random_state is None: random_state = self.random_state
        
        # 1. Split (Train / Val / Test)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt(
            val_size=val_size, test_size=test_size, random_state=random_state
        )

        # 2. Scale
        X_train, X_val, X_test = self._scale_data(X_train, X_val, X_test)

        results = []
        cms = []

        for name, model_original in self.models.items():
            param_grid = self.model_parameters.get(name, {})
            for params in self._generate_configs(param_grid):
                model = copy.deepcopy(model_original)
                model.set_params(**params)
                
                print(f"Treinando {name} {params} (Epochs={epochs})...")

                # CHAMA O HELPER INTELIGENTE
                _, best_model, history = self._train_eval_model(
                    model, X_train, y_train, X_val, y_val, epochs=epochs
                )

                # Avaliação final no Teste
                pred = best_model.predict(X_test)
                metrics = compute_metrics(y_test, pred)
                cm = compute_confusion_matrix(y_test, pred, labels=self.labels)
                results.append({"Model": name, **params, **metrics})
                cms.append(cm)

                # Plotar histórico se existir (só para este modelo/config)
                if history:
                    self._plot_history(history, f"{name} {params}")

        return pd.DataFrame(results), cms, np.sum(cms, axis=0)


    def run_tvt_with_feature_selection(self, val_size=0.2, test_size=0.2, 
                                       feature_ranking=None, train_subjects=None,
                                       val_subjects=None, test_subjects=None,
                                       epochs=None): 
        
        if train_subjects is not None:
             X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt_manual(train_subjects, val_subjects, test_subjects)
        else:
             X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt(val_size=val_size, test_size=test_size)
        
        X_train, X_val, X_test = self._scale_data(X_train, X_val, X_test)

        if feature_ranking is None:
            print("Computing ReliefF...")
            relief = ReliefF(n_neighbors=100, n_jobs=-1)
            relief.fit(X_train, y_train)
            ranking = np.argsort(-relief.feature_importances_)
        else:
            ranking = feature_ranking
        
        num_features = X_train.shape[1]
        models_results = []
        
        for name, model_original in self.models.items():
            model_results = {}
            param_grid = self.model_parameters.get(name, {})
            validation_f1s_model = []
            for k in range(1, num_features + 1):
                model = copy.deepcopy(model_original)
                sel_feat = ranking[:k]
                f1_val, _, _ = self._train_eval_model(
                    model, X_train[:, sel_feat], y_train, 
                    X_val[:, sel_feat], y_val, epochs=epochs
                )
                validation_f1s_model.append({
                    "model": name, "F1": f1_val, "k_features": k, 
                    "selected_features": sel_feat.tolist()
                })
            
            model_results["validation"] = validation_f1s_model
            
            f1_vals = [entry['F1'] for entry in validation_f1s_model]
            best_k = validation_f1s_model[np.argmax(f1_vals)]['k_features']
            best_features_indices = ranking[:best_k]
            best_params = None
            best_f1_param = -1
            best_model_trained = None 
            best_history = None
            
            for params in self._generate_configs(param_grid):
                model = copy.deepcopy(model_original)
                model.set_params(**params)
                
                f1, trained_model, history = self._train_eval_model(
                    model, X_train[:, best_features_indices], y_train, 
                    X_val[:, best_features_indices], y_val, epochs=epochs
                )
                
                if f1 > best_f1_param:
                    best_f1_param = f1
                    best_params = params
                    best_model_trained = trained_model
                    best_history = history
            
            models_results.append(model_results)
            
            y_test_pred = best_model_trained.predict(X_test[:, best_features_indices])
            metrics_test = compute_metrics(y_test, y_test_pred)
            
            model_results["test"] = {
                "model": name, "F1": metrics_test["F1"], 
                "k_features": best_k, "params": best_params,
                "metrics": metrics_test
            }
            if best_history:
                self._plot_history(best_history, f"{name} (Best: k={best_k})")
        return models_results

 
    def run_cross_validation(self, n_splits=10, n_repeats=10):
        all_results = []
        cm_stats = {}
        for name, model in self.models.items():
            param_grid = self.model_parameters.get(name, {})
            for params in self._generate_configs(param_grid):
                fold_metrics = {"Accuracy": [], "Recall": [], "Precision": [], "F1": []}
                fold_cms = []
                for repeat in range(n_repeats):
                    if self.subject_aware_mode:
                        folds = split_subjects_kfold(self.subjects, n_splits=n_splits, random_state=self.random_state + repeat)
                    else:
                        folds = split_data_kfold(self.X, n_splits=n_splits, random=repeat)

                    for fold_idx, (train_idx, test_idx) in enumerate(folds):
                        if self.subject_aware_mode:
                            train_mask = np.isin(self.subjects, train_idx)
                            test_mask = np.isin(self.subjects, test_idx)
                            X_train, X_test = self.X[train_mask], self.X[test_mask]
                            y_train, y_test = self.y[train_mask], self.y[test_mask]
                        else:
                            X_train, X_test = self.X[train_idx], self.X[test_idx]
                            y_train, y_test = self.y[train_idx], self.y[test_idx]
                        
                        X_train, X_test = self._scale_data(X_train, X_test)
                        model.set_params(**params)
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)
                        m = compute_metrics(y_test, pred)
                        cm = compute_confusion_matrix(y_test, pred, labels=self.labels)
                        for k in fold_metrics: fold_metrics[k].append(m[k])
                        fold_cms.append(cm)

                config_result = {
                    "Model": name, **params,
                    "Acc_Mean": np.mean(fold_metrics["Accuracy"]), "Acc_Std": np.std(fold_metrics["Accuracy"]),
                    "F1_Mean": np.mean(fold_metrics["F1"]), "F1_Std": np.std(fold_metrics["F1"])
                }
                all_results.append(config_result)
                cms_array = np.array(fold_cms)
                cm_stats[f"{name}_{str(params)}"] = {
                    "mean": np.mean(cms_array, axis=0),
                    "total": np.sum(cms_array, axis=0)
                }
        return pd.DataFrame(all_results), cm_stats

    def run_cross_with_validation_feature_selection(self, number_of_folds=10, n_repeats=1, test_size=0.3, feature_ranking=None):
        all_results = []
        for repeat in range(n_repeats):
            cur_rnd = self.random_state + repeat
            if self.subject_aware_mode:
                folds = split_subjects_kfold(self.subjects, n_splits=number_of_folds, random_state=cur_rnd)
            else:
                folds = split_data_kfold(self.X, n_splits=number_of_folds, random=cur_rnd)
            
            for fold_id, fold_data in enumerate(folds):
                
                # Setup Outer Split
                if self.subject_aware_mode:
                    train_subj, test_subj = fold_data
                    train_mask = np.isin(self.subjects, train_subj)
                    test_mask = np.isin(self.subjects, test_subj)
                    X_train, X_test = self.X[train_mask], self.X[test_mask]
                    y_train, y_test = self.y[train_mask], self.y[test_mask]
                    
                    # Inner Split (TrS vs Val)
                    uniq_tr = np.unique(self.subjects[train_mask])
                    np.random.shuffle(uniq_tr)
                    n_val = max(1, int(len(uniq_tr) * test_size))
                    val_subj_in = uniq_tr[:n_val]
                    trs_subj_in = uniq_tr[n_val:]
                    trs_mask = np.isin(self.subjects[train_mask], trs_subj_in)
                    val_mask = np.isin(self.subjects[train_mask], val_subj_in)
                    X_TrS, y_TrS = X_train[trs_mask], y_train[trs_mask]
                    X_V, y_V = X_train[val_mask], y_train[val_mask]
                else:
                    train_idx, test_idx = fold_data
                    X_train, X_test = self.X[train_idx], self.X[test_idx]
                    y_train, y_test = self.y[train_idx], self.y[test_idx]
                    X_TrS, X_V, y_TrS, y_V = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train, random_state=cur_rnd)

                # Scaling
                X_TrS_sc, X_V_sc = self._scale_data(X_TrS, X_V)
                X_train_sc, X_test_sc = self._scale_data(X_train, X_test)

                # Feature Selection
                if feature_ranking is None:
                    relief = ReliefF(n_neighbors=100)
                    relief.fit(X_TrS_sc, y_TrS)
                    ranking = np.argsort(-relief.feature_importances_)
                else:
                    ranking = feature_ranking

                num_feat = X_train.shape[1]
                fold_res = {"repetition": repeat, "fold_id": fold_id, "models": []}

                for name, model_orig in self.models.items():
                    param_grid = self.model_parameters.get(name, {})
                    defaults = model_orig.get_params()
                    def_params = {k: defaults.get(k) for k in param_grid.keys()}
                    
                    best_k = 1
                    best_f1_k = -1
                    
                    # Find Best K (Standard fit)
                    for k in range(1, num_feat + 1):
                        model = copy.deepcopy(model_orig)
                        model.set_params(**def_params)
                        sel = ranking[:k]
                        model.fit(X_TrS_sc[:, sel], y_TrS)
                        y_pv = model.predict(X_V_sc[:, sel])
                        f1 = compute_metrics(y_V, y_pv)["F1"]
                        if f1 > best_f1_k:
                            best_f1_k = f1
                            best_k = k
                    
                    best_feats = ranking[:best_k]
                    best_params = def_params
                    best_f1_p = best_f1_k
                    
                    # Find Best Params
                    for p in self._generate_configs(param_grid):
                        model = copy.deepcopy(model_orig)
                        model.set_params(**p)
                        model.fit(X_TrS_sc[:, best_feats], y_TrS)
                        y_pv = model.predict(X_V_sc[:, best_feats])
                        f1 = compute_metrics(y_V, y_pv)["F1"]
                        if f1 > best_f1_p:
                            best_f1_p = f1
                            best_params = p
                    
                    # Final Test
                    final_mod = copy.deepcopy(model_orig)
                    final_mod.set_params(**best_params)
                    final_mod.fit(X_train_sc[:, best_feats], y_train)
                    y_pt = final_mod.predict(X_test_sc[:, best_feats])
                    
                    met = compute_metrics(y_test, y_pt)
                    cm = compute_confusion_matrix(y_test, y_pt, labels=self.labels)
                    fold_res["models"].append({
                        "model_name": name, "best_k": best_k, "best_params": best_params,
                        "test_metrics": met, "cm": cm
                    })
                all_results.append(fold_res)
        return all_results
    

    def run_train_test_validation(self, val_size=0.2, test_size=0.2, epochs=None,
                                  train_subjects=None, val_subjects=None, test_subjects=None):
        """
        Executa uma Grid Search completa (TVT).
        1. Treina TODAS as combinações de parâmetros no Treino.
        2. Avalia TODAS na Validação (escolhe a melhor).
        3. Avalia APENAS A MELHOR no Teste.
        """
        
        if train_subjects is not None:
            # Opção A: Split Manual
            print(f"Manual Subject Split: Treino={len(train_subjects)}, Val={len(val_subjects)}, Teste={len(test_subjects)}")
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt_manual(
                train_subjects, val_subjects, test_subjects
            )
        else:
            # Opção B: Split Automático
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt(
                val_size=val_size, test_size=test_size
            )

        X_train, X_val, X_test = self._scale_data(X_train, X_val, X_test)

        validation_history = []  # Guarda TODAS as tentativas
        test_results = []        # Guarda apenas o VENCEDOR de cada modelo
        cms = []                 # Matriz de confusão do vencedor

        # --- 3. Ciclo por Modelo (ex: MLP, SVM...) ---
        for name, model_orig in self.models.items():
            print(f"🔬 A analisar {name}...")
            param_grid = self.model_parameters.get(name, {})
            
            # Variáveis para encontrar o campeão deste modelo
            best_val_f1 = -1.0
            best_params = {}
            best_model_trained = None 
            best_history = None       
            
            # --- 4. Ciclo por Combinação de Parâmetros (Grid Search) ---
            # Aqui é onde ele testa hidden_layer_sizes=(10,), depois (30,), etc.
            for params in self._generate_configs(param_grid):
                model = copy.deepcopy(model_orig)
                model.set_params(**params)
                
                # Helper treina e avalia na validação
                f1_val, trained_model, history = self._train_eval_model(
                    model, X_train, y_train, X_val, y_val, epochs=epochs
                )
                
                # A. Guardar no Histórico de Validação (Para tu veres que ele testou tudo)
                row_history = {
                    "Model": name,
                    "Params": str(params),
                    "val_F1": f1_val
                }
                validation_history.append(row_history)

                # B. Verificar se é o novo recordista
                if f1_val > best_val_f1:
                    best_val_f1 = f1_val
                    best_params = params
                    best_model_trained = trained_model 
                    best_history = history

            # --- 5. Teste Final (Apenas com o CAMPEÃO) ---
            if best_model_trained is not None:
                # Usamos o modelo campeão para prever no Test Set (dados nunca vistos)
                y_pred_test = best_model_trained.predict(X_test)
                
                metrics_test = compute_metrics(y_test, y_pred_test)
                cm = compute_confusion_matrix(y_test, y_pred_test, labels=self.labels)
                cms.append(cm)
                row_final = {
                    "Model": name,
                    "Best_Params": str(best_params),
                    "Val_Best_F1": best_val_f1,     
                    "Test_F1": metrics_test["F1"],  
                    "Test_Acc": metrics_test["Accuracy"],
                    "Test_Metrics": metrics_test
                }
                test_results.append(row_final)
                if best_history:
                    self._plot_history(best_history, f"{name} (Best Config)")

        return pd.DataFrame(validation_history), pd.DataFrame(test_results), cms





"""

    def run_cross_validation(self, n_splits=10, n_repeats=10):
        all_results = []
        cm_stats = {}
        for name, model in self.models.items():
            param_grid = self.model_parameters.get(name, {})
            for params in self._generate_configs(param_grid):
                fold_metrics = {"Accuracy": [], "Recall": [], "Precision": [], "F1": []}
                fold_cms = []
                for repeat in range(n_repeats):
                    if self.subject_aware_mode:
                        folds = split_subjects_kfold(self.subjects, n_splits=n_splits, random_state=self.random_state + repeat)
                    else:
                        folds = split_data_kfold(self.X, n_splits=n_splits, random=repeat)

                    for fold_idx, (train_idx, test_idx) in enumerate(folds):
                        if self.subject_aware_mode:
                            train_mask = np.isin(self.subjects, train_idx)
                            test_mask = np.isin(self.subjects, test_idx)
                            X_train, X_test = self.X[train_mask], self.X[test_mask]
                            y_train, y_test = self.y[train_mask], self.y[test_mask]
                        else:
                            X_train, X_test = self.X[train_idx], self.X[test_idx]
                            y_train, y_test = self.y[train_idx], self.y[test_idx]
                        
                        X_train, X_test = self._scale_data(X_train, X_test)
                        model.set_params(**params)
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)
                        m = compute_metrics(y_test, pred)
                        cm = confusion_matrix(y_test, pred, labels=self.labels)
                        for k in fold_metrics: fold_metrics[k].append(m[k])
                        fold_cms.append(cm)

                config_result = {
                    "Model": name, **params,
                    "Acc_Mean": np.mean(fold_metrics["Accuracy"]), "Acc_Std": np.std(fold_metrics["Accuracy"]),
                    "F1_Mean": np.mean(fold_metrics["F1"]), "F1_Std": np.std(fold_metrics["F1"])
                }
                all_results.append(config_result)
                cms_array = np.array(fold_cms)
                cm_stats[f"{name}_{str(params)}"] = {
                    "mean": np.mean(cms_array, axis=0),
                    "total": np.sum(cms_array, axis=0)
                }
        return pd.DataFrame(all_results), cm_stats


"""