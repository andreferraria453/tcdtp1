import copy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from data.data_splitting import split_data_kfold
from metrics.metrics import compute_accuracy, compute_f1_score, compute_precision, compute_recall
from itertools import product
from skrebate import ReliefF



def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": compute_accuracy(y_true, y_pred),
        "Recall": compute_recall(y_true, y_pred, average='weighted'),
        "Precision": compute_precision(y_true, y_pred, average='weighted'),
        "F1": compute_f1_score(y_true, y_pred, average='weighted')
    }


class Experiment:

    def __init__(self, X, y, models, model_parameters, labels=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.models = models
        self.model_parameters = model_parameters
        self.labels = labels if labels is not None else np.unique(y)
        self.random_state = 10

    # -------------------------------------------------------
    # Helper: generate combinations of hyperparameters
    # -------------------------------------------------------
    def _generate_configs(self, grid):
        keys = list(grid.keys())
        values = list(grid.values())
        for combo in product(*values):
            yield dict(zip(keys, combo))

    # -------------------------------------------------------
    # Train / Validation / Test Split
    # -------------------------------------------------------
    def split_tvt(self, val_size=0.2, test_size=0.2, random_state=0):
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
    # Train-Only
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
    # Train/Test
    # -------------------------------------------------------
    def run_train_test(self, test_size=0.3):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=10
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

 

    def run_train_test_validation(self, test_size=0.2, val_size=0.2):
        # 1. Split dos dados
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt(
                val_size=val_size, test_size=test_size
            )

        validation_history = []  # DF 1: Histórico de todos os params
        test_results = []        # DF 2: Resultado final dos melhores modelos
        cms = []                 # Matrizes de confusão finais

            # --- 2. Ciclo por Modelo ---
        for name, model in self.models.items():
            param_grid = self.model_parameters.get(name, {})
            
            best_f1_score = -1.0
            best_params = {}
            for params in self._generate_configs(param_grid):
                # Treinar
                model.set_params(**params)
                model.fit(X_train, y_train)
                
                # Avaliar na VALIDAÇÃO
                y_pred_val = model.predict(X_val)
                f1_val = compute_metrics(y_val, y_pred_val)["F1"]

                
                # A. Guardar no Histórico de Validação (Para gráficos)
                # Guardamos o nome, os params e o score
                row_history = {
                    "Model": name,
                    "Params": str(params), # String para leitura
                    "val_F1": f1_val
                }
                validation_history.append(row_history)

                # Verificar se é o melhor
                if f1_val > best_f1_score:
                    best_f1_score = f1_val
                    best_params = params

      
                model.set_params(**best_params)
                model.fit(X_train, y_train) # Treina com o melhor setup
                
                y_pred_test = model.predict(X_test)
                f1_test = compute_metrics(y_test, y_pred_test)["F1"]
                
                # Guardar Matriz de Confusão
                cm = confusion_matrix(y_test, y_pred_test, labels=self.labels)
                cms.append(cm)

                # B. Guardar Resultado Final do Teste
                row_final = {
                    "Model": name,
                    "Best_Params": str(best_params),
                    "Val_Best_F1": best_f1_score, # O valor que fez ele ser escolhido
                    "Test_F1": f1_test # O valor real no teste
                }
                # Adiciona as outras métricas do teste (Accuracy, Recall, etc.) se quiseres
                # row_final.update(metrics_test) 
                
                test_results.append(row_final)


        return pd.DataFrame(validation_history),pd.DataFrame(test_results),cms


    def run_tvt_with_feature_selection(self, val_size=0.2, test_size=0.2):

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_tvt(
            val_size=val_size, test_size=test_size)
        
        ####Validation of parameters using validation set####
        relief = ReliefF(n_neighbors=100,n_jobs=-1)
        relief.fit(X_train, y_train)
        ranking = np.argsort(-relief.feature_importances_)
        num_features = X_train.shape[1]
        print("relieff done")
        models_results=[]
        for name, model_original in self.models.items(): ## para cada modelo antes de testar a variação de hiperparametros
            print("Modelo: ", name)
            model_results={}
            param_grid = self.model_parameters.get(name, {}) #  #baseline model parameters

    
            ####Descobrir a melhor feature para o set de validação 
            validation_f1s_model=[]
            print("validation features")
            for k in range(1, num_features + 1):
                model=copy.deepcopy(model_original)
                selected_features = ranking[:k]
                #print(X_train[:, selected_features])
                model.fit(X_train[:, selected_features], y_train)
                y_val_pred = model.predict(X_val[:, selected_features])
                f1_val=compute_metrics(y_val, y_val_pred)["F1"]
                validation_f1s_model.append({
                        "model": name,
                        "F1": f1_val,
                        "k_features": k,
                        "selected_features": selected_features.tolist()
                })   
            model_results["validation"]=validation_f1s_model
            
            # Extrair o melhor score F1 e número de features necessárias
            f1_vals = [entry['F1'] for entry in validation_f1s_model]
            ks = [entry['k_features'] for entry in validation_f1s_model]
            max_index = np.argmax(f1_vals) #first index with best result
            best_k = ks[max_index]       # número de features que maximiza F1
            best_f1 = f1_vals[max_index] # valor do F1 máximo
            # Selecionar features com melhor score em VALIDATION
            best_features_indices = ranking[:best_k]
            ##Avaliar diferentes configurações de parâmetros
            best_params = None
            best_f1_param = -1
            print("testing models")
            for params in self._generate_configs(param_grid):
                model=copy.deepcopy(model_original)
                #Evaluate the use of different parameter values on validation set
                print(params)
                model.set_params(**params)
                ##treinar usando as melhores features de validação
                model.fit( X_train[:, best_features_indices], y_train)
                y_val_pred = model.predict(X_val[:, best_features_indices])
                f1 = compute_metrics(y_val, y_val_pred)["F1"]
                if f1 > best_f1_param:
                    best_f1_param = f1
                    best_params = params
            models_results.append(model_results)
            model.set_params(**best_params)

            ############train+validation############
            X_trainval = np.vstack([X_train, X_val])
            y_trainval = np.hstack([y_train, y_val])
            ##rain (using TrS+V, i.e., Tr) with the best feature set and parameter values 
            model.fit(X_trainval[:, best_features_indices], y_trainval)
            y_test_pred=model.predict(X_test[:, best_features_indices])
            f1 = compute_metrics(y_test, y_test_pred)["F1"]
            model_results["test"]={
                        "model": name,
                        "F1": f1,
                        "k_features": best_k,
                        "params": best_params}
            

        return models_results

    def run_cross_with_validation_feature_selection(self, number_of_folds=10, n_repeats=1, test_size=0.3):
        # 'all_results' guardará os resultados de TODAS as repetições e TODOS os folds
        all_results = []
        for repeat in range(n_repeats):
            print(f"\n[Repetição {repeat+1}/{n_repeats}]")
            # Se não mudarmos isto, ele repete exatamente a mesma divisão 10 vezes.
            current_random_state = self.random_state + repeat
            
            # 1. Criação dos Folds (Tr/Te) com a semente variável
            folds = split_data_kfold(self.X, n_splits=number_of_folds, random=current_random_state)
            
            # --- CICLO INTERNO: FOLDS (10 folds) ---
            for fold_id, (train_idx, test_idx) in enumerate(folds):
                print(f"  > Fold {fold_id+1}/{number_of_folds}")
                
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                X_TrS, X_V, y_TrS, y_V = train_test_split(
                    X_train, y_train,
                    test_size=test_size,
                    stratify=y_train,
                    random_state=current_random_state)
                
                # 2.2. Feature Ranking (usando TrS)
                relief = ReliefF(n_neighbors=100)
                relief.fit(X_TrS, y_TrS)
                ranking = np.argsort(-relief.feature_importances_)
                num_features = X_train.shape[1]

                fold_results = {
                    "repetition": repeat, 
                    "fold_id": fold_id, 
                    "models": []
                }
                
                # 2.3. Otimização e Seleção
                for name, model_original in self.models.items():
                    param_grid = self.model_parameters.get(name, {})
                    
                    # Preparação de Defaults (Calculado Apenas Uma Vez)
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
                    
                    # --- A. Feature Selection ---
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
                    
                    # --- B. Parameter Optimization ---
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
                            
                    # --- C. Final Evaluation on Test Fold (Te) ---
                    model = copy.deepcopy(model_original)
                    model.set_params(**best_params)
                    
                    # Treina no conjunto Tr Completo (X_train)
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
                        "test_confusion_matrix": cm})
                all_results.append(fold_results)            
        return all_results

    # -------------------------------------------------------
    # Cross-Validation
    # -------------------------------------------------------
    def run_cross_validation(self, n_splits=10):
        results = []
        cms = []
        for name, model in self.models.items():
            param_grid = self.model_parameters.get(name, {})

            for params in self._generate_configs(param_grid):
                scores = {"Accuracy": [], "Recall": [], "Precision": [], "F1": []}
                cms_local = []

                for repeat in range(10):  # 10 repetições
                    #needs to take into consideration stratification
                    folds = split_data_kfold(self.X, n_splits=n_splits, random=repeat)
                    for train_idx, test_idx in folds:
                        #training folds, testing folds
                        X_train, X_test = self.X[train_idx], self.X[test_idx]
                        y_train, y_test = self.y[train_idx], self.y[test_idx]
                        model.set_params(**params)
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)

                        metrics = compute_metrics(y_test, pred)
                        cm = confusion_matrix(y_test, pred, labels=self.labels)

                        # acumula métricas por fold
                        for m in scores:
                            scores[m].append(metrics[m])
                        cms_local.append(cm)

                # média das métricas sobre todas as repetições e folds
                row = {"Model": name, **params, **{m: np.mean(scores[m]) for m in scores}}
                results.append(row)

                # soma das matrizes de confusão locais
                cms.append(np.sum(cms_local, axis=0))

        cm_total = np.sum(cms, axis=0)
        return pd.DataFrame(results), cms, cm_total




def feature_selection_with_relief(X, y, models, model_parameters, val_size=0.3, test_size=0.3, show_plot=True):
    
    # ---------- 1. ReliefF ----------
    relief = ReliefF(n_neighbors=100)
    relief.fit(X, y)
    ranking = np.argsort(-relief.feature_importances_)
    
    # ---------- 2. Avaliação incremental ----------
    n_features = X.shape[1]
    f1_by_feature = []
    validation_f1s_by_feature = []
    
    for k in range(1, n_features + 1):
        selected_features = ranking[:k]
        exp = Experiment(X[:, selected_features], y, models, model_parameters)
        
        # use_val=True → calcula F1 no conjunto de validação
        experiments, cms, cm_total = exp.run_tvt(val_size=val_size, test_size=test_size, use_val=True)
        validation_f1s_by_feature.append(experiments["F1"].iloc[0])
        
    
    # ---------- 3. Gráfico Elbow ----------
    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        plt.plot(range(1, n_features + 1), validation_f1s_by_feature, marker='o', linestyle='-', color='blue')
        plt.title("Elbow: F1 vs Número de Features")
        plt.xlabel("Número de Features")
        plt.ylabel("F1 Score (validação)")
        plt.xticks(range(1, n_features + 1))
        plt.grid(True)
        plt.show()
    
    return ranking, validation_f1s_by_feature, f1_by_feature






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