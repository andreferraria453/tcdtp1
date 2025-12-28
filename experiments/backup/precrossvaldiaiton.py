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
                        # Split
                        if self.subject_aware_mode:
                            train_mask = np.isin(self.subjects, train_idx)
                            test_mask = np.isin(self.subjects, test_idx)
                            X_train, X_test = self.X[train_mask], self.X[test_mask]
                            y_train, y_test = self.y[train_mask], self.y[test_mask]
                        else:
                            X_train, X_test = self.X[train_idx], self.X[test_idx]
                            y_train, y_test = self.y[train_idx], self.y[test_idx]
                        
                        # --- APLICAR SCALER NESTE FOLD ---
                        X_train, X_test = self._scale_data(X_train, X_test)

                        # Treino/Predição
                        model.set_params(**params)
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)

                        m = compute_metrics(y_test, pred)
                        cm = confusion_matrix(y_test, pred, labels=self.labels)

                        for k in fold_metrics: fold_metrics[k].append(m[k])
                        fold_cms.append(cm)

                # Agregar resultados
                config_result = {
                    "Model": name, **params,
                    "Acc_Mean": np.mean(fold_metrics["Accuracy"]), "Acc_Std": np.std(fold_metrics["Accuracy"]),
                    "Prec_Mean": np.mean(fold_metrics["Precision"]), "Prec_Std": np.std(fold_metrics["Precision"]),
                    "Rec_Mean": np.mean(fold_metrics["Recall"]), "Rec_Std": np.std(fold_metrics["Recall"]),
                    "F1_Mean": np.mean(fold_metrics["F1"]), "F1_Std": np.std(fold_metrics["F1"])
                }
                all_results.append(config_result)
                cms_array = np.array(fold_cms)
                config_id = f"{name}_{str(params)}"
                cm_stats[config_id] = {
                    "mean": np.mean(cms_array, axis=0), "std": np.std(cms_array, axis=0), "total": np.sum(cms_array, axis=0)
                }

        df_results = pd.DataFrame(all_results)
        first_config = list(cm_stats.keys())[0] if cm_stats else None
        
        cm_total_sum = cm_stats[first_config]["total"] if first_config else None
        return df_results, cm_stats, cm_total_sum
