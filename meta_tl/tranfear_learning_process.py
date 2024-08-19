import sys
from utils import *
import xgboost as xgb

def transfear_learning_process_main(selected_linkage_tasks_from_communities,active_learning_folder,path_to_record_linkage_tasks,relevant_columns):
    for selected_linkage_task, coressponding_linkage_tasks in selected_linkage_tasks_from_communities.items():
        print(f"Train the model on the selected labeled task which is: {selected_linkage_task}")
        selected_linkage_task_df_processed = pd.read_csv(active_learning_folder + selected_linkage_task)
        #selected_linkage_task_df_processed = prepare_dataframe_prediction(selected_linkage_task_df,relevant_columns)

        X = selected_linkage_task_df_processed.iloc[:, 2:-2]
        y = selected_linkage_task_df_processed.iloc[: , -1]

        model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
        model.fit(X, y)

        print(f"Inference on the coresponding tasks!")
        for coressponding_linkage_task in coressponding_linkage_tasks:
            coressponding_linkage_task_df = pd.read_csv(path_to_record_linkage_tasks + coressponding_linkage_task)
            coressponding_linkage_task_df_processed = prepare_dataframe_prediction(coressponding_linkage_task_df,relevant_columns)

            X = coressponding_linkage_task_df_processed.iloc[:, 2:-1] # Features (all columns except the last one)
            y = coressponding_linkage_task_df_processed.iloc[: , -1] # Taregt variable (is_match)

            # Prediction
            predictions = model.predict(X)
            class_probs = model.predict_proba(X)
            coressponding_linkage_task_df_processed['pred'] = predictions
            coressponding_linkage_task_df_processed[['probabilties_0', 'probabilties_1']] = class_probs
            coressponding_linkage_task_df_processed.to_csv(os.path.join(active_learning_folder,coressponding_linkage_task))





