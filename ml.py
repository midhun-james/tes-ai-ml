
import xgboost as xgb
from xgboost import plot_importance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix, classification_report,accuracy_score

df=pd.read_csv("kpi_data.csv", parse_dates=["timestamp"])
neighbours= {
    "SITE1": ["SITE2", "SITE3", "SITE4"],
    "SITE2": ["SITE1", "SITE3"],
    "SITE3": ["SITE1", "SITE2"],
    "SITE4": ["SITE1", "SITE5"],
    "SITE5": ["SITE4", "SITE6"],
    "SITE6": ["SITE5", "SITE7"],
    "SITE7": ["SITE6", "SITE8"],
    "SITE8": ["SITE7", "SITE9"],
    "SITE9": ["SITE8", "SITE10"],
    "SITE10": ["SITE9"],
}
PRB_THRESHOLD = 40   # % PRB utilization threshold for neighbors
RRC_THRESHOLD = 1200
DL_TP_THRESHOLD = 30
# DROP_THRESHOLD = 0.1

# Function to check neighbor PRB utilization
def check_neighbour_prb(site_id, timestamp, df):
    neighs = neighbours.get(site_id, [])
    # Get subset of neighbours for the same timestamp
    neigh_data = df[(df["site_id"].isin(neighs)) & (df["timestamp"] == timestamp)]
    if neigh_data.empty:
        return False
    return all(neigh_data["prb_util"] < PRB_THRESHOLD)

# Apply logic row by row
def shutdown_logic(row, df):
    self_condition = (
        (row["rrc_conn"] < RRC_THRESHOLD) &
        (row["dl_throughput"] < DL_TP_THRESHOLD)
    )
    neigh_condition = check_neighbour_prb(row["site_id"], row["timestamp"], df)
    return int(self_condition and neigh_condition)

df["shut_down"] = df.apply(lambda row: shutdown_logic(row, df), axis=1)
df.to_csv("kpi_data_processed.csv", index=False)  # Save processed data

# df.to_csv("kpi_data_processed.csv", index=False)
features = ["rrc_conn", "prb_util", "ul_throughput", "dl_throughput","drop_rate", "handover_success_rate"]
X = df[features]
y = df["shut_down"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42,shuffle=True)
model= xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    base_score=0.5,
    eval_metric='logloss',

)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
plot_importance(model)
plt.show()

