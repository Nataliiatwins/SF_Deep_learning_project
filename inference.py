import joblib
import pandas as pd
import torch
import torch.nn as nn


cat_cols = [
    "sex",
    "fasting_blood_sugar",
    "resting_electrocardiographic_results",
    "exercise_induced_angina",
    "slope",
    "number_of_major_vessels",
    "thal",
]


class HeartNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


test = pd.read_csv("test.csv")
ids = test["ID"].copy()

X_test = test.drop(columns=["ID"])
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)

feature_columns = joblib.load("feature_columns.pkl")
X_test = X_test.reindex(columns=feature_columns, fill_value=0)

scaler = joblib.load("scaler.pkl")
X_test_scaled = scaler.transform(X_test)

model = HeartNet(X_test.shape[1])
model.load_state_dict(torch.load("heart_net.pth", map_location="cpu"))
model.eval()

X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

with torch.no_grad():
    logits = model(X_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).int().numpy().ravel()

submission = pd.DataFrame({
    "ID": ids,
    "class": preds,
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved")
