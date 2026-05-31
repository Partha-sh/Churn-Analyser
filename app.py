from collections import UserList
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"
FRONTEND_PATH = BASE_DIR / "templates" / "index.html"


def _install_sklearn_pickle_compat() -> None:
    """Restore a removed private sklearn type needed by older pickles."""
    import sklearn.compose._column_transformer as column_transformer

    if hasattr(column_transformer, "_RemainderColsList"):
        return

    class _RemainderColsList(UserList):
        def __init__(
            self,
            columns,
            *,
            future_dtype=None,
            warning_was_emitted=False,
            warning_enabled=True,
        ):
            super().__init__(columns)
            self.future_dtype = future_dtype
            self.warning_was_emitted = warning_was_emitted
            self.warning_enabled = warning_enabled

        def __getitem__(self, index):
            return super().__getitem__(index)

        def _repr_pretty_(self, printer, *_):
            printer.text(repr(self.data))

    column_transformer._RemainderColsList = _RemainderColsList


def _load_model():
    _install_sklearn_pickle_compat()
    return joblib.load(MODEL_PATH)


model = _load_model()
app = FastAPI(title="Churn Analyzer")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = int(model.predict(input_df)[0])
    response = {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "prediction_code": prediction,
    }

    if hasattr(model, "predict_proba"):
        response["churn_probability"] = float(model.predict_proba(input_df)[0][1])

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
