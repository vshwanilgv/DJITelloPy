# scripts/preprocess_data.py
import os, pandas as pd, numpy as np, joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent 
RAW_CSV = ROOT / "dataset" / "dataset.csv"
PROCESSED_CSV = ROOT / "dataset" / "processed_logs.csv"
DATA_DIR = ROOT / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FRAME_W = 960.0
FRAME_H = 720.0
H_SCALE = 200.0
SPEED_SCALE = 100.0

ACTION_KEEP = {"hover","move_left","move_right","move_forward"}  # allowed actions

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def process():
    df = pd.read_csv(RAW_CSV)
    print("Loaded rows:", len(df))

    df = df[df["confidence"].fillna(0) >= 0.3].copy()

    for c in ["x","y","w","h","height","speed_x","speed_y","speed_z","battery","confidence"]:
        df[c] = df[c].apply(lambda v: safe_float(v, -1))

    # detected flag
    df["detected"] = ((df["x"] >= 0) & (df["y"] >= 0)).astype(int)

    # bbox features
    df["bbox_area"] = df["w"] * df["h"]
    df["aspect_ratio"] = df["w"] / (df["h"] + 1e-6)

    # centroid movement per flight (dx,dy). compute per flight to avoid cross-flight diffs
    df.sort_values(["flight_id","timestamp"], inplace=True)
    df["dx"] = df.groupby("flight_id")["x"].diff().fillna(0)
    df["dy"] = df.groupby("flight_id")["y"].diff().fillna(0)

    # normalize numeric columns (store scaler)
    numeric_cols = ["detected","x","y","w","h","bbox_area","aspect_ratio",
                    "height","speed_x","speed_y","speed_z","dx","dy"]
    # Replace -1 (no-detection) with 0 for x,y,w,h so scaling doesn't blow up
    df.loc[df["detected"]==0, ["x","y","w","h","bbox_area","aspect_ratio"]] = 0.0

    scaler = StandardScaler()
    df_num = scaler.fit_transform(df[numeric_cols].fillna(0))
    df_scaled = pd.DataFrame(df_num, columns=[f"{c}_s" for c in numeric_cols])
    df = pd.concat([df.reset_index(drop=True), df_scaled], axis=1)

    # action encoding (keep only allowed actions)
    df["action"] = df["action"].fillna("hover").astype(str)
    df = df[df["action"].isin(ACTION_KEEP | {"manual"})].copy()  # filter unknowns
    # Map 'none' or 'none' like -> hover
    df.loc[df["action"]=="none", "action"] = "hover"

    le = LabelEncoder()
    df["action_encoded"] = le.fit_transform(df["action"])

    # Save processed csv
    df.to_csv(PROCESSED_CSV, index=False)
    print("Saved processed CSV:", PROCESSED_CSV)

    # create X and y for modeling
    feature_cols = [f"{c}_s" for c in numeric_cols]  # normalized features
    X = df[feature_cols].values.astype(np.float32)
    y = df["action_encoded"].values.astype(int)

    np.save(DATA_DIR / "X.npy", X)
    np.save(DATA_DIR / "y.npy", y)
    print("Saved X/y arrays:", DATA_DIR / "X.npy", DATA_DIR / "y.npy")

    # Save scaler & label encoder
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    print("Saved scaler+label encoder to models/")

    # Train/val/test split CSVs (by flight to reduce leakage)
    train, test = train_test_split(df["flight_id"].unique(), test_size=0.2, random_state=42)
    val_split = 0.2
    # create boolean masks per flight
    train_df = df[df["flight_id"].isin(train)]
    rest = df[~df["flight_id"].isin(train)]
    val_df, test_df = train_test_split(rest, test_size=0.5, random_state=42)

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    print("Saved splits in djitellopy/data/")

if __name__ == "__main__":
    process()
