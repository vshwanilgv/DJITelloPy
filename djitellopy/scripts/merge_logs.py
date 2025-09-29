import os
import pandas as pd
import pathlib

def main():
    # base_dir points to .../DJITelloPy/
    base_dir = pathlib.Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "dataset"

    # print(f"Looking inside: {dataset_dir}")

    # flights = [f for f in os.listdir(dataset_dir) if f.startswith("flight-")]

    # dfs = []
    # for flight in flights:
    #     flight_path = os.path.join(dataset_dir, flight)
    #     csv_file = None
    #     avi_file = None

    #     # find csv and avi inside the folder
    #     for f in os.listdir(flight_path):
    #         if f.endswith(".csv"):
    #             csv_file = os.path.join(flight_path, f)
    #         elif f.endswith(".avi"):
    #             avi_file = os.path.join(flight_path, f)

    #     if csv_file and avi_file:
    #         try:
    #             df = pd.read_csv(csv_file)
    #             df["video_path"] = avi_file
    #             df["flight_id"] = flight
    #             dfs.append(df)
    #         except Exception as e:
    #             print(f"Skipping {flight}, error: {e}")
    #     else:
    #         print(f"⚠️ Missing CSV or AVI in {flight}")

    # if dfs:
    #     merged = pd.concat(dfs, ignore_index=True)
    #     output_file = dataset_dir / "dataset.csv"
    #     merged.to_csv(output_file, index=False)
    #     print(f"Dataset created at {output_file}")
    # else:
    #     print("No valid flights found")

    merged_file = os.path.join(dataset_dir, "dataset.csv")

    all_dfs = []
    flights = [f for f in os.listdir(dataset_dir) if f.startswith("flight-")]

    for flight in flights:
        flight_dir = os.path.join(dataset_dir, flight)
        for file in os.listdir(flight_dir):
            if file.startswith("detections") and file.endswith(".csv"):
                df = pd.read_csv(os.path.join(flight_dir, file))
                df["flight_id"] = flight
                df["video_path"] = [
                    os.path.join(flight_dir, f) for f in os.listdir(flight_dir) if f.endswith(".avi")
                ][0]

                df["detection_id"] = [
                    f"{flight}_{i:05d}" for i in range(len(df))
                ]

                all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(merged_file, index=False)
    print(f"Merged dataset saved at {merged_file}")

if __name__ == "__main__":
    main()
