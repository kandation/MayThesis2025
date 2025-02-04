import pandas as pd
import numpy as np
from pathlib import Path

"""
2025-02-01 Clean and Format Data for 37t, 38t, 39t, 40t
"""


def prepare_data(file_path):
    # try:
    df = pd.read_excel(file_path, header=0, skiprows=0, engine="openpyxl")
    df.rename(columns={"ปี/เดือน/วัน": "Date", "ชั่วโมง": "Hour"}, inplace=True)

    df = df.iloc[1:]

    df["Date"] = df["Date"].astype(int).astype(str)
    df["Hour"] = df["Hour"].astype(int).astype(str)

    def convert_datetime(row):
        date_str = str(row["Date"]).zfill(6)  # Pad date to 6 digits
        hour_str = str(row["Hour"]).zfill(4)  # Pad hour to 4 digits

        year_prefix = "20"  # Assuming years are in 2000s
        if date_str.startswith("9"):
            year_prefix = "19"

        year = year_prefix + date_str[:2]
        month = date_str[2:4]
        day = date_str[4:6]

        hour = hour_str[:2]
        minute = hour_str[2:]

        datetime_str = f"{year}-{month}-{day} {hour}:{minute}:00"
        # print(datetime_str, date_str, hour_str)

        try:
            return pd.to_datetime(datetime_str)
        except ValueError:
            return pd.NaT  # Handle potential parsing errors

    df["Datetime"] = df.apply(convert_datetime, axis=1)
    df.set_index("Datetime", inplace=True)

    df = df[~df.index.duplicated(keep="first")]

    start_date, end_date = df.index.min(), df.index.max()
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")

    date_range = pd.date_range(start=start_date, end=end_date, freq="h")
    print("Duplicate timestamps:", df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")]

    df = df.reindex(date_range)
    # %%
    df.columns
    df.rename(columns={"index": "Datetime"}, inplace=True)  # Rename index column

    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")
    df = df.apply(
        lambda col: pd.to_numeric(col, errors="coerce") if col.dtypes == "O" else col
    )

    # remove not data columns
    remv_cols = ["Date", "Hour"]
    df = df.drop(columns=remv_cols, errors="ignore")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Datetime"}, inplace=True)

    # except Exception as e:
    #     print(f"Error processing {file_path}: {e}")

    return df


uris = [
    "datasets/(37t)ศาลหลักเมือง(ปิดสถานี).xlsx",
    "datasets/(37t)สถานีอุตุนิยมวิทยาลำปาง.xlsx",
    "datasets/(38t)โรงพยาบาลส่งเสริมสุขภาพตำลบ้านสบป้าด.xlsx",
    "datasets/(39t)โรงพยาบาลส่งเสริมสุขภาพตำบลท่าสี.xlsx",
    "datasets/(40t)การประปาส่วนภูมิภาคแม่เมาะ.xlsx",
]
current_dir_path = Path(__file__).parent.parent.parent
updated_uris = [current_dir_path / uri for uri in uris]

dfs = []
for uri in updated_uris:
    dfs.append(prepare_data(uri))
    break


output_dir = current_dir_path / "cleanned_datasets"
output_dir.mkdir(exist_ok=True)

for uri, df in zip(updated_uris, dfs):
    filename = Path(uri).stem  # Extract filename without extension
    print(f"\nDataset: {filename}")
    print(df.head())
    df.to_csv(output_dir / f"{filename}.csv", index=False)
