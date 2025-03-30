import pandas as pd
import random
from pathlib import Path
import re


# Helper function to normalize text
# def normalize_text(text):
#     """Normalize text by removing extra whitespace, special characters, and converting to lowercase"""
#     if pd.isna(text):
#         return ""
#     return re.sub(r"\s+", " ", str(text).strip()).lower()


# Your existing functions, modified to use normalized text
def find_headers(dataframe):
    """Find and set the header row containing 'Date/Time' with normalized text"""
    header_idx = -1
    target = normalize_text("Date/Time")

    for ri, rows in enumerate(dataframe.values):
        for ci, row in enumerate(rows):
            if target in normalize_text(str(row)):
                header_idx = ri + 1
                break
        if header_idx != -1:
            break

    if header_idx != -1:
        print(f"Header found at row index: {header_idx}")
        dataframe = pd.DataFrame(
            dataframe.values[header_idx:], columns=dataframe.iloc[header_idx - 1]
        )
    else:
        print("Header not found in the DataFrame")
    return dataframe.reset_index(drop=True)


def normalize_text(text):
    """Normalize text by converting to lowercase and removing extra whitespace."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def find_footer(dataframe):
    """
    Identify and remove the footer starting with 'Station Down ...' in the 'Date/Time' column.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with footer rows removed and index reset.
    """
    # Check if 'Date/Time' column exists
    if "Date/Time" not in dataframe.columns:
        print("Column 'Date/Time' not found in the DataFrame")
        return dataframe

    # Normalize the target prefix for comparison
    target_prefix = "station down"

    # Normalize the 'Date/Time' column efficiently using vectorized operations
    normalized_col = dataframe["Date/Time"].apply(normalize_text)

    # Find the first row where the normalized text starts with 'station down'
    footer_start = normalized_col.str.startswith(target_prefix, na=False).idxmax()

    # If no match is found, idxmax returns 0 by default; check if it's a false match
    if not normalized_col.iloc[footer_start].startswith(target_prefix):
        print("Text 'Station Down ...' not found in the 'Date/Time' column")
        return dataframe

    print(f"Footer found starting at row index: {footer_start}")
    # Return DataFrame up to (but not including) the footer row
    trimmed_df = dataframe.iloc[:footer_start]

    # Reset index for consistency
    return trimmed_df.reset_index(drop=True)


def process_and_clean_files(base_dirs, excluded_files, output_dir="cleaned"):
    """
    Find, process, and clean xlsx files, saving to a cleaned folder.
    Test on 5 random files.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get all xlsx files
    all_xlsx_files = []
    for base_dir in base_dirs:
        dir_path = Path(base_dir)
        if not dir_path.exists():
            print(f"Warning: Directory {base_dir} does not exist")
            continue
        all_xlsx_files.extend(list(dir_path.rglob("*.xlsx")))

    # Filter out excluded files (case-insensitive)
    excluded_files = [f.lower() for f in excluded_files]
    valid_files = [f for f in all_xlsx_files if f.name.lower() not in excluded_files]

    if not valid_files:
        print("No valid Excel files found")
        return

    # Select 5 random files (or fewer if there aren't 5)
    sample_files = random.sample(valid_files, min(5, len(valid_files)))

    print(f"Processing {len(sample_files)} random files:")
    for file_path in sample_files:
        print(f"\nProcessing: {file_path}")

        try:
            # Read the Excel file
            df = pd.read_excel(file_path)

            # Apply cropping with normalized text
            df = find_headers(df)
            df = find_footer(df)

            # Create output filename (maintain original name in cleaned folder)
            output_file = output_path / file_path.name
            # Save cleaned data
            df.to_excel(output_file, index=False)
            print(f"Saved cleaned data to: {output_file}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")


# Specify the base directories and excluded files
base_directories = [
    "/mnt/e/MayThesis2025/datasets/ข้อมูลย้อนหลัง10ปี/ผลการตรวจวัดคุณภาพอากาศ",
    "/mnt/e/MayThesis2025/datasets/ข้อมูลย้อนหลัง10ปี/ผลการตรวจวัดอุตุนิยมวิทยา",
]

excluded_files = [
    "Main station WSHxls.xlsx",
    "Main station WSLxls.xlsx",
    "Main station WSMxls.xlsx",
    "Main station WDH.xlsx",
    "Main station WDL.xlsx",
    "Main station WDM.xlsx",
]

# Run the processing
if __name__ == "__main__":
    process_and_clean_files(base_directories, excluded_files)
