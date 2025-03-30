"""
Read the XLS file and convert it to XLSX file
"""

from pathlib import Path
import win32com.client as win32
import os


def convert_xls_to_xlsx(path: Path) -> None:
    excel = win32.gencache.EnsureDispatch("Excel.Application")
    wb = excel.Workbooks.Open(path.absolute())

    # FileFormat=51 is for .xlsx extension
    wb.SaveAs(str(path.absolute().with_suffix(".xlsx")), FileFormat=51)
    wb.Close()
    excel.Application.Quit()


uri = r"D:\Work\Home-2023\may-project\project-mei\ข้อมูลย้อนหลัง10ปี\ผลการตรวจวัดอุตุนิยมวิทยา"
uri = r"E:\MayThesis2025\datasets"
#
#
# # Define the directory where you want to start the conversion
root_dir = Path(uri)
# convert_xls_to_xlsx(root_dir)
#
# Recursively traverse the directory and convert XLS to XLSX
# for root, _, files in os.walk(root_dir):
#     for file in files:
#         if file.endswith(".xls"):
#             xls_path = Path(root) / file
#             print("Converting:", xls_path)
#             convert_xls_to_xlsx(xls_path)

# print("Conversion complete.")


def remove_xls_files(directory_path: Path) -> None:
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.xls'):
                xls_path = Path(root) / file
                xls_path.unlink()
                print(f"Deleted: {xls_path}")



# Call the function to remove XLS files
remove_xls_files(root_dir)

print("XLS file removal complete.")
