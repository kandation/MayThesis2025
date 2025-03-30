import os
import re
from typing import Dict, List, Tuple, Any
import glob

# Base path where the data is stored
BASE_PATH = "/mnt/e/MayThesis2025/datasets/ข้อมูลย้อนหลัง10ปี"

# Exclude list of files to ignore
exclude_list = ["Main station WSHxls", "Main station WSLxls", "Main station WSMxls", "Main station WDM", "Main station WDH", "Main station WDL"]

# Mapping of Thai station names to English
name_mapping = {
    # Thai station names with English equivalents
    "บ้านท่าสี": {"name_eng": "banthasi", "name_th": "บ้านท่าสี"},
    "บ้านสบป้าด": {"name_eng": "bansoppad", "name_th": "บ้านสบป้าด"},
    "บ้านสบเมาะ": {"name_eng": "bansopmoe", "name_th": "บ้านสบเมาะ"},
    "บ้านหัวฝาย": {"name_eng": "banhuafai", "name_th": "บ้านหัวฝาย"},
    "บ้านหางฮุง": {"name_eng": "banhanghung", "name_th": "บ้านหางฮุง"},
    "บ้านห้วยคิง": {"name_eng": "banhuaiking", "name_th": "บ้านห้วยคิง"},
    "บ้านเสด็จ": {"name_eng": "bansadej", "name_th": "บ้านเสด็จ"},
    "บ้านแม่จาง": {"name_eng": "banmaechang", "name_th": "บ้านแม่จาง"},
    "บ้านใหม่รัตนโกสินทร์": {"name_eng": "banmairattakosin", "name_th": "บ้านใหม่รัตนโกสินทร์"},
    "ประตูผา": {"name_eng": "pratupha", "name_th": "ประตูผา"},
    "ศูนย์ราชการแม่เมาะ": {"name_eng": "government center", "name_th": "ศูนย์ราชการแม่เมาะ"},
    "สถานีตรวจอากาศหลัก": {"name_eng": "main station", "name_th": "สถานีตรวจอากาศหลัก"},
    "สถานีหลัก": {"name_eng": "main station", "name_th": "สถานีตรวจอากาศหลัก"},
    # Additional mappings for English names in 2009 folder
    "Main station": {"name_eng": "main station", "name_th": "สถานีตรวจอากาศหลัก"},
    "Government center": {"name_eng": "government center", "name_th": "ศูนย์ราชการแม่เมาะ"},
    "Banhanghung": {"name_eng": "banhanghung", "name_th": "บ้านหางฮุง"},
    "Banhuafai": {"name_eng": "banhuafai", "name_th": "บ้านหัวฝาย"},
    "Banhuaiking": {"name_eng": "banhuaiking", "name_th": "บ้านห้วยคิง"},
    "Banhuayking": {"name_eng": "banhuaiking", "name_th": "บ้านห้วยคิง"},
    "Banmaechang": {"name_eng": "banmaechang", "name_th": "บ้านแม่จาง"},
    "Banmairattakosin": {"name_eng": "banmairattakosin", "name_th": "บ้านใหม่รัตนโกสินทร์"},
    "Banmairattanakosin": {"name_eng": "banmairattakosin", "name_th": "บ้านใหม่รัตนโกสินทร์"},
    "Bansadej": {"name_eng": "bansadej", "name_th": "บ้านเสด็จ"},
    "Bansopmoe": {"name_eng": "bansopmoe", "name_th": "บ้านสบเมาะ"},
    "Bansoppad": {"name_eng": "bansoppad", "name_th": "บ้านสบป้าด"},
    "Banthasii": {"name_eng": "banthasi", "name_th": "บ้านท่าสี"},
    "Pratupha": {"name_eng": "pratupha", "name_th": "ประตูผา"},
    # Add variations based on problematic filenames in the dataset
    "ประตูผา station": {"name_eng": "pratupha", "name_th": "ประตูผา"},
    "บ้นห้วยฝาย": {"name_eng": "banhuafai", "name_th": "บ้านหัวฝาย"},
    "บ้านห้วยฝาย": {"name_eng": "banhuafai", "name_th": "บ้านหัวฝาย"},
}

def clean_filename(filename: str) -> str:
    """
    Clean filename by removing .xlsx extension and normalizing
    """
    # Remove extension
    filename = os.path.splitext(filename)[0]
    return filename

def get_all_station_files() -> List[str]:
    """
    Get all meteorological data files from both 2009 and 2010-2020 directories
    """
    data_path = os.path.join(BASE_PATH, "ผลการตรวจวัดอุตุนิยมวิทยา")
    all_files = []
    
    # Path for 2009 data
    path_2009 = os.path.join(data_path, "2009")
    # Path for 2010-2020 data
    path_2010_2020 = os.path.join(data_path, "2010-2020")
    
    # Recursive search for all xlsx files
    for root_dir in [path_2009, path_2010_2020]:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.xlsx'):
                    all_files.append(os.path.join(root, file))
    
    return all_files

def should_exclude(filename: str) -> bool:
    """
    Check if a file should be excluded based on the exclude list
    """
    clean_name = clean_filename(filename)
    for exclude_item in exclude_list:
        if exclude_item in clean_name:
            return True
    return False

def map_station_name(filename: str) -> Dict[str, str]:
    """
    Map a filename to station info (Thai and English names)
    """
    clean_name = clean_filename(os.path.basename(filename))
    
    # Try direct mapping
    if clean_name in name_mapping:
        return name_mapping[clean_name]
    
    # If not found, try case-insensitive matching
    for key, value in name_mapping.items():
        if key.lower() == clean_name.lower():
            return value
    
    # If still not found, try partial matching
    for key, value in name_mapping.items():
        if key.lower() in clean_name.lower() or clean_name.lower() in key.lower():
            return value
    
    # Return default if no match found
    return {"name_eng": clean_name, "name_th": clean_name}

def get_year_category(file_path: str) -> str:
    """
    Determine if a file is from 2009 or 2010-2020 based on its path
    """
    if "/2009/" in file_path:
        return "2009"
    elif "/2010-2020/" in file_path:
        return "2010"
    else:
        return "unknown"

def get_subdirectory(file_path: str) -> str:
    """
    Extract the subdirectory name (BP, DEV, etc.) from the file path
    """
    parts = file_path.split(os.sep)
    for i, part in enumerate(parts):
        if part in ["2009", "2010-2020"] and i+1 < len(parts):
            return parts[i+1].lower()
    return "unknown"

def organize_stations_data() -> Dict[str, Any]:
    """
    Organize all station data according to the required structure
    """
    result = {}
    all_files = get_all_station_files()
    
    for file_path in all_files:
        # Skip excluded files
        if should_exclude(os.path.basename(file_path)):
            continue
            
        # Get station info
        station_info = map_station_name(os.path.basename(file_path))
        eng_name = station_info["name_eng"].lower()
        
        # Get year category and subdirectory
        year_category = get_year_category(file_path)
        sub_dir = get_subdirectory(file_path)
        
        # Initialize station entry if not exists
        if eng_name not in result:
            result[eng_name] = {
                "info": {
                    "name_th": station_info["name_th"],
                    "name_eng": eng_name
                },
                "files": {}
            }
        
        # Initialize subdirectory if not exists
        if sub_dir not in result[eng_name]["files"]:
            result[eng_name]["files"][sub_dir] = {}
        
        # Add file path to appropriate year category
        result[eng_name]["files"][sub_dir][year_category] = file_path
    
    return result

def validate_results(result: Dict[str, Any], all_files: List[str]) -> List[str]:
    """
    Validate that all files are included in the result (except excluded ones)
    Return list of files not included in result
    """
    included_files = []
    for station_data in result.values():
        for sub_dir_data in station_data["files"].values():
            for file_path in sub_dir_data.values():
                included_files.append(file_path)
    
    missing_files = []
    for file_path in all_files:
        if file_path not in included_files and not should_exclude(os.path.basename(file_path)):
            missing_files.append(file_path)
    
    return missing_files

def main():
    # Organize all stations data
    result = organize_stations_data()
    
    # Validate results
    all_files = get_all_station_files()
    missing_files = validate_results(result, all_files)
    
    # Output results
    print("Organized meteorological data:")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Report missing files
    if missing_files:
        print("\nWarning: Some files were not included in the result:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print("\nAll files successfully organized!")

if __name__ == "__main__":
    main()