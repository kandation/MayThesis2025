import os
from pathlib import Path
import json
from typing import Dict, List, Any
from pythainlp.util import normalize

# Base path for air quality data
BASE_PATH = Path("/mnt/e/MayThesis2025/datasets/ข้อมูลย้อนหลัง10ปี/ผลการตรวจวัดคุณภาพอากาศ")

# Files to exclude
EXCLUDE_LIST = {
    "Main station WSHxls",
    "Main station WSLxls",
    "Main station WSMxls",
    "Main station WDM",
    "Main station WDH",
    "Main station WDL",
}

# Updated NAME_MAPPING
NAME_MAPPING = {
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
    "Main station": {"name_eng": "main station", "name_th": "สถานีตรวจอากาศหลัก"},
    "Government center": {
        "name_eng": "government center",
        "name_th": "ศูนย์ราชการแม่เมาะ",
    },
    "Banhanghung": {"name_eng": "banhanghung", "name_th": "บ้านหางฮุง"},
    "Banhuafai": {"name_eng": "banhuafai", "name_th": "บ้านหัวฝาย"},
    "Banhuaiking": {"name_eng": "banhuaiking", "name_th": "บ้านห้วยคิง"},
    "Banhuayking": {"name_eng": "banhuaiking", "name_th": "บ้านห้วยคิง"},
    "BanHuayking": {"name_eng": "banhuaiking", "name_th": "บ้านห้วยคิง"},
    "BanHuaFai": {"name_eng": "banhuafai", "name_th": "บ้านหัวฝาย"},
    "BanHuaiking": {"name_eng": "banhuaiking", "name_th": "บ้านห้วยคิง"},
    "Banmaechang": {"name_eng": "banmaechang", "name_th": "บ้านแม่จาง"},
    "Banmairattakosin": {"name_eng": "banmairattakosin", "name_th": "บ้านใหม่รัตนโกสินทร์"},
    "Banmairattanakosin": {
        "name_eng": "banmairattakosin",
        "name_th": "บ้านใหม่รัตนโกสินทร์",
    },
    "Bansadej": {"name_eng": "bansadej", "name_th": "บ้านเสด็จ"},
    "Bansopmoe": {"name_eng": "bansopmoe", "name_th": "บ้านสบเมาะ"},
    "Bansoppad": {"name_eng": "bansoppad", "name_th": "บ้านสบป้าด"},
    "Banthasii": {"name_eng": "banthasi", "name_th": "บ้านท่าสี"},
    "Pratupha": {"name_eng": "pratupha", "name_th": "ประตูผา"},
    "ประตูผา station": {"name_eng": "pratupha", "name_th": "ประตูผา"},
    "บ้นห้วยฝาย": {"name_eng": "banhuafai", "name_th": "บ้านหัวฝาย"},
    "บ้านห้วยฝาย": {"name_eng": "banhuafai", "name_th": "บ้านหัวฝาย"},
    "Mainstation": {"name_eng": "main station", "name_th": "สถานีตรวจอากาศหลัก"},
    "Maechang": {"name_eng": "banmaechang", "name_th": "บ้านแม่จาง"},
}


def clean_filename(filename: str) -> str:
    """Clean and normalize filename."""
    filename = Path(filename).stem
    return normalize(filename)


def is_excluded(filename: str) -> bool:
    """Check if a file should be excluded."""
    clean_name = clean_filename(filename)
    return any(exclude in clean_name for exclude in EXCLUDE_LIST)


def map_station_name(filename: str) -> Dict[str, str]:
    """Map a filename to station info with improved matching."""
    clean_name = clean_filename(filename)

    if clean_name in NAME_MAPPING:
        return NAME_MAPPING[clean_name]

    clean_name_lower = clean_name.lower().replace("ban", "").strip()
    for key, value in NAME_MAPPING.items():
        normalized_key = normalize(key).lower().replace("ban", "").strip()
        if normalized_key in clean_name_lower or clean_name_lower in normalized_key:
            return value

    print(f"Warning: No mapping found for '{clean_name}'")
    return {"name_eng": clean_name.lower(), "name_th": clean_name}


def get_year_category(file_path: Path) -> str:
    """Determine year category from file path."""
    path_str = str(file_path)
    if "2009" in path_str and "2010" not in path_str:
        return "2009"
    elif "2010-2020" in path_str or "2009-2019" in path_str:
        return "2010"
    return "unknown"


def get_subdirectory(file_path: Path) -> str:
    """Extract measurement type as subdirectory."""
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part == BASE_PATH.name and i + 1 < len(parts):
            sub_dir = parts[i + 1].split()[0].lower()
            return sub_dir
    return "unknown"


def get_all_files() -> List[Path]:
    """Retrieve all .xlsx files from the directory."""
    return list(BASE_PATH.rglob("*.xlsx"))


def organize_stations_data() -> Dict[str, Any]:
    """Organize station data into the required structure."""
    result = {}
    all_files = get_all_files()

    for file_path in all_files:
        if is_excluded(file_path.name):
            print(f"Excluded: {file_path}")
            continue

        station_info = map_station_name(file_path.name)
        eng_name = station_info["name_eng"]
        year = get_year_category(file_path)
        sub_dir = get_subdirectory(file_path)

        print(f"Processing: {file_path} -> {eng_name}, {year}, {sub_dir}")

        if eng_name not in result:
            result[eng_name] = {
                "info": {"name_th": station_info["name_th"], "name_eng": eng_name},
                "files": {},
            }

        if sub_dir not in result[eng_name]["files"]:
            result[eng_name]["files"][sub_dir] = {}

        result[eng_name]["files"][sub_dir][year] = str(file_path)

    return result


def validate_results(result: Dict[str, Any], all_files: List[Path]) -> List[str]:
    """Validate that all non-excluded files are in the result."""
    included_files = set()
    for station_data in result.values():
        for sub_dir_data in station_data["files"].values():
            included_files.update(sub_dir_data.values())

    missing_files = [
        str(file)
        for file in all_files
        if str(file) not in included_files and not is_excluded(file.name)
    ]
    return missing_files


def main():
    result = organize_stations_data()

    with open("organized_air_quality_data.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Organized air quality data saved to 'organized_air_quality_data.json'")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    all_files = get_all_files()
    missing_files = validate_results(result, all_files)

    if missing_files:
        print("\nWarning: The following files were not included:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print("\nSuccess: All files were organized correctly!")


if __name__ == "__main__":
    main()
