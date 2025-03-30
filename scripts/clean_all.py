import os
import shutil


def remove_python_artifacts():
    """
    Remove Python file artifacts such as .pyc files and __pycache__ directories.
    """
    for root, dirs, files in os.walk("."):  # Walk through all directories and files
        for file in files:
            if file.endswith(".pyc"):
                file_path = os.path.join(root, file)
                print(f"Removing {file_path}")
                os.remove(file_path)
        for dir_name in dirs:
            if dir_name == "__pycache__":
                dir_path = os.path.join(root, dir_name)
                print(f"Removing {dir_path}")
                shutil.rmtree(dir_path)


def remove_build_artifacts():
    """
    Remove build-related artifacts like build/, dist/, and *.egg-info directories.
    """
    artifacts = ["build", "dist"]
    for artifact in artifacts:
        if os.path.exists(artifact):
            print(f"Removing {artifact} directory")
            shutil.rmtree(artifact)

    for item in os.listdir(
        "."
    ):  # Check for .egg-info files or directories in the current directory
        if item.endswith(".egg-info"):
            egg_info_path = os.path.join(".", item)
            print(f"Removing {egg_info_path}")
            shutil.rmtree(egg_info_path)


if __name__ == "__main__":
    print("Removing Python file artifacts...")
    remove_python_artifacts()
    print("Removing build-related artifacts...")
    remove_build_artifacts()
    print("Cleanup complete.")
