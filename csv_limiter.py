import os
import glob

def trim_csv_files(folder_path, max_lines=15000):
    # Get all csv files in the given folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    for file_path in csv_files:
        print(f"Processing: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) > max_lines:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines[:max_lines])
            print(f"Trimmed {file_path} to {max_lines} lines.")
        else:
            print(f"{file_path} already has {len(lines)} lines, no trimming needed.")

if __name__ == "__main__":
    folder = input("Enter the folder path containing CSV files: ").strip()
    trim_csv_files(folder)
