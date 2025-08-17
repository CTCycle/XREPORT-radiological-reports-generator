import requests, zipfile, io, os, shutil

GITHUB_ZIP_URL = "https://github.com/USER/REPO/archive/refs/heads/main.zip"
ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

print("[INFO] Downloading latest version...")
r = requests.get(GITHUB_ZIP_URL)
z = zipfile.ZipFile(io.BytesIO(r.content))

temp_folder = os.path.join(ROOT_FOLDER, "_update_temp")
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
z.extractall(temp_folder)

# Find extracted folder name
extracted_dir = next(os.scandir(temp_folder)).path

print("[INFO] Updating files...")
for item in os.listdir(extracted_dir):
    src = os.path.join(extracted_dir, item)
    dst = os.path.join(ROOT_FOLDER, item)
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    shutil.move(src, dst)

shutil.rmtree(temp_folder)
print("[SUCCESS] Update complete.")