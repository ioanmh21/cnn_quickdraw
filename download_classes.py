import os
import requests

classes = []

f=open('classes.txt','r')
for line in f.readlines():
    line=line.split()
    for x in line:
        if x not in classes:
            classes.append(x)
f.close()

dest_dir = 'quickdraw_data'
os.makedirs(dest_dir, exist_ok=True)

base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

def download_file(class_name):
    formatted_class_name = class_name.lower().replace(" ", "_")
    file_url = f"{base_url}{formatted_class_name}.npy"
    save_path = os.path.join(dest_dir, f"{formatted_class_name}.npy")

    try:
        print(f"Downloading {formatted_class_name}.npy from {file_url}...")
        response = requests.get(file_url)

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ {formatted_class_name}.npy saved!")
        else:
            print(f"❌ Failed to download {formatted_class_name}.npy (HTTP {response.status_code})")
    except Exception as e:
        print(f"❌ Error downloading {formatted_class_name}.npy: {e}")

for class_name in classes:
    download_file(class_name)