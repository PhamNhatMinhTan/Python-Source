import os
import urllib.request
import tarfile

### Create data folder in current folder ###
data_dir = "./data"     # Directory path to data folder

# Check folder "data" exist or not
if not os.path.exists(data_dir):
    # If data folder is not exist then create new folder name "data"
    os.mkdir(data_dir)

### Download VOC 2012 dataset to the data folder ###
# Link URL to download page
url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
# Create the save directory path and file's name when downloading from the url
target_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar")

# Check the save directory path is exist or not
# If is not exist then download and extract file
if not os.path.exists(target_path):
    # Download file from url and save to target_path
    urllib.request.urlretrieve(url, target_path)

    # Extract file
    # Put the downloaded file to tarfile to handle tar file
    tar = tarfile.TarFile(target_path)
    # Extract all data from target_path to data_dir
    tar.extractall(data_dir)
    tar.close()  # Close file after processing
