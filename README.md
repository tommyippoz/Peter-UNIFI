# Peter-UNIFI
Code repository for Peter' work started in Florence

## Set up Python Environment
The code needs Python 3.9 or higher to be installed in your device.
Then you can create a new virtual environment as follows.
- Open a cmd line and navigate to a folder in which it is ok to create the venv
- Type 'python -m venv <my_venv_name>', where <my_venv_name> is the name of your venv. This should create a subfolder with <my_venv_name>
- Download the 'package-list.txt' file from the repository and put it in the current directory
- Call '<my_venv_name>\Scripts\python.exe -m pip -r package-list.txt'. This will install all packages from the TXT file into the new virtual environment
Now the venv is ready to be used

## Datasets
Datasets are available at this <a href="https://drive.google.com/file/d/1vOU5rYcGPEWhNFp-S5EIWd-KzGG04Y2V/view?usp=sharing">link</a>, it is a ZIP protected with password
Password is required as datasets are not owned by us: we cannot redistribute
Make sure to unzip datasets in the same folder as the mani.py file, or update the CSV_FOLDER variable in the script to match where datasets are.

## Running Script
Download the 'main.py' file and exercise it within the venv you created before.
This can be done from command line as follows
- go to where the main.py file is (and where the datasets folder is)
- locate where your venv <my_venv_name> is with respect to the main.py folder (say this is <path_to_venv_name>)
- type '<path_to_venv_name>\Scripts\python.exe main.py'
The code hould start as expected

