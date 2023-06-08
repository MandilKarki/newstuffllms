import os
import zipfile

def extract_zip_file(zip_filepath, destination_path):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        for member in zip_ref.namelist():
            filename_path = os.path.normpath(os.path.join(destination_path, member))
            upperdirs = os.path.dirname(filename_path)
            if upperdirs and not os.path.exists(upperdirs):
                os.makedirs(upperdirs)
            if member.endswith('/'):  # A directory
                if not os.path.isdir(filename_path):
                    os.mkdir(filename_path)
            else:  # A file
                with open(filename_path, "wb") as source, zip_ref.open(member) as target:
                    source.write(target.read())
            print(member)

# Usage:
zip_filepath = 'C:/Mandil/emailer/AI_Chatbot_In_Python_With_Source_Code.zip'  # Replace with the path to your zip file
destination_path = 'C:/Mandil/emailer'  # Replace with the path where you want to extract the zip
extract_zip_file(zip_filepath, destination_path)
