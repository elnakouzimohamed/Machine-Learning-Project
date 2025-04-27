# utils/file_manager.py

import os

def manage_uploads_folder(upload_dir):
    """
    Ensure the uploads folder exists and is clean before new upload.
    """
    os.makedirs(upload_dir, exist_ok=True)
    # Remove any existing files
    for file_name in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def save_uploaded_file(uploaded_file, upload_dir):
    """
    Save the uploaded Streamlit file into the upload directory.
    Returns:
        str: Full path of saved file
    """
    extension = uploaded_file.name.split('.')[-1]
    new_filename = f"uploaded_data.{extension}"
    full_path = os.path.join(upload_dir, new_filename)

    with open(full_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return full_path
