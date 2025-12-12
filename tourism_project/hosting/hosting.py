from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",   #local folder containing files
    repo_id="rojasnath/Tourism-Project",        #target repo
    repo_type="space",
    path_in_repo="",#dataset, model or space
)
