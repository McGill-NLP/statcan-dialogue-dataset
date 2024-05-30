from huggingface_hub import HfApi

# save files to repository called McGill-NLP/statcan-dialogue-dataset-retrieval
# this will be the repository that the dataset will be uploaded to

data_dir = 'retrieval_task/data_out'
repo_name = 'McGill-NLP/statcan-dialogue-dataset-retrieval'

api = HfApi()
api.create_repo(repo_name, repo_type='dataset', exist_ok=True)
api.upload_folder(repo_id=repo_name, repo_type='dataset', folder_path=data_dir, commit_message="Initial dataset upload")