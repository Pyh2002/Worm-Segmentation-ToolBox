import os

# Function to delete files with a specific prefix
def delete_files_with_prefix(prefix, directory='.'):
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def delete_all_files_in_directory(directory='.'):
    delete_files_with_prefix("movement_graph_")
    delete_files_with_prefix("processed_data_")
    delete_files_with_prefix("trace_graph_")
    delete_files_with_prefix("video_info_")

if __name__ == "__main__":
    delete_all_files_in_directory()

