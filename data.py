import numpy as np
import torch
from pathlib import Path
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

def download_dataset_covertype(cache_dir):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)   
    return fetch_covtype(data_home=cache_dir)
    
def verify_covtype(cache_dir):
    base_path = Path(cache_dir) / "covertype"

    print("checking:", base_path.resolve())
    feature_exist = (base_path / "samples_py3").is_file()
    target_path = (base_path / "targets_py3").is_file()

    return feature_exist and target_path

def process_covertype(dataset : Bunch):
    X = dataset.data.astype(np.float32)
    y = (dataset.target-1).astype(np.int64)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.10, random_state=42, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 9, random_state=42, shuffle=True,stratify=y_train_val)

    file_path = Path('./dataset_cache/covertype')
    file_path.mkdir(parents=True,exist_ok=True)

    np.savez(file_path / "covertype_train_test_val.npz", X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,X_val=X_val,y_val=y_val)
    torch.save({
        "X_train": torch.from_numpy(X_train),
        "y_train": torch.from_numpy(y_train),
        "X_val": torch.from_numpy(X_val),
        "y_val": torch.from_numpy(y_val),
        "X_test": torch.from_numpy(X_test),
        "y_test": torch.from_numpy(y_test),
        },
        file_path / "covertype_train_test_val.pt"
        )
    print(f"Saved numpy and pt file in {file_path}")

def file_verification(filepath: Path)-> bool:
    return filepath.exists()

if __name__=="__main__":
    data_path = "."

    filepath_np = Path('./dataset_cache/covertype/covertype_train_test_val.npz')
    filepath_pt = Path('./dataset_cache/covertype/covertype_train_test_val.pt')
    if file_verification(filepath_np) and file_verification(filepath_pt):
        print("Covertype Cache exists")
        covtype = fetch_covtype(data_home=data_path,download_if_missing=False)
    else:
        if verify_covtype(data_path):
            print("dataset already downloaded")
            covtype = fetch_covtype(data_home=data_path,download_if_missing=False)

            process_covertype(covtype)
        else:
            print(" Covertype dataset not downloaded")
            covtype = download_dataset_covertype(data_path)  
            process_covertype(covtype) 
    print(covtype.keys())

    


