import numpy as np
import torch
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import Bunch
from sklearn.datasets import fetch_openml
# from capymoa.datasets import CovtFD

def download_dataset_covertype(cache_dir):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)   
    return fetch_covtype(data_home=cache_dir)
    
def verify_covtype(cache_dir):
    base_path = Path(cache_dir) / "covertype"

    print("checking:", base_path.resolve())
    feature_exist = (base_path / "samples_py3").is_file()
    target_path = (base_path / "targets_py3").is_file()

    return feature_exist and target_path

def process_covertype(dataset : Bunch, cache_dir:Path):
    X = dataset.data.astype(np.float32)
    y = (dataset.target-1).astype(np.int64)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.10, random_state=42, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 9, random_state=42, shuffle=True,stratify=y_train_val)

    scaler = StandardScaler()
    X_train[:, :10] = scaler.fit_transform(X_train[:, :10]).astype(np.float32)
    X_val[:, :10] = scaler.transform(X_val[:,:10]).astype(np.float32)
    X_test[:,:10] = scaler.transform(X_test[:,:10]).astype(np.float32)

    file_path = cache_dir
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

def process_helena(dataset: Bunch, cache_dir:Path):
    X = dataset.data.astype(np.float32)
    y_raw = dataset.target
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.10, random_state=42, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 9, random_state=42, shuffle=True,stratify=y_train_val)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    file_path = cache_dir
    file_path.mkdir(parents=True,exist_ok=True)
    np.savez(file_path / "helena_train_test_val.npz", X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,X_val=X_val,y_val=y_val)
    torch.save({
        "X_train": torch.from_numpy(X_train),
        "y_train": torch.from_numpy(y_train),
        "X_val": torch.from_numpy(X_val),
        "y_val": torch.from_numpy(y_val),
        "X_test": torch.from_numpy(X_test),
        "y_test": torch.from_numpy(y_test),
        },
        file_path / "helena_train_test_val.pt"
        )
    print(f"Saved numpy and pt file in {file_path.resolve()}")

def process_higgs(higgs_csv, cache_dir:Path):
    data = np.loadtxt(higgs_csv, delimiter=",", dtype=np.float32)
    X = data[:, 1:].astype(np.float32)
    y = data[:, 0].astype(np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.10, random_state=42, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 9, random_state=42, shuffle=True,stratify=y_train_val)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    file_path = cache_dir
    file_path.mkdir(parents=True,exist_ok=True)
    # np.savez(file_path / "higgs_train_test_val.npz", X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,X_val=X_val,y_val=y_val)
    torch.save({
        "X_train": torch.from_numpy(X_train),
        "y_train": torch.from_numpy(y_train),
        "X_val": torch.from_numpy(X_val),
        "y_val": torch.from_numpy(y_val),
        "X_test": torch.from_numpy(X_test),
        "y_test": torch.from_numpy(y_test),
        },
        file_path / "higgs_train_test_val.pt"
        )
    print(f"Saved numpy and pt file in {file_path.resolve()}")

def process_epsilon(npz_path, cache_dir:Path):
    d = np.load(npz_path)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(np.int64)
    X_val = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(np.int64)
    X_test = d["X_test"].astype(np.float32)
    y_test = d["y_test"].astype(np.int64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    file_path = cache_dir
    file_path.mkdir(parents=True,exist_ok=True)
    torch.save({
    "X_train": torch.from_numpy(X_train),
    "y_train": torch.from_numpy(y_train),
    "X_val": torch.from_numpy(X_val),
    "y_val": torch.from_numpy(y_val),
    "X_test": torch.from_numpy(X_test),
    "y_test": torch.from_numpy(y_test),
    },
    file_path / "epsilon_train_test_val.pt"
    )
    print(f"Saved numpy and pt file in {file_path.resolve()}")

def process_adult(np_tr, np_te, np_val, cache_dir:Path):
    tr, va, te = np.load(np_tr), np.load(np_val), np.load(np_te)
    X_train, y_train = tr[:, 1:].astype(np.float32), tr[:, 0].astype(np.int64)
    X_val, y_val = va[:, 1:].astype(np.float32), va[:, 0].astype(np.int64)
    X_test, y_test = te[:, 1:].astype(np.float32), te[:, 0].astype(np.int64)

    file_path = cache_dir
    file_path.mkdir(parents=True,exist_ok=True)
    torch.save({
    "X_train": torch.from_numpy(X_train),
    "y_train": torch.from_numpy(y_train),
    "X_val": torch.from_numpy(X_val),
    "y_val": torch.from_numpy(y_val),
    "X_test": torch.from_numpy(X_test),
    "y_test": torch.from_numpy(y_test),
    },
    file_path / "adult_train_test_val.pt"
    )
    print(f"Saved numpy and pt file in {file_path.resolve()}")


def load_data(cache_dir: Path, filename):
    file = Path(cache_dir/filename).resolve() 

    if not file.exists():
        raise FileNotFoundError(
            f"File does not exist in the {cache_dir} "
        )

    data = torch.load(file, map_location="cpu", weights_only=True)

    return (
        data["X_train"],
        data["y_train"], 
        data["X_val"], 
        data["y_val"], 
        data["X_test"],
        data["y_test"]
    )

# def capymoa_covtype_test():
#     download_location = Path("./covfd")
#     download_location.mkdir(parents=True, exist_ok=True)
#     covfd = CovtFD("./covfd",auto_download=False ,file_type="csv")
#     return covfd


if __name__=="__main__":
    data_path = Path('.')
    cache_dir = Path("./dataset_cache/covertype")
    filepath_np = cache_dir / 'covertype_train_test_val.npz'
    filepath_pt = cache_dir / 'covertype_train_test_val.pt' 

    if filepath_np.exists() and filepath_pt.exists():
        print("Covertype Cache exists")
    else:
        if verify_covtype(data_path):
            print("dataset already downloaded")
            covtype = fetch_covtype(data_home=data_path,download_if_missing=False)

            process_covertype(covtype, data_path)
        else:
            print(" Covertype dataset not downloaded")
            covtype = download_dataset_covertype(data_path)  
            process_covertype(covtype, data_path) 

    helena_data = Path('./dataset_cache/helena')
    helena_np = helena_data / 'helena_train_test_val.npz'
    helena_pt = helena_data / 'helena_train_test_val.pt'

    if helena_np.exists() and helena_pt.exists():
        print("helena Cache exists")
    else:
        print("Processing Helena Dataset") 
        dataset = fetch_openml(data_id=41169, as_frame=False, parser="auto", data_home=helena_data)
        process_helena(dataset, helena_data)

    higgs_csv = "./dataset_cache/higgs/HIGGS.csv"
    higgs_path = Path("./dataset_cache/higgs")
    higgs_file = higgs_path / 'higgs_train_test_val.pt'

    if higgs_file.exists():
        print("Higgs cache exist")
    else:
        process_higgs(higgs_csv,higgs_path )

    epsilon_path = Path("./dataset_cache/epsilon")
    epsilon_file = epsilon_path / "epsilon_train_test_val.pt"
    epsilon_npz = epsilon_path / 'epsilon_train_test_val.npz'
    if epsilon_file.exists():
        print("epsilon cache exist")
    else:
        process_epsilon(epsilon_npz, epsilon_path)

    adultincome_path = Path("./dataset_cache/adult")
    adult_file  = adultincome_path / 'adult_train_test_val.pt'
    np_train = "./dataset_cache/adult/adult_train.npy"
    np_test = "./dataset_cache/adult/adult_test.npy"
    np_val = "./dataset_cache/adult/adult_val.npy"
    if adult_file.exists():
        print("adult cache exist")
    else:
        process_adult(np_train, np_test, np_val, adultincome_path)




