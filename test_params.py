def params(dataset:str):
    best_params = None
    if dataset == 'adult':
        best_params = {
        "n_trees": 128,
        "depth": 8,
        "lr": 0.002581913110939373,
        "weight_decay": 5.517801628551815e-06,
        "batch_size": 4096,
        "dropout": 0.14912115999013786,
        "label_smoothing": 0.04637834583445727
        }
    elif dataset=='covertype':
        best_params = {
        "lr": 0.003148121658590051,
        "weight_decay": 0.001254566975334968,
        "batch_size": 4096,
        "depth": 10,
        "n_trees": 152,
        "dropout": 0.001172966795034678,
        "label_smoothing": 0.07605117024796712
        }
    elif dataset == 'helena':
        best_params = {
        "lr": 0.002184330094608213,
        "weight_decay": 0.0007588480288102768,
        "batch_size": 4096,
        "depth": 4,
        "n_trees": 128,
        "dropout": 0.15270768019648032,
        "label_smoothing": 0.030844888867940973
        }
    elif dataset=='higgs':
        best_params = {
        "n_trees": 128,
        "depth": 6,
        "lr": 0.0005570439595587535,
        "weight_decay": 0.002399464620511693,
        "batch_size": 4096,
        "dropout": 0.23660561539379882,
        "label_smoothing": 0.027405871164800394
        }

    elif dataset=='epsilon':
        best_params = {
        "lr": 0.00012539772230485877,
        "weight_decay": 0.0007839316798099645,
        "batch_size": 4096,
        "depth": 4,
        "n_trees": 112,
        "dropout": 0.2836212718617301,
        "label_smoothing": 0.05574958984466744
        }
    return best_params
