import os
import torch
import importlib.resources as pkg_resources

def load_dataset(dataset_name='interactions_human'):
    valid_datasets = [
        'interactions_human',
        'interactions_human_nonichenetv2',
        'interactions_mouse',
        'interactions_mouse_nonichenetv2'
    ]
    if dataset_name not in valid_datasets:
        raise ValueError(f"Invalid dataset name. Choose from {', '.join(valid_datasets)}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', f'{dataset_name}.pth')
    if os.path.exists(data_path):
        database = torch.load(data_path, weights_only=False)
        return database

def get_unique_lists(second_order_list):
    results = [list(t) for t in {tuple(l) for l in second_order_list}]
    print("There are", len(results), "ligands measured in this dataset")
    return results

def search_ligands(genes, species, use_nichenetv2=True, select_liangd_strict=True):
    strict = select_liangd_strict
    assert species in ["human", "mouse"], "Species must be one of human or mouse"
    if species == "human":
        if use_nichenetv2:
            database = load_dataset('interactions_human')
        else:
            database = load_dataset('interactions_human_nonichenetv2')
    elif species == "mouse":
        if use_nichenetv2:
            database = load_dataset('interactions_mouse')
        else:
            database = load_dataset('interactions_mouse_nonichenetv2')
    else:
        raise ValueError("Species must be one of human or mouse")
    ligands = []
    sources = {}
    steps = {}
    for interactioni in database:
        flag = None
        if strict:
            flag = True
            for ligandi in interactioni[0]:
                if ligandi not in genes:
                    flag = False
            if flag:
                ligands.append(interactioni[0])
                sources["".join(interactioni[0])] = interactioni[-1]
                steps["".join(interactioni[0])] = interactioni[1]
        else:
            flag = False
            used_ligands = []
            for ligandi in interactioni[0]:
                if ligandi in genes:
                    used_ligands.append(ligandi)
                    flag = True
            if flag:
                ligands.append(used_ligands)
                sources["".join(used_ligands)] = interactioni[-1]
                steps["".join(used_ligands)] = interactioni[1]
    ligands = get_unique_lists(ligands)
    sources = [sources["".join(ligandi)] for ligandi in ligands]
    steps = [steps["".join(ligandi)] for ligandi in ligands]
    if not os.path.exists(os.path.join(os.getcwd(), "data")):
        os.mkdir(os.path.join(os.getcwd(), "data"))
    ligand_path = os.path.join(os.getcwd(), 'data', 'ligands.pth')
    torch.save([ligands, steps], ligand_path)
    return [ligands, steps, sources]

