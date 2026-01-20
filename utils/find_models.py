def find_models(base_path: str, session: str) -> list[str]:
    import os
    
    if not os.path.exists(os.path.join(base_path, session)):
        return []

    model_folder = os.path.join(base_path, session)
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]
    return model_files

def find_last_model(base_path: str, session: str) -> str | None:
    import os
    from datetime import datetime

    model_files = find_models(base_path, session)
    if not model_files:
        return None

    model_files.sort(key=lambda x: datetime.strptime(x.split('_')[-2] + '_' + x.split('_')[-1].replace('.pth', ''), '%d-%m-%y_%Hh%M'), reverse=True)
    return model_files[0]