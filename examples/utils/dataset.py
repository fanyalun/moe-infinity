import os
import pandas as pd

def load_realworldqa_dataset(data_dir):
    """Load RealWordQA dataset from parquet files"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory {data_dir} not found.")
    
    data_path = os.path.join(data_dir, "data")
    if not os.path.exists(data_path):
        # Try looking directly in data_dir
        data_path = data_dir
        
    files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.parquet')]
    
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")
        
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    data = pd.concat(dfs, ignore_index=True)
    # Add index as id if not present
    if 'id' not in data.columns:
        data['id'] = data.index
    return data

def load_mmbench_dataset(data_path):
    """Load MMBench dataset from TSV file"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} not found.")
    
    # MMBench TSV usually has 'index' as the first column
    df = pd.read_csv(data_path, sep='\t')
    return df

def load_mmstar_dataset(data_path):
    """Load MMStar dataset from TSV file"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} not found.")
    
    # MMBench TSV usually has 'index' as the first column
    df = pd.read_csv(data_path, sep='\t')
    return df

