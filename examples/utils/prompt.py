import pandas as pd
import os
import base64
import hashlib
from PIL import Image
import io

def dump_image(line, img_root):
    os.makedirs(img_root, exist_ok=True)
    tgt_path = []
    
    if 'image' in line and pd.notna(line['image']):
        val = line['image']
        
        def save_img(img_val, idx=None):
            """Save image with content-based hash filename to avoid duplicates"""
            try:
                img_bytes = None
                # Handle dict with bytes
                if isinstance(img_val, dict) and 'bytes' in img_val:
                    img_bytes = img_val['bytes']
                # Handle bytes directly
                elif isinstance(img_val, bytes):
                    img_bytes = img_val
                # Handle base64 string
                elif isinstance(img_val, str):
                    try:
                        # Remove data URI prefix if present
                        if img_val.startswith('data:image'):
                            img_val = img_val.split(',', 1)[1]
                        img_bytes = base64.b64decode(img_val)
                    except Exception:
                        # If not base64, might be a URL or file path - return as is
                        return img_val
                
                if img_bytes:
                    # Use MD5 hash of image content as filename
                    img_hash = hashlib.md5(img_bytes).hexdigest()
                    path = os.path.join(img_root, f"{img_hash}.png")
                    
                    # Only save if not already exists (deduplication)
                    if not os.path.exists(path):
                        try:
                            image = Image.open(io.BytesIO(img_bytes))
                            image = image.convert('RGB')  # 转换为RGB以处理某些格式问题
                            image.save(path)
                        except (IOError, OSError) as e:
                            # 处理图像加载失败
                            print(f"Warning: Failed to load/save image: {e}")
                            return None
                    
                    return path
                    
            except Exception as e:
                print(f"Warning: Failed to save image: {e}")
            return None

        if isinstance(val, list):
            for i, v in enumerate(val):
                p = save_img(v, i)
                if p: tgt_path.append(p)
        else:
            p = save_img(val)
            if p: tgt_path.append(p)
            
    return tgt_path

def build_mmbench_prompt(line, img_root):
    """Build prompt for MMBench"""
    question = f"Question: {str(line.get('question', ''))}"
    hint = str(line.get('hint', ''))
    
    # 如果有 hint，拼接到 question 前面
    if pd.notna(hint) and hint != 'nan' and hint.strip():
        question = f"Hint: {hint}\n{question}"
        
    options = []
    for char in ['A', 'B', 'C', 'D', 'E', 'F']:
        if char in line and pd.notna(line[char]):
            val = str(line[char])
            options.append(f"{char}. {val}")
    
    option_str = "\n".join(options)
    
    prompt = f"{question}\nOptions:\n{option_str}\nAnswer with the option's letter from the given choices directly.\nAnswer:"
    
    # Handle image
    image_paths = []
    if 'image' in line and pd.notna(line['image']):
        image_paths = dump_image(line, img_root)
        
    content = []
    for img_path in image_paths:
        content.append({"type": "image", "image": img_path})
    
    content.append({"type": "text", "text": prompt})
    
    messages = [{"role": "user", "content": content}]
    return messages

def build_mmstar_prompt(line, img_root):    
    """Build prompt for MMStar"""
    image_paths = dump_image(line, img_root)
    question = line['question']
    prompt = f"{question}\nAnswer with the option letter only.\nThe answer is:"
    
    content = []
    for image_path in image_paths:
        content.append({"type": "image", "image": image_path})
    
    content.append({"type": "text", "text": prompt})
    
    messages = [{"role": "user", "content": content}]
    return messages

def build_realworldqa_prompt(line, img_root):
    """Build prompt for RealWordQA"""
    image_paths = dump_image(line, img_root)
    question = line['question']
    # RealWordQA questions usually include options like "A. ... B. ..."
    prompt = f"Question: {question}\nAnswer:"
    
    content = []
    for image_path in image_paths:
        content.append({"type": "image", "image": image_path})
    
    content.append({"type": "text", "text": prompt})
    
    messages = [{"role": "user", "content": content}]
    return messages




