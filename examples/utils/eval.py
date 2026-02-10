import os
import json    
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def eval_realworldqa(results_file=None, results=None, output_dir=None):
    """
    Evaluate RealWorldQA results from a JSON file.
    """
    # Load results
    if results is None:
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    
    # Calculate Accuracy
    correct = 0
    total = 0
    for r in results:
        pred = r['prediction'].strip()
        gt = r['gt_answer'].strip()
        
        is_correct = False
        
        # 1. Handle Yes/No/Numbers (Case insensitive)
        if gt.lower() in ["yes", "no"] or gt.isdigit():
            # Remove punctuation and lower case
            pred_norm = re.sub(r'[^\w\s]', '', pred).lower()
            gt_norm = re.sub(r'[^\w\s]', '', gt).lower()
            pred_words = pred_norm.split()
            if pred_words and pred_words[0] == gt_norm:
                is_correct = True
                
        # 2. Handle MCQ (Single Letter A-F)
        elif len(gt) == 1 and gt.upper() in "ABCDEF":
            gt_char = gt.upper()
            
            # 构造检查列表：优先检查全文，如果有多行，再额外检查最后一行
            check_contents = [pred]
            lines = pred.strip().split('\n')
            if len(lines) > 1:
                check_contents.append(lines[-1])
            
            found_match = False
            for content in check_contents:
                if not content: continue
                
                # --- 优先策略：开头匹配 ---
                
                # Check for "(A)" at start
                match = re.match(r'^\s*\(([A-F])\)', content, re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    is_correct = True
                    found_match = True
                    break

                # Check for "A.", "A)", "A " at start
                match = re.match(r'^\s*([A-F])[\.\)\s]', content, re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    is_correct = True
                    found_match = True
                    break

                # Check for exact match "A" (entire content is just the letter)
                if content.strip().upper() == gt_char:
                    is_correct = True
                    found_match = True
                    break

                # --- 语义策略：关键词提取 ---

                # Check for "The answer/option/choice is A"
                # 扩展支持 "Option is A", "Choice is A" 等变体
                if any(k in content.lower() for k in ["answer", "option", "choice"]):
                    match = re.search(r'(?:answer|option|choice)(?: is|:| is:)?\s*\(?([A-F])\)?', content, re.IGNORECASE)
                    if match and match.group(1).upper() == gt_char:
                        is_correct = True
                        found_match = True
                        break
                
                # --- 兜底策略：文末匹配 (针对 "at the end" Hint) ---
                
                # Check for letter at the very end of the string
                # Pattern: 非字母数字字符(或开头) + 字母 + 可选标点(.) + 字符串结尾
                # 例如: "... therefore A.", "... option C", "... (D)"
                match = re.search(r'(?:^|\s|[^a-zA-Z0-9])([A-F])[\.\)]?$', content.strip(), re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    is_correct = True
                    found_match = True
                    break
            
            # Fallback: starts with letter (Only check original pred to avoid false positives)
            if not found_match and len(pred) > 0 and pred[0].upper() == gt_char:
                is_correct = True
                
        # 3. Fallback exact match
        elif pred.lower() == gt.lower():
            is_correct = True
            
        if is_correct:
            correct += 1
        total += 1
        
    acc = correct / total if total > 0 else 0
    print(f"\nAccuracy: {acc:.4f} ({correct}/{total})")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    os.makedirs(output_dir, exist_ok=True)
    acc_file = os.path.join(output_dir, "accuracy.json")
    with open(acc_file, 'w') as f:
        json.dump({"accuracy": acc, "correct": correct, "total": total}, f, indent=2)
    
    print(f"Accuracy saved to {acc_file}")


def eval_mmbench(results_file=None, results=None, output_dir=None):
    """
    Evaluate MMBench results.
    """
    # 1. Load results
    if results is None:
        if results_file is None or not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        print(f"Loading results from {results_file}...")
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    
    # 2. Define Answer Checker Helper
    def check_answer_correct(pred, gt):
        """Check if prediction matches ground truth"""
        if not isinstance(pred, str):
            pred = str(pred)
        if not isinstance(gt, str):
            gt = str(gt)
            
        pred = pred.strip()
        gt = gt.strip()
        
        # A. Handle Yes/No/Numbers (Case insensitive)
        if gt.lower() in ["yes", "no"] or gt.isdigit():
            pred_norm = re.sub(r'[^\w\s]', '', pred).lower()
            gt_norm = re.sub(r'[^\w\s]', '', gt).lower()
            pred_words = pred_norm.split()
            if pred_words and pred_words[0] == gt_norm:
                return True
                
        # B. Handle MCQ (Single Letter A-F)
        elif len(gt) == 1 and gt.upper() in "ABCDEF":
            gt_char = gt.upper()
            
            check_contents = [pred]
            lines = pred.strip().split('\n')
            if len(lines) > 1:
                check_contents.append(lines[-1])
            
            for content in check_contents:
                if not content: continue
                
                # Check for "(A)" at start
                match = re.match(r'^\s*\(([A-F])\)', content, re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    return True

                # Check for "A.", "A)", "A " at start
                match = re.match(r'^\s*([A-F])[\.\)\s]', content, re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    return True
                    
                # Check for exact match "A"
                if content.strip().upper() == gt_char:
                    return True
                
                # Check for "The answer/option is A"
                if any(k in content.lower() for k in ["answer", "option", "choice"]):
                    match = re.search(r'(?:answer|option|choice)(?: is|:| is:)?\s*\(?([A-F])\)?', content, re.IGNORECASE)
                    if match and match.group(1).upper() == gt_char:
                        return True
                
                # Check for letter at the very end (supporting "at the end" Hint)
                match = re.search(r'(?:^|\s|[^a-zA-Z0-9])([A-F])[\.\)]?$', content.strip(), re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    return True
            
            # Fallback: starts with letter
            if len(pred) > 0 and pred[0].upper() == gt_char:
                return True
                
        # C. Fallback exact match
        elif pred.lower() == gt.lower():
            return True
            
        return False

    # 3. ID Parsing Helper
    def get_base_id(qid):
        try:
            qid_int = int(qid)
        except (ValueError, TypeError):
            return qid

        if qid_int >= 3000000:
            return qid_int - 3000000
        elif qid_int >= 2000000:
            return qid_int - 2000000
        elif qid_int >= 1000000:
            return qid_int - 1000000
        else:
            return qid_int

    # 4. Group results by base_id
    question_groups = defaultdict(list)
    for r in results:
        qid = r.get('id', r.get('index', None))
        if qid is None:
            continue
            
        base_id = get_base_id(qid)
        question_groups[base_id].append(r)
    
    # 5. Calculate Accuracy
    correct_groups = 0
    total_groups = len(question_groups)
    total_samples = 0
    correct_samples = 0
    
    for base_id, samples in question_groups.items():
        group_correct = True
        for sample in samples:
            pred = sample.get('prediction', '')
            gt = sample.get('gt_answer', sample.get('answer', ''))
            
            if check_answer_correct(pred, gt):
                correct_samples += 1
            else:
                group_correct = False
            total_samples += 1
        
        if group_correct:
            correct_groups += 1

    # 6. Compute Metrics
    acc = correct_groups / total_groups if total_groups > 0 else 0.0
    sample_acc = correct_samples / total_samples if total_samples > 0 else 0.0

    print("=" * 40)
    print(f"MMBench Evaluation Results")
    print("=" * 40)
    print(f"Total Unique Questions (Groups): {total_groups}")
    print(f"Correct Groups (All Pass):       {correct_groups}")
    print(f"Circular Accuracy:               {acc:.4f}")
    print("-" * 40)
    print(f"Total Individual Samples:        {total_samples}")
    print(f"Correct Samples:                 {correct_samples}")
    print(f"Sample Accuracy:                 {sample_acc:.4f}")
    print("=" * 40)

    # 7. Save Results
    if output_dir is None:
        if results_file:
            output_dir = os.path.dirname(results_file)
        else:
            output_dir = "./results"
            
    os.makedirs(output_dir, exist_ok=True)
    acc_file = os.path.join(output_dir, "accuracy.json")
    
    with open(acc_file, 'w', encoding='utf-8') as f:
        json.dump({
            "accuracy": acc,
            "sample_accuracy": sample_acc,
            "correct_questions": correct_groups, 
            "total_questions": total_groups,
            "total_samples": total_samples
        }, f, indent=2)
    print(f"Detailed metrics saved to {acc_file}")


def eval_mmstar(results_file=None, results=None, output_dir=None):
    # Load results
    if results is None:
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

    # Calculate Accuracy
    correct = 0
    total = 0
    for r in results:
        pred = r['prediction'].strip()
        gt = r['gt_answer'].strip()
        is_correct = False
        
        # 1. Handle Yes/No/Numbers
        if gt.lower() in ["yes", "no"] or gt.isdigit():
            pred_norm = re.sub(r'[^\w\s]', '', pred).lower()
            gt_norm = re.sub(r'[^\w\s]', '', gt).lower()
            pred_words = pred_norm.split()
            if pred_words and pred_words[0] == gt_norm:
                is_correct = True
        
        # 2. Handle MCQ (Single Letter A-F)
        elif len(gt) == 1 and gt.upper() in "ABCDEF":
            gt_char = gt.upper()
            
            check_contents = [pred]
            lines = pred.strip().split('\n')
            if len(lines) > 1:
                check_contents.append(lines[-1])
            
            found_match = False
            for content in check_contents:
                if not content: continue
                
                # Check for "(A)" at start
                match = re.match(r'^\s*\(([A-F])\)', content, re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    is_correct = True
                    found_match = True
                    break

                # Check for "A.", "A)", "A " at start
                match = re.match(r'^\s*([A-F])[\.\)\s]', content, re.IGNORECASE)
                if match:
                    if match.group(1).upper() == gt_char:
                        is_correct = True
                        found_match = True
                        break
                        
                # Check for exact match "A"
                if content.strip().upper() == gt_char:
                    is_correct = True
                    found_match = True
                    break
                
                # Check for "The answer/option is A"
                if any(k in content.lower() for k in ["answer", "option", "choice"]):
                    match = re.search(r'(?:answer|option|choice)(?: is|:| is:)?\s*\(?([A-F])\)?', content, re.IGNORECASE)
                    if match and match.group(1).upper() == gt_char:
                        is_correct = True
                        found_match = True
                        break

                # Check for letter at the very end
                match = re.search(r'(?:^|\s|[^a-zA-Z0-9])([A-F])[\.\)]?$', content.strip(), re.IGNORECASE)
                if match and match.group(1).upper() == gt_char:
                    is_correct = True
                    found_match = True
                    break
            
            # Fallback
            if not found_match and len(pred) > 0 and pred[0].upper() == gt_char:
                is_correct = True
                
        # 3. Fallback exact match
        elif pred.lower() == gt.lower():
            is_correct = True
            
        if is_correct:
            correct += 1
        total += 1
        
    acc = correct / total if total > 0 else 0
    print(f"\nAccuracy: {acc:.4f} ({correct}/{total})")

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    os.makedirs(output_dir, exist_ok=True)
    acc_file = os.path.join(output_dir, "accuracy.json")
    with open(acc_file, 'w') as f:
        json.dump({"accuracy": acc, "correct": correct, "total": total}, f, indent=2)
    print(f"Results saved to {output_dir}")


def eval_mme(results_file=None, results=None, output_dir=None):

    eval_type_dict = {
            "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
            "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
        }

    class calculate_metrics:
        def divide_chunks(self, l, n=2):
            # looping till length l
            for i in range(0, len(l), n): 
                yield l[i:i + n]
            
            return 

        def parse_pred_ans(self, pred_ans):
            pred_label = None
            if pred_ans in ["yes", "no"]:
                pred_label = pred_ans
            else:
                prefix_pred_ans = pred_ans[:4]

                if "yes" in prefix_pred_ans:
                    pred_label = "yes"
                elif "no" in prefix_pred_ans:
                    pred_label = "no"
                else:
                    pred_label = "other"

            return pred_label


        def compute_metric(self, gts, preds):
            assert len(gts) == len(preds)

            label_map = {
                "yes": 1,
                "no": 0,
                "other": -1,
            }
            
            gts = [label_map[x] for x in gts]
            preds = [label_map[x] for x in preds]

            acc = accuracy_score(gts, preds) 

            clean_gts = []
            clean_preds = []
            other_num = 0 
            for gt, pred in zip(gts, preds):
                if pred == -1:
                    other_num += 1
                    continue
                clean_gts.append(gt)
                clean_preds.append(pred)
            

            conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
            precision = precision_score(clean_gts, clean_preds, average='binary')
            recall = recall_score(clean_gts, clean_preds, average='binary')
            tp, fn = conf_mat[0]
            fp, tn = conf_mat[1]

            metric_dict = dict()
            metric_dict = {
                "TP": tp,
                "FN": fn,
                "TN": tn,
                "FP": fp,
                "precision": precision,
                "recall": recall,
                "other_num": other_num,
                "acc": acc,
            }

            return metric_dict


        def process_result(self, results_dir):

            model_score_dict = dict()
            for eval_type, task_name_list in eval_type_dict.items():
                print("===========", eval_type, "===========")
            
                scores = 0
                task_score_dict = dict()

                for task_name in task_name_list:

                    task_txt = os.path.join(results_dir, task_name + ".txt")
                    lines = open(task_txt, 'r').readlines()
                    chunk_lines = list(self.divide_chunks(lines)) # one image corresponds to two questions
                    
                    img_num = len(chunk_lines)
                    task_other_ans_num = 0
                    task_score = 0
                    acc_plus_correct_num = 0
                    gts = []
                    preds = []

                    for img_items in chunk_lines:
                        assert len(img_items) == 2
                        img_correct_num = 0

                        for img_item in img_items:
                            img_name, question, gt_ans, pred_ans = img_item.split("\t")

                            gt_ans = gt_ans.lower()
                            pred_ans = pred_ans.lower()

                            assert gt_ans in ["yes", "no"] # gt can only be yes or no.

                            pred_ans = self.parse_pred_ans(pred_ans)
                            assert pred_ans in ["yes", "no", "other"]

                            gts.append(gt_ans)
                            preds.append(pred_ans)
                            
                            if gt_ans == pred_ans:
                                img_correct_num += 1
                            
                            if pred_ans not in ["yes", "no"]:
                                task_other_ans_num += 1

                        if img_correct_num == 2:
                            acc_plus_correct_num += 1

                    # cal TP precision acc, etc.
                    metric_dict = self.compute_metric(gts, preds)
                    acc_plus = acc_plus_correct_num / img_num
                    metric_dict["acc_plus"] = acc_plus
                    
                    
                    for k, v in metric_dict.items():
                        if k in ["acc", "acc_plus"]:
                            task_score += v*100
                    
                    task_score_dict[task_name] = task_score
                    
                    scores += task_score

                print("total score:", scores, "\n")
                for task_name, score in task_score_dict.items():
                    print("\t", task_name, " score:", score)
                print("\n")
            
            return 

    # Load results
    if results is None:
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

    cal = calculate_metrics()
    cal.process_result(results_file)

