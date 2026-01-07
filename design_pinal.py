import pickle
import random
import time
import os
import json
import subprocess
import pandas as pd

# ================= 配置区域 =================

# 输入文件 (直接使用你处理好的文件)
DATA_FILE = 'test_data_desc.pkl'

# 输出设置
OUTPUT_DIR = "curl_generated_proteins"
RESULT_CSV = "final_design_result.csv"

# API 地址
BASE_URL = "http://www.denovo-pinal.com"
API_POST = f"{BASE_URL}/gradio_api/call/design_and_protrek_score"

# ================= Bash Curl 包装函数 (保持不变) =================

def curl_post_task(prompt, num=5):
    """发送 POST 请求获取任务 ID"""
    data = json.dumps({"data": [prompt, num]})
    cmd = [
        "curl", "-s", "-X", "POST", API_POST,
        "-H", "Content-Type: application/json",
        "-d", data
    ]
    try:
        # 增加 check=False 防止非 0 退出码导致崩溃，手动处理
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Curl error: {result.stderr}")
            return None
            
        # 尝试解析 JSON
        try:
            response_json = json.loads(result.stdout)
            return response_json.get("event_id")
        except json.JSONDecodeError:
            # 有时候 API 可能返回非 JSON 的 500 错误页
            print(f"Invalid JSON response: {result.stdout[:100]}...")
            return None
            
    except Exception as e:
        print(f"System error in curl POST: {e}")
        return None

def curl_get_stream_result(event_id):
    """监听结果流并提取文件 URL"""
    url = f"{API_POST}/{event_id}"
    cmd = ["curl", "-N", "-s", url]
    
    try:
        # 设置超时 120s
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout
        
        lines = output.split('\n')
        is_complete = False
        final_data_str = None
        
        # 解析 Server-Sent Events
        for line in lines:
            if line.strip() == "event: complete":
                is_complete = True
            if is_complete and line.startswith("data: "):
                final_data_str = line[len("data: "):]
                break
        
        if final_data_str:
            return json.loads(final_data_str)
        return None
        
    except subprocess.TimeoutExpired:
        print("Error: Curl request timed out.")
        return None
    except Exception as e:
        print(f"Error in curl GET stream: {e}")
        return None

def curl_download_file(file_url, local_path):
    """下载结果文件"""
    cmd = ["curl", "-s", "-o", local_path, file_url]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def process_result_tsv(tsv_path, top_n=3):
    """解析 TSV 筛选最佳序列"""
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        df.columns = [c.strip() for c in df.columns]
        
        # 模糊匹配列名
        score_col = next((c for c in df.columns if "Protrek Score" in c), None)
        seq_col = next((c for c in df.columns if "Protein Sequence" in c), None)
        
        if not score_col or not seq_col:
            return None
            
        candidates = df.head(top_n).copy()
        if candidates.empty: return None
            
        best_idx = candidates[score_col].idxmax()
        best_row = candidates.loc[best_idx]
        
        return {
            'sequence': best_row[seq_col],
            'score': float(best_row[score_col])
        }
    except Exception as e:
        print(f"TSV parsing error: {e}")
        return None

# ================= 主逻辑 (已更新) =================

def main():
    # 1. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 2. 加载数据 (直接读取 test_data_desc.pkl)
    print(f"Loading {DATA_FILE}...")
    with open(DATA_FILE, 'rb') as f:
        data_list = pickle.load(f)
    # data_list = [{'id': 'Q60888', 'conditions': ['GO:0004930', 'GO:0004984'], 'conditions_desc': ['G protein-coupled receptor activity', 'olfactory receptor activity'], 'conditions_prompt': 'Generate a protein sequence that functions as: G protein-coupled receptor activity, olfactory receptor activity.'}]

    final_results = []
    print(f"Starting process for {len(data_list)} proteins...")
    
    # 3. 遍历处理
    for i, item in enumerate(data_list):
        uid = item.get('id', f"Unknown_{i}")
        base_prompt = item.get('conditions_prompt', "")
        
        # --- 关键步骤：追加随机长度指令 ---
        target_len = random.randint(200, 400)
        final_prompt = f"{base_prompt} The sequence length is approximately {target_len}."
        
        print(f"\n[{i+1}/{len(data_list)}] ID: {uid}")
        print(f"  Prompt: {final_prompt[:80]}...") # 只打印前80个字符避免刷屏
        
        # --- Step A: 提交任务 ---
        event_id = curl_post_task(final_prompt, num=5)
        if not event_id:
            print("  [Skip] Failed to get Event ID.")
            time.sleep(1)
            continue
            
        # --- Step B: 等待结果 ---
        api_data = curl_get_stream_result(event_id)
        if not api_data:
            print("  [Skip] Failed to retrieve stream data.")
            time.sleep(1)
            continue
            
        # print(f"api_data: {api_data}")
        # exit()
        
            
# --- Step C: 获取下载链接 (关键修复点) ---
        file_url = None
        
        try:
            # api_data[1] 是包含 file info 的字典
            if len(api_data) > 1 and isinstance(api_data[1], dict):
                comp_data = api_data[1]
                
                # 结构是 {'value': {'url': '...', ...}, ...}
                if 'value' in comp_data and isinstance(comp_data['value'], dict):
                    file_url = comp_data['value'].get('url')
                # 兼容性备选：如果直接在顶层
                elif 'url' in comp_data:
                    file_url = comp_data['url']
        except Exception as e:
            print(f"  [Error] Parsing JSON structure failed: {e}")
        
        if not file_url:
            print(f"  [Skip] Could not find file URL in response.")
            # 调试用：打印一下结构方便排查
            # print(f"DEBUG: {api_data[1]}") 
            continue
            
        # --- Step D: 下载并筛选 ---
        local_filename = f"{uid}_{int(time.time())}.tsv"
        local_path = os.path.join(OUTPUT_DIR, local_filename)
        
        if curl_download_file(file_url, local_path):
            best_res = process_result_tsv(local_path, top_n=3)
            
            if best_res:
                print(f"  >>> Success! Best Score: {best_res['score']:.4f}")
                final_results.append({
                    'id': uid,
                    'prompt': final_prompt,
                    'target_length': target_len,
                    'best_sequence': best_res['sequence'],
                    'protrek_score': best_res['score']
                })
            else:
                print("  [Warning] TSV parsing failed (maybe empty file).")
        
        time.sleep(2)

    if final_results:
        df = pd.DataFrame(final_results)
        df.to_csv(RESULT_CSV, index=False)
        print(f"\nSaved {len(final_results)} results to {RESULT_CSV}")

if __name__ == "__main__":
    main()