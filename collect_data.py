import os, sys, h5py, json
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import robosuite

# 1. 路径与环境配置
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBERO_REPO_PATH = os.path.join(ROOT_DIR, "LIBERO")

if LIBERO_REPO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_REPO_PATH)

os.environ["LIBERO_DATASET_PATH"] = os.path.join(LIBERO_REPO_PATH, "datasets")
os.environ["MUJOCO_GL"] = "egl"

from libero.libero.envs import OffScreenRenderEnv

# 2. 路径重写函数：干掉那些 "yifengz" 的幽灵路径
def rewrite_xml_paths(model_xml):
    # 获取你本地 robosuite 和 libero 资源的真实路径
    robosuite_root = os.path.dirname(robosuite.__file__)
    libero_assets_root = os.path.join(LIBERO_REPO_PATH, "libero/libero/assets")

    tree = ET.fromstring(model_xml)
    asset = tree.find("asset")
    if asset is None: return model_xml
    
    # 遍历所有 mesh 和 texture 节点
    for elem in asset.findall("mesh") + asset.findall("texture"):
        old_path = elem.get("file")
        if not old_path: continue

        # 逻辑：如果路径里包含 robosuite，就换成你本地的 robosuite 路径
        parts = old_path.split("/")
        if "robosuite" in parts:
            idx = max(i for i, v in enumerate(parts) if v == "robosuite")
            new_path = os.path.join(robosuite_root, *parts[idx + 1 :])
            elem.set("file", new_path)
        # 如果路径里包含 assets/libero 或 chiliocosm，换成你本地的 libero 资源路径
        elif "assets" in parts:
            idx = max(i for i, v in enumerate(parts) if v == "assets")
            new_path = os.path.normpath(os.path.join(libero_assets_root, *parts[idx + 1 :]))
            elem.set("file", new_path)

    return ET.tostring(tree, encoding="utf8").decode("utf8")

# 3. 采集主函数
OUT_DIR = os.path.join(ROOT_DIR, "dataset_v1")
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

DEMO_PATH = os.path.join(LIBERO_REPO_PATH, "datasets/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5")
BDDL_PATH = os.path.join(LIBERO_REPO_PATH, "libero/libero/bddl_files/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket.bddl")

def collect():
    print(f"[准备] 正在启动剧组，使用任务: {os.path.basename(BDDL_PATH)}")
    env = OffScreenRenderEnv(bddl_file_name=BDDL_PATH, camera_heights=224, camera_widths=224)
    all_samples = []
    
    with h5py.File(DEMO_PATH, "r") as f:
        demo_keys = list(f["data"].keys())[:10] # 采集前10条
        
        for d_idx, d_key in enumerate(demo_keys):
            print(f"正在录制轨迹: {d_key} (第 {d_idx+1}/10 条)...")
            demo = f[f"data/{d_key}"]
            actions = demo["actions"][()]
            states = demo["states"][()]
            
            # --- 核心修复：重写 XML 路径后再重置环境 ---
            model_xml = demo.attrs["model_file"]
            clean_xml = rewrite_xml_paths(model_xml)
            
            env.reset()
            env.reset_from_xml_string(clean_xml)
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()
            
            env._update_observables(force=True)
            obs = env.env._get_observations()

            for step in range(len(actions)):
                img = np.flipud((obs["agentview_image"] * 255).astype(np.uint8))
                img_name = f"demo_{d_idx}_step_{step}.jpg"
                Image.fromarray(img).save(os.path.join(IMG_DIR, img_name))
                
                all_samples.append({
                    "image": img_name,
                    "instruction": "pick up the orange juice and place it in the basket",
                    "action": actions[step].tolist(),
                })
                
                obs, _, _, _ = env.step(actions[step])

    with open(os.path.join(OUT_DIR, "dataset.jsonl"), "w") as jf:
        for s in all_samples:
            jf.write(json.dumps(s) + "\n")
            
    env.close()
    print(f"\n[采集成功] 样本总数: {len(all_samples)}")

if __name__ == "__main__":
    collect()