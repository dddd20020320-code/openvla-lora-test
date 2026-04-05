import os
import sys
import xml.etree.ElementTree as ET

import h5py
import imageio
import numba
import numpy as np


ROOT_DIR = "/home/zz/openvla/LIBERO"
if sys.path[0] != ROOT_DIR:
    sys.path.insert(0, ROOT_DIR)

os.environ["LIBERO_DATASET_PATH"] = os.path.join(ROOT_DIR, "datasets")
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

_original_numba_jit = numba.jit


def _jit_without_cache(*args, **kwargs):
    kwargs.pop("cache", None)
    return _original_numba_jit(*args, **kwargs)


numba.jit = _jit_without_cache

from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.mjcf_utils import find_elements
import robosuite


BDDL_PATH = os.path.join(
    ROOT_DIR,
    "libero/libero/bddl_files/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket.bddl",
)
DEMO_PATH = os.path.join(
    ROOT_DIR,
    "datasets/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
)
VIDEO_PATH = os.path.join(ROOT_DIR, "vla_expert_grasp.mp4")


def frame_from_obs(obs):
    return np.flipud((obs["agentview_image"] * 255).astype(np.uint8))


def load_demo(demo_path):
    with h5py.File(demo_path, "r") as f:
        demo_keys = sorted(list(f["data"].keys()))
        if not demo_keys:
            raise RuntimeError(f"Demo file is empty: {demo_path}")

        demo_key = demo_keys[0]
        actions = np.array(f[f"data/{demo_key}/actions"][()])
        states = np.array(f[f"data/{demo_key}/states"][()])
        model_xml = f[f"data/{demo_key}"].attrs["model_file"]

    return demo_key, actions, states, model_xml


def rewrite_demo_xml_paths(model_xml):
    robosuite_root = os.path.dirname(robosuite.__file__)
    libero_assets_root = os.path.join(ROOT_DIR, "libero", "libero", "assets")

    tree = ET.fromstring(model_xml)
    asset = tree.find("asset")
    all_elements = asset.findall("mesh") + asset.findall("texture")

    for elem in all_elements:
        old_path = elem.get("file")
        if not old_path:
            continue

        parts = old_path.split("/")
        if "robosuite" in parts:
            idx = max(i for i, value in enumerate(parts) if value == "robosuite")
            new_path = os.path.join(robosuite_root, *parts[idx + 1 :])
            elem.set("file", new_path)
            continue

        if "assets" in parts and ("chiliocosm" in parts or "libero" in parts):
            idx = max(i for i, value in enumerate(parts) if value == "assets")
            new_path = os.path.normpath(os.path.join(libero_assets_root, *parts[idx + 1 :]))
            elem.set("file", new_path)

    for camera in find_elements(root=tree, tags="camera", return_first=False):
        camera_name = camera.get("name")
        if camera_name:
            camera.set("name", camera_name)

    return ET.tostring(tree, encoding="utf8").decode("utf8")


def main():
    os.chdir(ROOT_DIR)
    demo_key, actions, states, model_xml = load_demo(DEMO_PATH)
    model_xml = rewrite_demo_xml_paths(model_xml)

    env = OffScreenRenderEnv(
        bddl_file_name=BDDL_PATH,
        camera_heights=224,
        camera_widths=224,
        horizon=max(5000, len(actions) + 200),
        ignore_done=True,
    )
    video_writer = imageio.get_writer(VIDEO_PATH, fps=20)

    try:
        print("\n[专家回放] 开始重放 orange juice -> basket 的 expert demo")
        print(f"[数据集] demo={demo_key} | actions={len(actions)} | video={VIDEO_PATH}")
        print("-" * 60)

        env.reset()
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        env._post_process()
        env._update_observables(force=True)
        obs = env.env._get_observations()

        video_writer.append_data(frame_from_obs(obs))

        success_step = None
        for step, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            video_writer.append_data(frame_from_obs(obs))

            if step % 25 == 0:
                orange_pos = np.round(obs["orange_juice_1_pos"], 3)
                basket_pos = np.round(obs["basket_1_pos"], 3)
                eef_pos = np.round(obs["robot0_eef_pos"], 3)
                print(
                    f"step={step:03d} | eef={eef_pos} | orange={orange_pos} | basket={basket_pos}"
                )

            if env.check_success():
                success_step = step
                print(f"[成功] 在 step {step} 达成目标。")
                break

        if success_step is None:
            success = env.check_success()
            print(f"[结果] success={success} | 已回放完整条 expert 轨迹")
        else:
            success = True

        print("-" * 60)
        print(f"[视频生成] success={success} | 保存路径: {VIDEO_PATH}")

    finally:
        video_writer.close()
        env.close()


if __name__ == "__main__":
    main()
