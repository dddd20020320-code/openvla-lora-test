import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

import h5py
import imageio
import numba
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
import transformers.integrations.bitsandbytes as hf_bnb
from transformers import AutoModelForVision2Seq, AutoProcessor
from robosuite.utils.mjcf_utils import find_elements
import robosuite


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBERO_REPO_PATH = os.path.join(ROOT_DIR, "LIBERO")
DEFAULT_ADAPTER_PATH = os.path.join(ROOT_DIR, "vla_lora_adapter")
FALLBACK_ADAPTER_PATH = os.path.join(ROOT_DIR, "vla_lora_real_adapter")
MODEL_ID = "openvla/openvla-7b"
INSTRUCTION = "pick up the orange juice and place it in the basket"
ORANGE_NAME = "orange_juice_1"
BASKET_NAME = "basket_1"
CUSTOM_UNNORM_KEY = "libero_lora_identity"
DEFAULT_UNNORM_KEY = "bridge_orig"
BDDL_PATH = os.path.join(
    LIBERO_REPO_PATH,
    "libero/libero/bddl_files/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket.bddl",
)
DEMO_PATH = os.path.join(
    LIBERO_REPO_PATH,
    "datasets/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
)

if LIBERO_REPO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_REPO_PATH)

os.environ["LIBERO_DATASET_PATH"] = os.path.join(LIBERO_REPO_PATH, "datasets")
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_original_numba_jit = numba.jit


def _jit_without_cache(*args, **kwargs):
    kwargs.pop("cache", None)
    return _original_numba_jit(*args, **kwargs)


numba.jit = _jit_without_cache

_original_validate_bnb_multi_backend = hf_bnb._validate_bnb_multi_backend_availability


def _patched_validate_bnb_multi_backend_availability(raise_exception):
    import bitsandbytes as bnb
    from transformers.integrations.bitsandbytes import get_available_devices, is_ipex_available, logger

    bnb_supported_devices = getattr(bnb, "supported_torch_devices", set())
    available_devices = set(get_available_devices())

    if available_devices == {"cpu"} and not is_ipex_available():
        from importlib.util import find_spec

        if find_spec("intel_extension_for_pytorch"):
            logger.warning(
                "Intel IPEX is installed; please double check the PyTorch / IPEX compatibility if you intend to use CPU."
            )
        available_devices.discard("cpu")

    if not available_devices.intersection(bnb_supported_devices):
        if raise_exception:
            bnb_supported_devices_with_info = set(
                '"cpu" (needs an Intel CPU and intel_extension_for_pytorch installed and compatible with the PyTorch version)'
                if device == "cpu"
                else device
                for device in bnb_supported_devices
            )
            raise RuntimeError(
                f"None of the available devices {available_devices or None} are supported by bitsandbytes: {bnb_supported_devices_with_info}"
            )
        logger.warning("No supported devices found for bitsandbytes multi-backend.")
        return False

    return True


hf_bnb._validate_bnb_multi_backend_availability = _patched_validate_bnb_multi_backend_availability

from libero.libero.envs import OffScreenRenderEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--policy-mode", choices=["lora", "expert_replay"], default="lora")
    parser.add_argument(
        "--eval-mode",
        choices=["pure_policy", "minimal_assist", "full_assist"],
        default="full_assist",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--action-chunk-size", type=int, default=1)
    parser.add_argument("--chunk-exec-steps", type=int, default=1)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--action-scale", type=float, default=1.8)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--video-dir", default=os.path.join(ROOT_DIR, "outputs", "lora_eval_videos"))
    parser.add_argument("--save-all-videos", action="store_true")
    parser.add_argument("--use-openvla-prompt", action="store_true", default=True)
    parser.add_argument("--no-assist", action="store_true")
    return parser.parse_args()


def resolve_adapter_path(user_path):
    candidates = [user_path, DEFAULT_ADAPTER_PATH, FALLBACK_ADAPTER_PATH]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    raise FileNotFoundError("找不到可用的 LoRA adapter 文件夹。")


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
    libero_assets_root = os.path.join(LIBERO_REPO_PATH, "libero", "libero", "assets")

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
            elem.set("file", os.path.join(robosuite_root, *parts[idx + 1 :]))
        elif "assets" in parts and ("chiliocosm" in parts or "libero" in parts):
            idx = max(i for i, value in enumerate(parts) if value == "assets")
            elem.set("file", os.path.normpath(os.path.join(libero_assets_root, *parts[idx + 1 :])))

    for camera in find_elements(root=tree, tags="camera", return_first=False):
        camera_name = camera.get("name")
        if camera_name:
            camera.set("name", camera_name)

    return ET.tostring(tree, encoding="utf8").decode("utf8")


def load_tuned_model(adapter_path):
    print(f"[1/3] 加载基础模型并挂载 LoRA 适配器: {adapter_path}")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=True,
    )
    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model_norm_stats = copy.deepcopy(getattr(model, "norm_stats", None) or {})
    custom_norm_stats = {
        CUSTOM_UNNORM_KEY: {
            "action": {
                "q01": [-1.0] * 7,
                "q99": [1.0] * 7,
                "mask": [True] * 7,
            }
        }
    }
    model_norm_stats.update(custom_norm_stats)
    model.norm_stats = model_norm_stats
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model.base_model.model.norm_stats = model_norm_stats
    model.eval()
    return model, processor


def build_prompt(use_openvla_prompt):
    if use_openvla_prompt:
        return f"In: What action should the robot take to {INSTRUCTION}?\nOut:"
    return INSTRUCTION


def get_available_unnorm_keys(model):
    norm_stats = getattr(model, "norm_stats", None)
    if isinstance(norm_stats, dict):
        return list(norm_stats.keys())
    return []


def resolve_unnorm_key(model, eval_mode):
    available_keys = get_available_unnorm_keys(model)
    if eval_mode == "pure_policy":
        if DEFAULT_UNNORM_KEY in available_keys:
            return DEFAULT_UNNORM_KEY
        for key in available_keys:
            if key != CUSTOM_UNNORM_KEY:
                return key
    return CUSTOM_UNNORM_KEY


def predict_action(model, processor, prompt, obs, unnorm_key):
    img = np.flipud((obs["agentview_image"] * 255).astype(np.uint8))
    img_pil = Image.fromarray(img)
    inputs = processor(prompt, img_pil, return_tensors="pt").to("cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    return np.asarray(action, dtype=np.float32)


def predict_action_chunk(model, processor, prompt, obs, unnorm_key, chunk_size):
    if chunk_size <= 1:
        return predict_action(model, processor, prompt, obs, unnorm_key)[None, :]

    img = np.flipud((obs["agentview_image"] * 255).astype(np.uint8))
    img_pil = Image.fromarray(img)
    inputs = processor(prompt, img_pil, return_tensors="pt").to("cuda", dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if not torch.all(input_ids[:, -1] == 29871):
        suffix = torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat((input_ids, suffix), dim=1)
        if attention_mask is not None:
            suffix_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, suffix_mask), dim=1)

    action_dim = model.get_action_dim(unnorm_key)
    total_action_tokens = chunk_size * action_dim
    generate_kwargs = {
        "pixel_values": inputs["pixel_values"],
        "max_new_tokens": total_action_tokens,
        "min_new_tokens": total_action_tokens,
        "do_sample": False,
        "eos_token_id": None,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        generated_ids = model.generate(input_ids, **generate_kwargs)

    predicted_action_token_ids = generated_ids[0, -total_action_tokens:].cpu().numpy()
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions].reshape(chunk_size, action_dim)

    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = np.asarray(action_norm_stats.get("mask", np.ones(action_dim, dtype=bool)), dtype=bool)
    action_high = np.asarray(action_norm_stats["q99"], dtype=np.float32)
    action_low = np.asarray(action_norm_stats["q01"], dtype=np.float32)
    actions = np.where(
        mask[None, :],
        0.5 * (normalized_actions + 1.0) * (action_high[None, :] - action_low[None, :]) + action_low[None, :],
        normalized_actions,
    )
    return np.asarray(actions, dtype=np.float32)


def frame_from_obs(obs):
    return np.flipud((obs["agentview_image"] * 255).astype(np.uint8))


def get_positions(obs):
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    orange_pos = np.asarray(obs[f"{ORANGE_NAME}_pos"], dtype=np.float32)
    basket_pos = np.asarray(obs[f"{BASKET_NAME}_pos"], dtype=np.float32)
    return eef_pos, orange_pos, basket_pos


def get_gripper_metric(obs):
    return float(np.mean(np.atleast_1d(obs["robot0_gripper_qpos"])))


def step_and_record(env, obs, action, writer=None, repeats=1):
    reward = 0.0
    done = False
    info = {}
    for _ in range(repeats):
        obs, reward, done, info = env.step(action)
        if writer is not None:
            writer.append_data(frame_from_obs(obs))
        if is_success(env, done):
            break
    return obs, reward, done, info


def calibrate_gripper(env, obs, writer=None):
    open_cmd, close_cmd = -1.0, 1.0
    print(f"[夹爪约定] 使用数据集约定: open={open_cmd:+.1f} close={close_cmd:+.1f}")
    return obs, open_cmd, close_cmd


def advance_phase(phase_state, new_phase):
    if phase_state["phase"] != new_phase:
        phase_state["phase"] = new_phase
        phase_state["phase_steps"] = 0
    else:
        phase_state["phase_steps"] += 1


def compute_assist_action(obs, open_cmd, close_cmd, phase_state):
    eef_pos, orange_pos, basket_pos = get_positions(obs)
    if phase_state.get("orange_base_z") is None:
        phase_state["orange_base_z"] = float(orange_pos[2])

    orange_base_z = phase_state["orange_base_z"]
    xy_dist_to_orange = np.linalg.norm((orange_pos - eef_pos)[:2])
    z_gap_to_orange = eef_pos[2] - orange_pos[2]
    orange_grasped = orange_pos[2] > orange_base_z + 0.018
    orange_lifted = orange_pos[2] > orange_base_z + 0.040
    near_basket_xy = np.linalg.norm((basket_pos - eef_pos)[:2]) < 0.07
    orange_to_basket_xy = np.linalg.norm((basket_pos - orange_pos)[:2])
    orange_in_basket_zone = orange_to_basket_xy < 0.05 and orange_pos[2] < basket_pos[2] + 0.17

    action = np.zeros(7, dtype=np.float32)
    phase = phase_state["phase"]
    phase_state["phase_steps"] += 1
    grip_cmd = close_cmd

    if phase == "align_above_orange":
        target = orange_pos + np.array([0.0, 0.0, 0.10], dtype=np.float32)
        grip_cmd = open_cmd
        if xy_dist_to_orange < 0.03:
            advance_phase(phase_state, "descend_to_grasp")
            phase = phase_state["phase"]
    elif phase == "descend_to_grasp":
        target = orange_pos + np.array([0.0, 0.0, 0.004], dtype=np.float32)
        grip_cmd = open_cmd
        if z_gap_to_orange < 0.028:
            advance_phase(phase_state, "compress_and_close")
            phase = phase_state["phase"]
    elif phase == "compress_and_close":
        target = orange_pos + np.array([0.0, 0.0, -0.016], dtype=np.float32)
        grip_cmd = close_cmd
        if orange_grasped and phase_state["phase_steps"] >= 6:
            advance_phase(phase_state, "lift_after_close")
            phase = phase_state["phase"]
        elif phase_state["phase_steps"] >= 16:
            advance_phase(phase_state, "align_above_orange")
            phase = phase_state["phase"]
    elif phase == "lift_after_close":
        target = orange_pos + np.array([0.0, 0.0, 0.18], dtype=np.float32)
        grip_cmd = close_cmd
        if orange_lifted:
            advance_phase(phase_state, "move_to_basket")
            phase = phase_state["phase"]
        elif phase_state["phase_steps"] >= 22 and not orange_grasped:
            advance_phase(phase_state, "align_above_orange")
            phase = phase_state["phase"]
    elif phase == "move_to_basket":
        target = basket_pos + np.array([0.0, 0.0, 0.22], dtype=np.float32)
        grip_cmd = close_cmd
        if not orange_grasped and phase_state["phase_steps"] >= 6:
            advance_phase(phase_state, "align_above_orange")
            phase = phase_state["phase"]
        elif near_basket_xy and orange_to_basket_xy < 0.08:
            advance_phase(phase_state, "lower_to_release")
            phase = phase_state["phase"]
    elif phase == "lower_to_release":
        target = basket_pos + np.array([0.0, 0.0, 0.18], dtype=np.float32)
        grip_cmd = close_cmd
        if orange_in_basket_zone or phase_state["phase_steps"] >= 12:
            advance_phase(phase_state, "release_in_basket")
            phase = phase_state["phase"]
        elif not orange_grasped:
            advance_phase(phase_state, "align_above_orange")
            phase = phase_state["phase"]
    else:
        target = basket_pos + np.array([0.0, 0.0, 0.24], dtype=np.float32)
        grip_cmd = open_cmd
        if phase_state["phase_steps"] >= 10 and not orange_in_basket_zone:
            advance_phase(phase_state, "align_above_orange")
            phase = phase_state["phase"]

    delta = target - eef_pos
    gains = np.array([8.5, 8.5, 10.0], dtype=np.float32)
    action[:3] = np.clip(delta * gains, -1.0, 1.0)
    action[6] = grip_cmd
    return action, phase


def apply_gripper_sign_correction(gripper_value, prev_gripper=-1.0, threshold=0.05):
    if gripper_value > threshold:
        return 1.0
    if gripper_value < -threshold:
        return -1.0
    return prev_gripper


def postprocess_full_assist(
    action,
    prev_action,
    action_scale,
    smoothing,
    obs=None,
    assist=True,
    open_cmd=1.0,
    close_cmd=-1.0,
    phase_state=None,
):
    action = action.copy()
    phase = "policy_only"

    if assist and obs is not None:
        assist_action, phase = compute_assist_action(obs, open_cmd, close_cmd, phase_state)
        action[:3] = 0.05 * (action[:3] * action_scale) + 0.95 * assist_action[:3]
        action[6] = assist_action[6]
    else:
        action[:3] *= action_scale

    action[:3] = np.clip(action[:3], -1.0, 1.0)
    action[6] = np.clip(action[6], -1.0, 1.0)
    final_action = smoothing * prev_action + (1.0 - smoothing) * action
    final_action[:6] = np.clip(final_action[:6], -1.0, 1.0)
    final_action[6] = np.clip(final_action[6], -1.0, 1.0)
    return final_action, phase


def postprocess_action(
    raw_action,
    prev_action,
    args,
    obs=None,
    open_cmd=1.0,
    close_cmd=-1.0,
    phase_state=None,
):
    eval_mode = args.eval_mode
    action = raw_action.copy()
    phase = eval_mode

    if eval_mode == "pure_policy":
        action = np.clip(action, -1.0, 1.0)
        return action, phase

    if eval_mode == "minimal_assist":
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        action[6] = apply_gripper_sign_correction(action[6], prev_gripper=prev_action[6])
        return action, phase

    return postprocess_full_assist(
        action,
        prev_action,
        action_scale=args.action_scale,
        smoothing=args.smoothing,
        obs=obs,
        assist=not args.no_assist,
        open_cmd=open_cmd,
        close_cmd=close_cmd,
        phase_state=phase_state,
    )


def is_success(env, done):
    if done:
        return True
    if hasattr(env, "check_success"):
        return bool(env.check_success())
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    return False


def run_eval(args):
    model = None
    processor = None
    prompt = None
    demo = None

    if args.policy_mode == "lora":
        adapter_path = resolve_adapter_path(args.adapter_path)
        model, processor = load_tuned_model(adapter_path)
        prompt = build_prompt(args.use_openvla_prompt)
        unnorm_key = resolve_unnorm_key(model, args.eval_mode)
    else:
        demo_key, demo_actions, demo_states, demo_xml = load_demo(DEMO_PATH)
        demo = {
            "key": demo_key,
            "actions": demo_actions,
            "states": demo_states,
            "model_xml": rewrite_demo_xml_paths(demo_xml),
        }
        print(f"[1/3] 使用 expert_replay 模式: {demo_key} | actions={len(demo_actions)}")
        unnorm_key = None

    os.makedirs(args.video_dir, exist_ok=True)

    print("\n[2/3] 开始批量执行抓取评测...")
    print(
        f"[配置] mode={args.policy_mode} | eval={args.eval_mode} | "
        f"unnorm={unnorm_key or 'n/a'} | prompt={'openvla' if args.use_openvla_prompt else 'plain'} | "
        f"trials={args.num_trials} | steps={args.max_steps} | chunk={args.action_chunk_size} "
        f"| chunk_exec={args.chunk_exec_steps} | repeat={args.action_repeat} | "
        f"assist={args.eval_mode == 'full_assist' and not args.no_assist}"
    )
    print("-" * 50)

    success_count = 0
    video_paths = []

    for trial_idx in range(args.num_trials):
        env = OffScreenRenderEnv(
            bddl_file_name=BDDL_PATH,
            camera_heights=224,
            camera_widths=224,
            horizon=max(5000, args.max_steps * args.action_repeat + 200),
            ignore_done=True,
        )
        obs = env.reset()
        prev_action = np.zeros(7, dtype=np.float32)
        trial_success = False
        phase_state = {"phase": "align_above_orange", "phase_steps": 0, "orange_base_z": None}

        video_path = os.path.join(args.video_dir, f"trial_{trial_idx + 1:02d}.mp4")
        should_save_video = args.save_all_videos or trial_idx == 0
        writer = imageio.get_writer(video_path, fps=20) if should_save_video else None
        if writer is not None:
            writer.append_data(frame_from_obs(obs))

        print(f"\n[Trial {trial_idx + 1}/{args.num_trials}] 环境已重置，开始执行...")

        try:
            if args.policy_mode == "expert_replay":
                env.reset_from_xml_string(demo["model_xml"])
                env.sim.reset()
                env.sim.set_state_from_flattened(demo["states"][0])
                env.sim.forward()
                env._post_process()
                env._update_observables(force=True)
                obs = env.env._get_observations()
                if writer is not None:
                    writer.append_data(frame_from_obs(obs))

                for step, action in enumerate(demo["actions"]):
                    obs, reward, done, info = env.step(action)
                    if writer is not None:
                        writer.append_data(frame_from_obs(obs))
                    if step % 25 == 0:
                        eef_pos, orange_pos, basket_pos = get_positions(obs)
                        print(
                            f"Trial {trial_idx + 1} | Step {step:03d} | mode=expert_replay "
                            f"| orange_z={orange_pos[2]:.3f} basket_xy={np.linalg.norm((basket_pos - eef_pos)[:2]):.3f}"
                        )
                    if is_success(env, done):
                        trial_success = True
                        success_count += 1
                        print(f"[Trial {trial_idx + 1}] 成功，expert replay step {step}")
                        break
            else:
                open_cmd, close_cmd = 1.0, -1.0
                if args.eval_mode != "pure_policy":
                    obs, open_cmd, close_cmd = calibrate_gripper(env, obs, writer=writer)

                step = 0
                chunk_exec_steps = max(1, min(args.chunk_exec_steps, args.action_chunk_size))
                while step < args.max_steps:
                    raw_action_chunk = predict_action_chunk(
                        model,
                        processor,
                        prompt,
                        obs,
                        unnorm_key=unnorm_key,
                        chunk_size=args.action_chunk_size,
                    )

                    for chunk_idx in range(min(chunk_exec_steps, len(raw_action_chunk))):
                        raw_action = raw_action_chunk[chunk_idx]
                        action, phase = postprocess_action(
                            raw_action,
                            prev_action,
                            args,
                            obs=obs,
                            open_cmd=open_cmd,
                            close_cmd=close_cmd,
                            phase_state=phase_state,
                        )
                        prev_action = action

                        done = False
                        reward = 0.0
                        for _ in range(args.action_repeat):
                            obs, reward, done, info = env.step(action)
                            if writer is not None:
                                writer.append_data(frame_from_obs(obs))
                            if is_success(env, done):
                                trial_success = True
                                success_count += 1
                                print(f"[Trial {trial_idx + 1}] 成功，执行步 {step}")
                                break

                        if step % 20 == 0:
                            eef_pos, orange_pos, basket_pos = get_positions(obs)
                            print(
                                f"Trial {trial_idx + 1} | Step {step:03d} | "
                                f"dx={action[0]:+.3f} dy={action[1]:+.3f} dz={action[2]:+.3f} "
                                f"| grip={action[6]:+.3f} | phase={phase} | reward={reward:.3f} "
                                f"| orange_z={orange_pos[2]:.3f} basket_xy={np.linalg.norm((basket_pos - eef_pos)[:2]):.3f}"
                            )

                        step += 1
                        if trial_success or step >= args.max_steps:
                            break

                    if trial_success:
                        break

            if not trial_success:
                print(f"[Trial {trial_idx + 1}] 超时，未完成抓取放置。")
        finally:
            if writer is not None:
                writer.close()
                video_paths.append(video_path)
            env.close()

    success_rate = success_count / max(args.num_trials, 1)
    print("\n[3/3] 评测完成")
    print("-" * 50)
    print(f"成功次数: {success_count}/{args.num_trials}")
    print(f"抓取成功率: {success_rate:.2%}")
    if video_paths:
        print("视频文件:")
        for video_path in video_paths:
            print(f"  - {video_path}")

    return success_rate


if __name__ == "__main__":
    run_eval(parse_args())
