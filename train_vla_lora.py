import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "dataset_v1")
DEFAULT_SAVE_DIR = os.path.join(ROOT_DIR, "vla_lora_adapter")
MODEL_ID = "openvla/openvla-7b"
IMAGE_PATTERN = re.compile(r"^(?P<traj>.+)_step_(?P<step>\d+)\.[^.]+$")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--chunk-stride", type=int, default=1)
    parser.add_argument("--tail-mode", choices=["repeat_last", "drop"], default="repeat_last")
    return parser.parse_args()


def action_to_tokens(action, vocab_size, bins=256):
    action = np.clip(action, -1.0, 1.0)
    token_bins = np.linspace(-1.0, 1.0, bins)
    discretized = np.digitize(action, token_bins)
    return vocab_size - discretized


def parse_image_key(image_name):
    match = IMAGE_PATTERN.match(os.path.basename(image_name))
    if not match:
        raise ValueError(f"无法从图片名解析轨迹和步数: {image_name}")
    return match.group("traj"), int(match.group("step"))


def build_chunk_windows(samples, chunk_size, chunk_stride, tail_mode):
    grouped = defaultdict(list)
    for item in samples:
        traj_name, step_idx = parse_image_key(item["image"])
        grouped[traj_name].append((step_idx, item))

    windows = []
    traj_lengths = {}

    for traj_name, items in grouped.items():
        ordered_items = [item for _, item in sorted(items, key=lambda x: x[0])]
        traj_lengths[traj_name] = len(ordered_items)

        for start_idx in range(0, len(ordered_items), chunk_stride):
            chunk_items = ordered_items[start_idx : start_idx + chunk_size]
            if len(chunk_items) < chunk_size:
                if tail_mode == "drop":
                    break
                chunk_items = chunk_items + [chunk_items[-1]] * (chunk_size - len(chunk_items))

            anchor_item = ordered_items[start_idx]
            chunk_actions = np.asarray([sample["action"] for sample in chunk_items], dtype=np.float32)
            windows.append(
                {
                    "image": anchor_item["image"],
                    "instruction": anchor_item["instruction"],
                    "actions": chunk_actions,
                    "traj_name": traj_name,
                    "start_step": start_idx,
                }
            )

    return windows, traj_lengths


class ActionChunkDataset(Dataset):
    def __init__(self, jsonl_path, img_dir, processor, vocab_size, chunk_size, chunk_stride, tail_mode):
        self.processor = processor
        self.img_dir = img_dir
        self.vocab_size = vocab_size
        self.tokenizer = processor.tokenizer

        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_samples = [json.loads(line) for line in f]

        self.samples, self.traj_lengths = build_chunk_windows(
            raw_samples,
            chunk_size=chunk_size,
            chunk_stride=chunk_stride,
            tail_mode=tail_mode,
        )
        self.chunk_size = chunk_size
        self.tail_mode = tail_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(os.path.join(self.img_dir, item["image"])).convert("RGB")

        chunk_action_tokens = action_to_tokens(item["actions"], self.vocab_size).reshape(-1)
        action_token_str = self.tokenizer.decode(chunk_action_tokens.tolist())
        prompt_prefix = f"In: What action should the robot take to {item['instruction']}?\nOut:"
        prompt = f"{prompt_prefix} {action_token_str}</s>"

        inputs = self.processor(prompt, image, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, fill_value=-100)
        labels[-len(chunk_action_tokens) :] = input_ids[-len(chunk_action_tokens) :]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


def build_collate_fn(processor):
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    def collate_fn(batch):
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
        pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }

    return collate_fn


def train():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/4] 加载处理器: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    vocab_size = len(processor.tokenizer)
    print(f"[检查] vocab_size={vocab_size}")

    print("[2/4] 加载基础模型并注入 LoRA...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    jsonl_path = os.path.join(args.data_dir, "dataset.jsonl")
    img_dir = os.path.join(args.data_dir, "images")
    dataset = ActionChunkDataset(
        jsonl_path=jsonl_path,
        img_dir=img_dir,
        processor=processor,
        vocab_size=vocab_size,
        chunk_size=args.chunk_size,
        chunk_stride=args.chunk_stride,
        tail_mode=args.tail_mode,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=build_collate_fn(processor),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_trajs = len(dataset.traj_lengths)
    avg_traj_len = sum(dataset.traj_lengths.values()) / max(total_trajs, 1)
    target_token_count = args.chunk_size * 7
    effective_batch_size = args.batch_size * args.grad_accum_steps
    print(f"[3/4] 开始训练，窗口样本数: {len(dataset)} | 轨迹数: {total_trajs}")
    print(
        f"[数据] chunk_size={args.chunk_size} stride={args.chunk_stride} tail_mode={args.tail_mode} "
        f"| avg_traj_len={avg_traj_len:.1f} | target_action_tokens={target_token_count}"
    )
    print(
        f"[配置] epochs={args.epochs} batch_size={args.batch_size} "
        f"grad_accum={args.grad_accum_steps} logical_batch={effective_batch_size} lr={args.lr}"
    )
    print(f"[监督] loss 仅作用于每个样本末尾的 {target_token_count} 个 action tokens")
    if args.chunk_size >= 16:
        print("[显存提示] 当前 chunk_size>=16，建议继续保持 batch_size=1，并视情况进一步增大 grad_accum_steps。")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )

            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum_steps

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if batch_idx % args.log_every == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Batch {batch_idx:04d}/{len(train_loader):04d} | "
                    f"Loss {loss.item() * args.grad_accum_steps:.4f}"
                )

        if len(train_loader) % args.grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"[Epoch 完成] epoch={epoch + 1} avg_loss={avg_loss:.4f} global_step={global_step}")

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    print(f"[4/4] 训练完成，LoRA adapter 已保存到: {args.save_dir}")


if __name__ == "__main__":
    train()
