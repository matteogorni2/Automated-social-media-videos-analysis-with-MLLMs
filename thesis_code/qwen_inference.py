import os
import sys
from tqdm import tqdm
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import torch
import argparse

from pathlib import Path
import yaml
import json
import shutil

# ----------------------
# Parse manifest path from CLI
# ----------------------
parser = argparse.ArgumentParser(description="Qwen3-VL multi-GPU video inference")
parser.add_argument("--manifest", type=str, required=True, help="Path to YAML manifest with parameters")
args = parser.parse_args()

manifest_path = Path(args.manifest)
if not manifest_path.exists():
    raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

# ----------------------
# Load manifest
# ----------------------
with open(manifest_path, 'r') as f:
    config = yaml.safe_load(f)

# ----------------------
# Assign variables
# ----------------------
MODEL_PATH = Path(config["model_path"])
VIDEO_DIR = config["video_path"]
NUM_GPUS = int(config.get("ntasks"))  # fallback to 1 if not specified
OUTPUT_DIR = config.get("output_dir", "results")
FINAL_OUTPUT = config.get("final_output", "final_results.json")
DO_QUANTIZE_MODEL = config.get("do_quantize_model", False)
MAX_PIXELS = int(config.get("max_pixels"))
MAX_FRAMES = int(config.get("max_frames"))
MAX_NEW_TOKENS = int(config.get("max_new_tokens"))
RESIZED_HEIGHT = int(config.get("resized_height",608))
RESIZED_WIDTH= int(config.get("resized_width",352))
PROMPT = config.get("prompt", "describe what you see")
DEFAULT_SYSTEM_PROMPT= "you are a professional video content moderator. Your task is to analyze videos based on specific guidelines and provide accurate annotations in JSON format."
SYSTEM_PROMPT=config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
# ----------------------
# Accelerator init
# ----------------------
accelerator = Accelerator()
device = accelerator.device

rank = accelerator.process_index
world_size = accelerator.num_processes

if NUM_GPUS != world_size and accelerator.is_main_process:
    print(f"[WARNING] manifest ntasks ({NUM_GPUS}) != accelerate processes ({world_size})")

# ----------------------
# Inference function
# ----------------------
def inference(
    model,
    processor,
    video,
    prompt,
    max_new_tokens=2048,
    total_pixels=20480 * 32 * 32,
    min_pixels=64 * 32 * 32,
    max_frames=2048,
    sample_fps=2,
):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"video": video,
             "total_pixels": total_pixels,
             "min_pixels": min_pixels,
             "max_frames": max_frames,
             "sample_fps": sample_fps,
             "resized_height": RESIZED_HEIGHT,
             "resized_width": RESIZED_WIDTH},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
    except Exception as e:
        print(f"[WARNING] Skipping video due to loading failure: {video}")
        print(f"[ERROR] {e}")
        return None

    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return output_text[0]

def sanitize_json(text: str) -> str:
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start == -1:
        return text
    try:
        _, end = decoder.raw_decode(text[start:])
        return text[start:start + end]
    except Exception:
        return text

def safe_json_load(output_text):
        try:
            return json.loads(output_text)
        except Exception as e:
            print(f"JSON conversion error for video")
            return None
# ----------------------
# Model + processor loading
# ---------------------

if DO_QUANTIZE_MODEL:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": device},
        attn_implementation="flash_attention_2"
    )
else:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": device},
        attn_implementation="flash_attention_2"
    )

processor = AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)
model.eval()
# Load and split videos
# ----------------------
video_files = sorted(
    os.path.join(VIDEO_DIR, f)
    for f in os.listdir(VIDEO_DIR)
)

videos_for_rank = video_files[rank::world_size]

if accelerator.is_main_process:
    print(f"Total videos: {len(video_files)}")
    print(f"GPUs: {world_size}")

print(f"[Rank {rank}] Processing {len(videos_for_rank)} videos")

# ----------------------
# Inference loop
# ----------------------
model_annotations = []

for video_path in tqdm(videos_for_rank, desc=f"Rank {rank} videos", ncols=100,file=sys.stdout):
    result = inference(
        model,
        processor,
        video_path,
        PROMPT,
        total_pixels=MAX_PIXELS,
        max_frames=MAX_FRAMES,
        max_new_tokens=MAX_NEW_TOKENS
    )

    if result is None:
        print(f"[INFO] Skipping video due to inference failure: {video_path}")
        continue

    video_name = Path(video_path).stem  # video.mp4 → video
    sanitized = sanitize_json(result)
    parsed = safe_json_load(sanitized)

    if parsed is not None:
        parsed["video_name"] = video_name
        model_annotations.append(parsed)

###OUTPUT paths defnition
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

EXPERIMENT_DIR = os.path.join(
    OUTPUT_DIR, f"experiment_{JOB_ID}_{FINAL_OUTPUT}"
)
Path(EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)

rank_json_path = os.path.join(EXPERIMENT_DIR, f"annotations_rank_{rank}.json")
with open(rank_json_path, "w") as _f:
    json.dump(model_annotations, _f, ensure_ascii=False, indent=2)

print(f"[Rank {rank}] Wrote {rank_json_path}")

# ----------------------
# Synchronization barrier
# ----------------------
accelerator.wait_for_everyone()

# ----------------------
# Final merge (rank 0 only)
# ----------------------
if accelerator.is_main_process:
    all_annotations = []

    for r in range(world_size):
        rank_file = os.path.join(
            EXPERIMENT_DIR, f"annotations_rank_{r}.json"
        )
        if os.path.exists(rank_file):
            with open(rank_file, "r") as _f:
                try:
                    data = json.load(_f)
                    if isinstance(data, list):
                        all_annotations.extend(data)
                    else:
                        all_annotations.append(data)
                except Exception as e:
                    print(f"[MAIN] Failed to load {rank_file}: {e}")

    if all_annotations:
        final_name = FINAL_OUTPUT
        if not final_name.lower().endswith('.json'):
            final_name = final_name + '.json'
        final_json_path = os.path.join(EXPERIMENT_DIR, final_name)
        with open(final_json_path, 'w') as _f:
            json.dump(all_annotations, _f, ensure_ascii=False, indent=2)
        print(f"[MAIN] Full annotations written to {final_json_path}")
    else:
        print("[MAIN] No annotations were collected from any rank.")
    # ----------------------
    # Delete intermediate rank files
    # ----------------------
    for r in range(world_size):
        rank_file = os.path.join(EXPERIMENT_DIR, f"annotations_rank_{r}.json")
        if os.path.exists(rank_file):
            try:
                os.remove(rank_file)
                print(f"[MAIN] Removed intermediate file {rank_file}")
            except Exception as e:
                print(f"[MAIN] Failed to remove {rank_file}: {e}")

    shutil.copy(
        manifest_path,
        os.path.join(EXPERIMENT_DIR, manifest_path.name)
    )
    print("[MAIN] Manifest copied to experiment directory")
