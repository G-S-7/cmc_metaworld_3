import os
import subprocess
from pathlib import Path
from cmc_rl.tasks import SIX_TASK_SEQUENCE, TASKS

def render_all():
    tasks = SIX_TASK_SEQUENCE
    labels = ["T1", "T2", "T3", "T4", "T5", "T6"]

    experiments = [
        ("BEFORE CMC", "results/before_cmc"),
        ("WITH CMC", "results/after_cmc")
    ]

    for exp_name, res_dir in experiments:
        print(f"\n==============================================")
        print(f" Rendering videos for {exp_name}")
        print(f"==============================================")
        
        for i, (task_name, label) in enumerate(zip(tasks, labels)):
            ckpt_path = Path(res_dir) / "checkpoints" / f"stage_{label}.pt"
            
            if not ckpt_path.exists():
                print(f"  [SKIPPED] Checkpoint not found: {ckpt_path}")
                continue
            
            out_dir = Path(res_dir) / "videos"
            video_name = f"task_{i+1:02d}_{task_name}"
            
            cmd = [
                "python", "render_agent.py",
                "--checkpoint", str(ckpt_path),
                "--task", task_name,
                "--episodes", "2",
                "--fps", "50",
                "--out-dir", str(out_dir)
            ]
            
            print(f"  Rendering {task_name} from {label} checkpoint...")
            subprocess.run(cmd, check=False)
            
            # rename the video to match the required naming scheme
            safe_task = task_name.replace("-", "_")
            mp4_path = out_dir / f"{safe_task}.mp4"
            if mp4_path.exists():
                mp4_path.rename(out_dir / f"{video_name}.mp4")

if __name__ == "__main__":
    render_all()
