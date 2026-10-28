import shutil
from pathlib import Path

import cv2
from tqdm.auto import tqdm

VIDEO_PATH = Path("dataset/train/videos/vid0004.mp4")
OUTPUT_ROOT = Path("frames")
FPS_OUT = 4


def decompose_video(video_path: Path, fps_out: float = FPS_OUT, output_root: Path = OUTPUT_ROOT) -> list[Path]:
    """Estrae fps_out frame al secondo dal video e li salva in output_root/{nome_video}."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire il video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = fps_in / fps_out

    out_dir = output_root / video_path.stem
    # Si scrive in una cartella temporanea rinominata a fine estrazione: se esiste, la cartella finale e' completa.
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    names = []
    next_idx = 0.0
    for idx in tqdm(range(total), desc=f"Decompongo {video_path.stem}", unit="frame", leave=False, mininterval=1.0):
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= round(next_idx):
            name = f"frame_{len(names):04d}.jpg"
            cv2.imwrite(str(tmp_dir / name), frame)
            names.append(name)
            next_idx += step

    cap.release()
    if out_dir.exists():
        shutil.rmtree(out_dir)  # evita frame rimasti da esecuzioni precedenti
    tmp_dir.rename(out_dir)
    return [out_dir / name for name in names]


if __name__ == "__main__":
    paths = decompose_video(VIDEO_PATH)
    print(f"Salvati {len(paths)} frame in {paths[0].parent}")
