"""IWDD 2027 contest - WasteVision submission.

    python test.py --videos <videos folder> --results <results folder>

For every video in --videos it writes <video file name>.txt in --results (e.g. vid0041.mp4.txt): the second at which the illegal waste
dumping is detected (an integer), or an empty file when there is none. The method is described in README.md; the work is done in infer.py.
"""
import argparse


def init_parameter():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='results/', help="Results folder")
    # optional
    parser.add_argument("--chunk", type=int, default=None, help="questions scored per GPU block (default 40: the validated setting, ~11.3 GiB of VRAM; 8 uses ~6.3 GiB but slightly changes the logits)")
    parser.add_argument("--resume", action="store_true", help="skip the videos that already have a result file (to continue an interrupted run)")
    parser.add_argument("--workers", type=int, default=None, help="CPU processes preparing the next videos while the GPU works (default: CPUs - 1, max 4)")
    parser.add_argument("--work-dir", type=str, default=None, help="folder for temporary frames (default: system temp)")
    parser.add_argument("--log", type=str, default="submission_log.csv", help="per-video log (written outside the results folder)")
    parser.add_argument("--download-only", action="store_true", help="only download the base model weights from the Hugging Face Hub, then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = init_parameter()
    from infer import run
    raise SystemExit(run(args.videos, args.results, chunk=args.chunk, resume=args.resume, workers=args.workers, work_dir=args.work_dir,
                         log_path=args.log, download_only=args.download_only))
