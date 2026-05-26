from pathlib import Path

from .utils import FrameRecords, export_single_framearray, iter_frame_types


def export_debug_txt_merged(
    output_path: str | Path, frame_records: FrameRecords
) -> None:
    """Export all config frame types into one human-readable text file."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    frame_path = out / "cfg_frames.txt"
    with frame_path.open("w") as frame_file:
        for cp, frames in frame_records:
            frame_file.write(f"# Core at coord (X,Y)=({cp.coord.x},{cp.coord.y}):\n")
            for idx, frame_array in iter_frame_types(frames):
                frame_file.write(f"\ttype{idx}:\n")
                export_single_framearray(frame_array, frame_file, prefix="\t\t0x")


def export_debug_txt_files(
    output_path: str | Path, frame_records: FrameRecords
) -> None:
    """Export per-type config frames as human-readable text files."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    frame1_path = out / "cfg_frame1.txt"
    frame2_path = out / "cfg_frame2.txt"
    frame3_path = out / "cfg_frame3.txt"
    with (
        frame1_path.open("w") as frame1_file,
        frame2_path.open("w") as frame2_file,
        frame3_path.open("w") as frame3_file,
    ):
        files = [frame1_file, frame2_file, frame3_file]
        for cp, frames in frame_records:
            coord_line = f"# Core at coord (X,Y)=({cp.coord.x},{cp.coord.y}):\n"
            frame1_file.write(coord_line)
            frame2_file.write(coord_line)
            frame3_file.write(coord_line)
            for idx, frame_array in iter_frame_types(frames):
                export_single_framearray(frame_array, files[idx - 1], prefix="\t0x")
