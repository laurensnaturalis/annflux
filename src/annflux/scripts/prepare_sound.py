# This is a sample Python script.
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from pydub import AudioSegment
from spectrogrammer import get_spectrogram, plot_spectrogram


def main(input_path, out_folder):
    analyzer = Analyzer()
    out_path = Path(out_folder)
    out_path.mkdir(exist_ok=True)
    wav_out_folder = out_path / "wav"
    wav_out_folder.mkdir(exist_ok=True)
    spec_out = out_path / "images"
    spec_out.mkdir(exist_ok=True)

    full_sounds = AudioSegment.from_wav(input_path)
    print(full_sounds.duration_seconds / 3)
    time_offsets = range(0, int(full_sounds.duration_seconds), 3)
    print(f"{len(time_offsets)=}")
    multithreaded = False
    if not multithreaded:
        features = []
        for time_offset in time_offsets:
            print(time_offset)
            features.append(
                process_segment(
                    time_offset, analyzer, full_sounds, spec_out, wav_out_folder
                )
            )
    else:
        with Pool(16) as p:
            # print(f"{len(list(data_for_map))=}")
            features = p.starmap_async(
                process_segment,
                zip(
                    time_offsets,
                    [
                        None,
                    ]
                    * len(time_offsets),
                    [full_sounds] * len(time_offsets),
                    [spec_out] * len(time_offsets),
                    [wav_out_folder] * len(time_offsets),
                ),
            )
            print(f"{features.ready()=}")
            while not features.ready():
                pass
                # print(features._number_left)
            print(f"{features.ready()=}")
        # print(f"{features.get()=}")
        features = features.get()
    # print("")
    # np.savez("embeddings.npz", np.vstack())
    np.savez("embeddings.npz", np.vstack(features))


def process_segment(start_time, analyzer, full_sounds, spec_out, wav_out):
    print(start_time / 3)
    segment = full_sounds[(start_time * 1000) : ((start_time + 3) * 1000)]
    segment_wav_path = wav_out / f"t_{start_time}s.wav"
    segment.export(segment_wav_path)
    recording = Recording(
        analyzer,
        segment_wav_path,
        lat=51.77667,
        lon=5.93611,
        date=datetime(year=2024, month=5, day=11),  # TODO
        min_conf=0.05,
    )
    spectrogram = get_spectrogram(segment_wav_path)
    plot_spectrogram(spectrogram)
    plt.axis("off")
    plt.savefig(spec_out / f"t_{start_time}s.jpg", bbox_inches="tight")
    plt.close(plt.gcf())
    # recording.analyze()
    # recording.extract_embeddings()
    return np.zeros(512)  # recording.embeddings[0]["embeddings"]


if __name__ == "__main__":
    main(
        "/media/lhogeweg/My Book/hazehorst/audiomoth-bathroom/20250204_070000.WAV",
        "/mnt/big/indeed/audiomoth_20250204_070000_test",
    )
