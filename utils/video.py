from pathlib import Path
from typing import Union, Optional

import ffmpeg
from ffmpeg.nodes import FilterableStream
import numpy as np
from skimage import io, color, img_as_ubyte, exposure
from tqdm import tqdm, trange

from .path import ensure_dir, sglob


class VideoError(Exception):
    pass



class Video:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.info = self.get_info() if self.path.is_file() else None

    @property
    def num_frames(self):
        # https://github.com/kkroening/ffmpeg-python/issues/110#issuecomment-412517933
        try:
            n = float(self.info['nb_frames'])
        except KeyError:
            n = self.fps * self.duration
        n = np.round(n).astype(int)
        return n

    @property
    def fps(self):
        r_fps = self.info['r_frame_rate']
        num, denom = r_fps.split('/')
        fps = float(num) / float(denom)
        return fps

    @property
    def duration(self):
        return float(self.info['duration'])

    @property
    def width(self):
        return int(self.info['width'])

    @property
    def height(self):
        return int(self.info['height'])

    @property
    def size(self):
        return self.width, self.height

    @property
    def shape(self):
        return self.height, self.width

    def get_info(self):
        probe = ffmpeg.probe(str(self.path))
        video_info = next(
            (
                stream
                for stream
                in probe['streams']
                if stream['codec_type'] == 'video'
            ),
            None,
        )
        return video_info

    def get_frames_pattern_string(self):
        """
        Return the pattern for the output frames names with the necessary
        zero-padding
        """
        n = len(str(self.num_frames))
        return f'{self.path.stem}_%0{n}d'

    def overlay(
            self,
            output_path: Union[str, Path],
            frames: bool = True,
            time: bool = True,
            downscale_factor: int = 2,
            subtitles_path: Optional[Union[str, Path, None]] = None,
            burn_subtitles: bool = True,
        ):
        """
        https://ffmpeg.org/ffmpeg-filters.html#drawtext-1
        """
        stream = ffmpeg.input(self.path)
        stream = ffmpeg.filter(
            stream,
            'scale',
            f'iw/{downscale_factor}',
            f'ih/{downscale_factor}',
        )
        if frames:
            stream = ffmpeg.filter(
                stream,
                'drawtext',
                text='%{n}',  # frame number
                fontcolor='Yellow',
                x=0,
                y=0,
            )
        if time:
            stream = ffmpeg.filter(
                stream,
                'drawtext',
                text='%{pts}',  # timestamp
                fontcolor='Yellow',
                x=0,
                y='h-text_h',
            )
        if subtitles_path is not None:
            # https://stackoverflow.com/a/13125122/3956024
            subtitles_path = Path(subtitles_path)
            if burn_subtitles:
                stream = ffmpeg.filter(
                    stream,
                    'subtitles',
                    str(subtitles_path),
                )
            else:
                pass  # TODO
        output = stream.output(str(output_path))
        output.run(quiet=True, overwrite_output=True)

    def write(self, output_path: Union[str, Path]):
        sex = SequenceExtractor(self, self.fps)
        stream = sex.get_sequence_stream()
        sex.write_stream(stream, output_path)

    def read_frame_from_process(self, process):
        in_bytes = process.stdout.read(self.width * self.height * 3)
        if not in_bytes:
            return None
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([self.height, self.width, 3])
        )
        return in_frame

    def get_motion_array(self):
        step = 10
        sex = SequenceExtractor(self)
        period = 1 / self.fps
        stream = sex.get_sequence_stream(0)
        previous = sex.get_array_from_stream(stream, period)
        mses = []
        for n in trange(step, self.num_frames, step):
            ss = frame_to_time(n, self.fps)
            stream = sex.get_sequence_stream(ss)
            current = sex.get_array_from_stream(stream, period)
            mse = self.get_mse(previous, current)
            mses.append(mse)
            previous = current
        return np.array(mses)

    def get_mse(self, a, b):
        diff = a - b
        sq = diff ** 2
        return sq.mean()



class SequenceExtractor:
    def __init__(
            self,
            video: Video,
            output_fps: Optional[float] = None,
            round_position: bool = True,
            quiet: bool = True,
            overwrite_output: bool = True,
        ):
        self.video = video
        self.output_fps = video.fps if output_fps is None else output_fps
        self.round_position = round_position
        self.quiet = quiet
        self.overwrite_output = overwrite_output

    def parse_position(self, position: float):
        if position < 0:
            message = 'Position is negative'
            raise VideoError(message)
        if position > self.video.duration:
            message = 'Position is larger than video duration'
            raise VideoError(message)

    def parse_duration(self, position, duration):
        """
        Unused for now
        """
        end = position + duration
        if end > self.video.duration:
            message = 'End position is larger than video duration'
            raise VideoError(message)

    def get_sequence_stream(
            self,
            position: Optional[float] = 0,
        ) -> FilterableStream:

        if self.round_position:
            position = round_time(position, self.video.fps)

        self.parse_position(position)

        stream = (
            ffmpeg
            .input(
                str(self.video.path),
                ss=position,  # always in input, it's faster
            )
            .filter(
                'fps',
                fps=self.output_fps,
                round='up',  # gave me the exact results
                eof_action='pass',
            )
        )
        return stream

    def get_buffer_from_stream(
            self,
            stream: FilterableStream,
            duration: Optional[float] = None,
        ) -> bytes:
        output_kwargs = dict(
            format='rawvideo',
            pix_fmt='rgb24',
        )
        if duration is not None:
            num_frames = self.output_fps * duration
            num_frames = np.round(num_frames).astype(int)
            output_kwargs['frames:v'] = num_frames
        out, _ = (
            stream
            .output('pipe:', **output_kwargs)
            .run(quiet=True)
        )
        return out

    def write_stream(
            self,
            stream: FilterableStream,
            output_path: Union[str, Path],
            duration: Optional[float] = None,
            extension='.jpg',
            pattern: Optional[str] = None,
            jpeg_quality: int = 2,  # 2-31 with 31 being the worst quality
        ):
        output_path = Path(output_path)
        ensure_dir(output_path)

        kwargs = {'qscale:v': jpeg_quality}
        if duration is not None:
            num_frames = self.output_fps * duration
            num_frames = np.round(num_frames).astype(int)
            kwargs['frames:v'] = num_frames
        is_dir = not output_path.suffix
        if is_dir:
            if pattern is None:
                pattern = self.video.get_frames_pattern_string()
            format_string = pattern + extension
            output_path = output_path / format_string
            kwargs['start_number'] = 0

        (
            stream
            .output(str(output_path), **kwargs)
            .run(quiet=self.quiet, overwrite_output=self.overwrite_output)
        )

    def write_gif(
            self,
            stream: FilterableStream,
            duration: float,
            output_path: Union[str, Path],
            palette_path: Optional[Union[str, Path]] = None,
        ):
        """
        TODO
        """
        # scale = ffmpeg.filter(stream, 'scale', width=320,
        #                       height=-1, flags='lanczos')
        if palette_path is None:
            palette_path = '/tmp/palette.png'
        palettegen = ffmpeg.filter(stream, 'palettegen')
        output = palettegen.output(str(palette_path))
        output.run(quiet=True)

    def get_array_from_stream(self, stream, duration=None) -> np.ndarray:
        buffer = self.get_buffer_from_stream(stream, duration)
        width, height = self.video.size
        array = (
            np
            .frombuffer(buffer, np.uint8)
            .reshape([-1, height, width, 3])
        )
        return array



def frames_to_video(
        frames_dir: Union[Path, str],
        video_path: Union[Path, str],
        fps: int = 25,
        format_: str = '*.jpg',
        ) -> None:
    ensure_dir(video_path)
    frames_dir = Path(frames_dir)
    input_pattern = frames_dir / format_
    stream = ffmpeg.input(input_pattern, pattern_type='glob', framerate=fps)
    stream = ffmpeg.output(stream, str(video_path))
    ffmpeg.run(stream, overwrite_output=True)

def time_to_frame(time, fps, floor=False):
    frame = time * fps
    if floor:
        frame = int(np.floor(frame))
    return frame

def round_time(time, fps):
    frame = np.floor(time_to_frame(time, fps))  # zero-based
    new_time = frame / fps
    return new_time

def frame_to_time(frame, fps):
    return frame / fps

def get_mean_frame(frames_dir):
    fps = sglob(frames_dir)
    one = io.imread(fps[0])
    result = np.zeros_like(one, float)
    for fp in tqdm(fps):
        result += io.imread(fp)
    return result / len(fps)

def subtract_mean_frame(input_dir, output_dir=None, mean=None):
    mean = get_mean_frame(input_dir) if mean is None else mean
    output_dir = input_dir if output_dir is None else output_dir
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    fps = sglob(input_dir)
    for src in tqdm(fps):
        image = io.imread(src)
        diff = np.abs(mean - image)
        dst = output_dir / src.name
        io.imsave(dst, diff.astype(np.uint8))

def enhance(input_dir, output_dir=None):
    output_dir = input_dir if output_dir is None else output_dir
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    fps = sglob(input_dir)
    for src in tqdm(fps):
        image = io.imread(src)
        hsv = color.rgb2hsv(image)
        v = hsv[..., 2]
        p1, p2 = np.percentile(v, (1, 99))
        v = exposure.rescale_intensity(v, in_range=(p1, p2))
        hsv[..., 2] = v
        rgb = color.hsv2rgb(hsv)
        dst = output_dir / src.name
        io.imsave(dst, img_as_ubyte(rgb))
