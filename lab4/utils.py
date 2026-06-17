"""Lab3 CTC-ASR 辅助工具:
- AN4 数据加载 (按 fileids 白名单, 不再扫整个 wav 目录)
- 文本 <-> 索引转换 / CTC 贪心解码
- SpecAugment 与 speed perturbation 增强器
- 一致的 CER / WER 评测 (跨 batch 累积分子分母再求商)
"""
import os
import re
from typing import List, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcodec.decoders._audio_decoder import AudioDecoder
import jiwer


# ============================================================
# 1. 数据加载
# ============================================================

def load_an4_split(data_dir: str, split: str) -> List[Tuple[str, torch.Tensor, int, str]]:
    """读取 AN4 的 train 或 test 子集。

    参数
    ----
    data_dir: AN4 根目录, 例如 './data/AN4'
    split:    'train' 或 'test'

    返回
    ----
    一个 list, 每个元素是 (file_id, waveform, sample_rate, transcript_text)
    其中 transcript_text 已去掉 '<s>' '</s>' 起止标记。
    """
    assert split in ("train", "test")
    fileids_path = os.path.join(data_dir, "etc", f"an4_{split}.fileids")
    trans_path   = os.path.join(data_dir, "etc", f"an4_{split}.transcription")

    # 读取 fileids: 每行一条相对路径 (不含 .wav)
    with open(fileids_path) as f:
        fileids = [line.strip() for line in f if line.strip()]

    # 读取 transcription, 形如:
    #   <s> RUBOUT J B X R Z NINE TWENTY </s> (an1-mblw-b)
    transcripts = {}
    line_re = re.compile(r"^(.*)\(([^)]+)\)\s*$")
    inner_re = re.compile(r"<s>\s*(.+?)\s*</s>")
    with open(trans_path) as f:
        for line in f:
            m = line_re.match(line.strip())
            if not m:
                continue
            text_raw, fid_short = m.group(1).strip(), m.group(2).strip()
            inner = inner_re.match(text_raw)
            text = inner.group(1) if inner else text_raw
            transcripts[fid_short] = text

    data = []
    for fid in fileids:
        # fid 例如 'an4_clstk/fash/an251-fash-b'
        wav_path = os.path.join(data_dir, "wav", fid + ".wav")
        samples = AudioDecoder(wav_path).get_all_samples()
        waveform, sr = samples.data, int(samples.sample_rate)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        fid_short = os.path.basename(fid)
        transcript = transcripts.get(fid_short)
        if transcript is None:
            continue
        data.append((fid_short, waveform, sr, transcript))
    return data


# ============================================================
# 2. Mel 频谱
# ============================================================

def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Build triangular HTK mel filters with ``(n_freqs, n_mels)`` orientation."""
    f_max = float(sample_rate / 2 if f_max is None else f_max)
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0.0, sample_rate / 2, n_freqs)

    mel_min = _hz_to_mel(torch.tensor(float(f_min)))
    mel_max = _hz_to_mel(torch.tensor(float(f_max)))
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
    f_pts = _mel_to_hz(mel_pts)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down_slopes = -slopes[:, :-2] / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    return torch.clamp(torch.minimum(down_slopes, up_slopes), min=0.0)


class TorchMelSpectrogram(nn.Module):
    """Small torch-only MelSpectrogram replacement."""

    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        win_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft if win_length is None else win_length
        self.power = power
        self.center = center
        self.pad_mode = pad_mode

        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer(
            "mel_fb",
            _build_mel_filterbank(sample_rate, n_fft, n_mels, f_min=f_min, f_max=f_max),
            persistent=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        window = self.get_buffer("window").to(device=waveform.device, dtype=waveform.dtype)
        mel_fb = self.get_buffer("mel_fb").to(device=waveform.device, dtype=waveform.dtype)

        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            return_complex=True,
        )
        power_spec = spec.abs().pow(self.power)       # (..., n_freqs, T)
        return torch.matmul(mel_fb.transpose(0, 1), power_spec)


# ============================================================
# 3. 字符 <-> 索引
# ============================================================

def text_to_indices(text: str, vocab: Sequence[str]) -> List[int]:
    """把字符序列转成词表索引序列, 不在词表里的字符 (如标点) 会被丢弃。"""
    ch2i = {ch: i for i, ch in enumerate(vocab)}
    return [ch2i[c] for c in text if c in ch2i]


def indices_to_text(indices: Sequence[int], vocab: Sequence[str]) -> str:
    """索引序列转回字符串, 跳过 blank(0)。"""
    return "".join(vocab[i] for i in indices if i != 0 and i < len(vocab))


# ============================================================
# 4. CTC 贪心解码
# ============================================================

def ctc_greedy_decode(indices_2d, blank_index: int = 0) -> List[List[int]]:
    """对每条序列做 CTC 贪心解码: 先合并相邻重复, 再去掉 blank。

    参数
    ----
    indices_2d: shape (B, T) 的 numpy / list / tensor
    """
    out: List[List[int]] = []
    for seq in indices_2d:
        decoded: List[int] = []
        prev = None
        for idx in seq:
            idx = int(idx)
            if idx != blank_index and idx != prev:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out


# ============================================================
# 5. 评测: 跨 batch 累积再求商
# ============================================================

class WerCerMeter:
    """正确的 CER / WER 跨 batch 聚合: 把所有 ref/hyp 攒到 list, 最后整体调 jiwer。

    用法:
        m = WerCerMeter()
        for batch:  m.update(ref_list, hyp_list)
        cer, wer = m.compute()
    """
    def __init__(self):
        self.refs: List[str] = []
        self.hyps: List[str] = []

    def update(self, refs: Sequence[str], hyps: Sequence[str]) -> None:
        self.refs.extend(refs)
        self.hyps.extend(hyps)

    def compute(self) -> Tuple[float, float]:
        if not self.refs:
            return 0.0, 0.0
        cer = jiwer.cer(self.refs, self.hyps)
        wer = jiwer.wer(self.refs, self.hyps)
        return float(cer), float(wer)

    def reset(self) -> None:
        self.refs.clear(); self.hyps.clear()


# ============================================================
# 6. 增强: SpecAugment & Speed Perturbation
# ============================================================

def _random_mask(x: torch.Tensor, dim: int, max_width: int) -> torch.Tensor:
    size = x.size(dim)
    width_limit = min(max_width, size)
    if width_limit <= 0:
        return x

    width = int(torch.randint(0, width_limit + 1, (), device=x.device).item())
    if width == 0:
        return x

    start = int(torch.randint(0, size - width + 1, (), device=x.device).item())
    y = x.clone()
    index = [slice(None)] * y.dim()
    index[dim] = slice(start, start + width)
    y[tuple(index)] = 0
    return y

class SpecAugment:
    """简化版 SpecAugment: 频率遮挡 + 时间遮挡 (各做一次)。

    输入: (T, F) mel-spec  →  输出: (T, F) 同形状, 部分位置被置 0。
    只在训练阶段调用, 评测阶段不要用。
    """
    def __init__(self, freq_mask_param: int = 15, time_mask_param: int = 35):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(self, mel_TxF: torch.Tensor) -> torch.Tensor:
        x = _random_mask(mel_TxF, dim=1, max_width=self.freq_mask_param)
        return _random_mask(x, dim=0, max_width=self.time_mask_param)


def _linear_resample(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    if orig_freq == new_freq:
        return waveform

    old_length = waveform.size(-1)
    new_length = max(1, int(round(old_length * float(new_freq) / float(orig_freq))))
    flat = waveform.reshape(-1, 1, old_length)
    resampled = F.interpolate(flat, size=new_length, mode="linear", align_corners=False)
    return resampled.reshape(*waveform.shape[:-1], new_length)


def speed_perturb(waveform: torch.Tensor, sr: int, speed: float) -> torch.Tensor:
    """通过 resample 实现"变速 + 变调"的廉价 perturbation。

    speed=1.0  原音频; speed=0.9 慢一点; speed=1.1 快一点。
    """
    if speed == 1.0:
        return waveform
    new_sr = int(sr * speed)
    return _linear_resample(waveform, sr, new_sr)


def expand_with_speed_perturb(records, speeds=(0.9, 1.0, 1.1)):
    """对一个 AN4 数据列表做三档 speed perturb, 返回 3× 大小的新列表。"""
    out = []
    for fid, wav, sr, txt in records:
        for sp in speeds:
            new_wav = speed_perturb(wav, sr, sp)
            new_fid = f"{fid}_sp{sp}"
            # 注意: 即使 sample_rate 实际变了, 我们仍按 16000 喂给 MelSpec
            #   这等价于"加速 + 升调", 教学够用
            out.append((new_fid, new_wav, sr, txt))
    return out
