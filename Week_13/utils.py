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
import torchaudio
import torchaudio.functional as AF
from torchaudio.transforms import FrequencyMasking, TimeMasking
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
        waveform, sr = torchaudio.load(wav_path)
        fid_short = os.path.basename(fid)
        transcript = transcripts.get(fid_short)
        if transcript is None:
            continue
        data.append((fid_short, waveform, sr, transcript))
    return data


# ============================================================
# 2. 字符 <-> 索引
# ============================================================

def text_to_indices(text: str, vocab: Sequence[str]) -> List[int]:
    """把字符序列转成词表索引序列, 不在词表里的字符 (如标点) 会被丢弃。"""
    ch2i = {ch: i for i, ch in enumerate(vocab)}
    return [ch2i[c] for c in text if c in ch2i]


def indices_to_text(indices: Sequence[int], vocab: Sequence[str]) -> str:
    """索引序列转回字符串, 跳过 blank(0)。"""
    return "".join(vocab[i] for i in indices if i != 0 and i < len(vocab))


# ============================================================
# 3. CTC 贪心解码
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
# 4. 评测: 跨 batch 累积再求商
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
# 5. 增强: SpecAugment & Speed Perturbation
# ============================================================

class SpecAugment:
    """简化版 SpecAugment: 频率遮挡 + 时间遮挡 (各做一次)。

    输入: (T, F) mel-spec  →  输出: (T, F) 同形状, 部分位置被置 0。
    只在训练阶段调用, 评测阶段不要用。
    """
    def __init__(self, freq_mask_param: int = 15, time_mask_param: int = 35):
        self.freq_mask = FrequencyMasking(freq_mask_param)
        self.time_mask = TimeMasking(time_mask_param)

    def __call__(self, mel_TxF: torch.Tensor) -> torch.Tensor:
        x = self.freq_mask(mel_TxF.transpose(0, 1))   # (F, T)
        x = self.time_mask(x)
        return x.transpose(0, 1)                       # (T, F)


def speed_perturb(waveform: torch.Tensor, sr: int, speed: float) -> torch.Tensor:
    """通过 resample 实现"变速 + 变调"的廉价 perturbation。

    speed=1.0  原音频; speed=0.9 慢一点; speed=1.1 快一点。
    """
    if speed == 1.0:
        return waveform
    new_sr = int(sr * speed)
    return AF.resample(waveform, sr, new_sr)


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
