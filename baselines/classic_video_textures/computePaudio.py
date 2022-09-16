import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_Paudio(
    t_audio_eg: torch.Tensor, driving_audio: torch.Tensor
) -> torch.Tensor:
    s_a = F.normalize(t_audio_eg, dim=1)
    d_a = F.normalize(driving_audio, dim=0)
    d_a = d_a.unsqueeze(0)

    cos = nn.CosineSimilarity(dim=1)
    p_audio = cos(d_a.repeat([s_a.shape[0], 1]), s_a)

    p_audio = p_audio / (p_audio.sum() + 1e-6)

    return p_audio
