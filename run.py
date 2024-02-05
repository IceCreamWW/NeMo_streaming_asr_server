import copy

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from omegaconf import OmegaConf, open_dict

def extract_transcriptions(hyps):
    """
        The transcribed_texts returned by CTC and RNNT models are different.
        This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions


def init_preprocessor(asr_model):
    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    cfg.preprocessor.normalize = "None"

    preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
    preprocessor.to(asr_model.device)

    return preprocessor


class NeMoTranscriber:
    # 1) "stt_en_fastconformer_hybrid_large_streaming_multi"
    # 2) "stt_en_fastconformer_hybrid_large_streaming_80ms"
    # 3) "stt_en_fastconformer_hybrid_large_streaming_480ms"
    # 4) "stt_en_fastconformer_hybrid_large_streaming_1040ms"
    model_name = "stt_en_fastconformer_hybrid_large_streaming_1040ms"
    decoder_type = "rnnt"
    lookahead_size = 1040
    encoder_step_length = 80
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    left_context_size = asr_model.encoder.att_context_size[0]
    asr_model.encoder.set_default_att_context_size(
        [left_context_size, int(lookahead_size / encoder_step_length)]
    )
    # make sure the model's decoding strategy is optimal
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        # save time by doing greedy decoding and not trying to record the alignments
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = False
        if hasattr(asr_model, "joint"):  # if an RNNT model
            # restrict max_symbols to make sure not stuck in infinite loop
            decoding_cfg.greedy.max_symbols = 10
            # sensible default parameter, but not necessary since batch size is 1
            decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(decoding_cfg)
    asr_model.eval()

    # init params we will use for streaming
    previous_hypotheses = None
    pred_out_stream = None
    step_num = 0
    pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
    # cache-aware models require some small section of the previous processed_signal to
    # be fed in at each timestep - we initialize this to a tensor filled with zeros
    # so that we will do zero-padding for the very first chunk(s)
    num_channels = asr_model.cfg.preprocessor.features

    preprocessor = init_preprocessor(asr_model)
    chunk_size = lookahead_size + encoder_step_length

    @classmethod
    def get_initial_states(cls):
        (
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
        ) = cls.asr_model.encoder.get_initial_cache_state(batch_size=1)

        step_num = 0
        previous_hypotheses = None
        pred_out_stream = None
        cache_pre_encode = torch.zeros(
            (1, cls.num_channels, cls.pre_encode_cache_size), device=cls.asr_model.device
        )
        states = (cache_last_channel, cache_last_time, cache_last_channel_len, previous_hypotheses, pred_out_stream, step_num, cache_pre_encode)
        return states

    @classmethod
    def preprocess_audio(cls, audio):
        device = cls.asr_model.device

        # doing audio preprocessing
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
        processed_signal, processed_signal_length = cls.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length

    @classmethod
    def transcribe_chunk(cls, audio_chunk, prev_states):

        (cache_last_channel, cache_last_time, cache_last_channel_len, previous_hypotheses, pred_out_stream, step_num, cache_pre_encode) = prev_states

        # new_chunk is provided as np.int16, so we convert it to np.float32
        # as that is what our ASR models expect
        audio_data = audio_chunk.astype(np.float32)
        audio_data = audio_data / 32768.0

        # get mel-spectrogram signal & length
        processed_signal, processed_signal_length = cls.preprocess_audio(audio_data)

        # prepend with cache_pre_encode
        processed_signal = torch.cat([cache_pre_encode, processed_signal], dim=-1)
        processed_signal_length += cache_pre_encode.shape[1]

        # save cache for next time
        cache_pre_encode = processed_signal[:, :, -cls.pre_encode_cache_size:]

        with torch.no_grad():
            (
                pred_out_stream,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = cls.asr_model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )

        final_streaming_tran = extract_transcriptions(transcribed_texts)
        step_num += 1

        states = (cache_last_channel, cache_last_time, cache_last_channel_len, previous_hypotheses, pred_out_stream, step_num, cache_pre_encode)

        return final_streaming_tran[0], states


if __name__ == "__main__":

    import librosa
    # audio, sr = librosa.load('tests/jfk.flac', sr=16000)
    audio, sr = librosa.load(
        "/mnt/wsl/PHYSICALDRIVE0p3/home/vv/downloads/data/zheda240130/audios/1_en.wav", sr=16000
    )
    audio = (audio * 32768.0).astype(np.int16)

    states = NeMoTranscriber.get_initial_states()
    for i in range(0, len(audio), int(sr * NeMoTranscriber.chunk_size / 1000)):
        chunk = audio[i : i + int(sr * NeMoTranscriber.chunk_size / 1000)]
        if len(chunk) < int(sr * NeMoTranscriber.chunk_size / 1000):
            chunk = np.pad(chunk, (0, int(sr * NeMoTranscriber.chunk_size / 1000) - len(chunk)))
        text, states = NeMoTranscriber.transcribe_chunk(chunk, states)
        print(text)

