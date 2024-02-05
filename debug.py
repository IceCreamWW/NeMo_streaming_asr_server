# import dependencies
import copy
import librosa
import time
import pyaudio as pa
import numpy as np
import torch
from nemo_live.transcriber import NeMoTranscriber

from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

# specify sample rate we will use for recording audio
SAMPLE_RATE = 16000 # Hz


# You may wish to try different values of model_name and lookahead_size

# Choose a the name of a model to use.
# Currently available options:
# 1) "stt_en_fastconformer_hybrid_large_streaming_multi"
# 2) "stt_en_fastconformer_hybrid_large_streaming_80ms"
# 3) "stt_en_fastconformer_hybrid_large_streaming_480ms"
# 4) "stt_en_fastconformer_hybrid_large_streaming_1040ms"

model_name = "stt_en_fastconformer_hybrid_large_streaming_1040ms"

# Specify the lookahead_size.
# If model_name == "stt_en_fastconformer_hybrid_large_streaming_multi" then
# lookahead_size can be 0, 80, 480 or 1040 (ms)
# Else, lookahead_size should be whatever is written in the model_name:
# "stt_en_fastconformer_hybrid_large_streaming_<lookahead_size>ms"

lookahead_size = 1040 # in milliseconds

# Specify the decoder to use.
# Can be "rnnt" or "ctc"
decoder_type = "rnnt"


# setting up model and validating the choice of model_name and lookahead size
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)


# specify ENCODER_STEP_LENGTH (which is 80 ms for FastConformer models)
ENCODER_STEP_LENGTH = 80 # ms

# update att_context_size if using multi-lookahead model
# (for single-lookahead models, the default context size will be used and the
# `lookahead_size` variable will be ignored)
if model_name == "stt_en_fastconformer_hybrid_large_streaming_multi":
    # check that lookahead_size is one of the valid ones
    if lookahead_size not in [0, 80, 480, 1040]:
        raise ValueError(
            f"specified lookahead_size {lookahead_size} is not one of the "
            "allowed lookaheads (can select 0, 80, 480 or 1040 ms)"
        )

    # update att_context_size
    left_context_size = asr_model.encoder.att_context_size[0]
    asr_model.encoder.set_default_att_context_size([left_context_size, int(lookahead_size / ENCODER_STEP_LENGTH)])


# make sure we use the specified decoder_type
asr_model.change_decoding_strategy(decoder_type=decoder_type)

# make sure the model's decoding strategy is optimal
decoding_cfg = asr_model.cfg.decoding
with open_dict(decoding_cfg):
    # save time by doing greedy decoding and not trying to record the alignments
    decoding_cfg.strategy = "greedy"
    decoding_cfg.preserve_alignments = False
    if hasattr(asr_model, 'joint'):  # if an RNNT model
        # restrict max_symbols to make sure not stuck in infinite loop
        decoding_cfg.greedy.max_symbols = 10
        # sensible default parameter, but not necessary since batch size is 1
        decoding_cfg.fused_batch_size = -1
    asr_model.change_decoding_strategy(decoding_cfg)


# set model to eval mode
asr_model.eval()


# get parameters to use as the initial cache state
cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
    batch_size=1
)


# init params we will use for streaming
previous_hypotheses = None
pred_out_stream = None
step_num = 0
pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
# cache-aware models require some small section of the previous processed_signal to
# be fed in at each timestep - we initialize this to a tensor filled with zeros
# so that we will do zero-padding for the very first chunk(s)
num_channels = asr_model.cfg.preprocessor.features
cache_pre_encode = torch.zeros((1, num_channels, pre_encode_cache_size), device=asr_model.device)


# helper function for extracting transcriptions
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

# define functions to init audio preprocessor and to
# preprocess the audio (ie obtain the mel-spectrogram)
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

preprocessor = init_preprocessor(asr_model)

def preprocess_audio(audio, asr_model):
    device = asr_model.device

    # doing audio preprocessing
    audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
    audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
    processed_signal, processed_signal_length = preprocessor(
        input_signal=audio_signal, length=audio_signal_len
    )
    return processed_signal, processed_signal_length


def transcribe_chunk(new_chunk):
    
    global cache_last_channel, cache_last_time, cache_last_channel_len
    global previous_hypotheses, pred_out_stream, step_num
    global cache_pre_encode
    
    # new_chunk is provided as np.int16, so we convert it to np.float32
    # as that is what our ASR models expect
    audio_data = new_chunk.astype(np.float32)
    audio_data = audio_data / 32768.0

    # get mel-spectrogram signal & length
    processed_signal, processed_signal_length = preprocess_audio(audio_data, asr_model)
     
    # prepend with cache_pre_encode
    processed_signal = torch.cat([cache_pre_encode, processed_signal], dim=-1)
    processed_signal_length += cache_pre_encode.shape[1]
    
    # save cache for next time
    cache_pre_encode = processed_signal[:, :, -pre_encode_cache_size:]
    
    with torch.no_grad():
        (
            pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = asr_model.conformer_stream_step(
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
    
    return final_streaming_tran[0]

chunk_size = lookahead_size + ENCODER_STEP_LENGTH

audio, sr = librosa.load("tests/jfk.flac", sr=16000)
audio = (audio * 32768).astype(np.int16)
chunk_size_samples = int(sr * chunk_size / 1000)

states = None
for i in range(0, len(audio), chunk_size_samples):
    chunk = audio[i:i+chunk_size_samples]
    if len(chunk) < chunk_size_samples:
        chunk = np.pad(chunk, (0, chunk_size_samples - len(chunk)), 'constant')
    text = transcribe_chunk(chunk)
    text1, states = NeMoTranscriber.transcribe_chunk(chunk, states)
    print(text)
    print(text1)


