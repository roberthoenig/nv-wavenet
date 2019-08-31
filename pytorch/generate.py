import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
# warnings.filterwarnings("ignore", category=UserWarning) 
import os
from scipy.io.wavfile import write
import torch
# import nv_wavenet
import utils
from time import time
import numpy as np

def chunker(seq, size):
    """
    https://stackoverflow.com/a/434328
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def main(audio_file_path, model_filename, output_path):
    model = torch.load(model_filename, map_location=torch.device('cpu'))['model']
    
    # mels = []
    # for file_path in files:
    #     print(file_path)
    #     mel = torch.load(file_path)
    #     mel = utils.to_gpu(mel)
    #     mels.append(torch.unsqueeze(mel, 0))
    # cond_input = model.get_cond_input(torch.cat(mels, 0))
    # audio_data = wavenet.infer(cond_input, implementation)
    first_audio_data, _ = utils.load_wav_to_torch(audio_file_path)
    first_audio_data = first_audio_data[:10000]
    first_audio_data = utils.mu_law_encode(first_audio_data / utils.MAX_WAV_VALUE, 256)
    print("first_audio_data.shape", first_audio_data.shape)
    print("first_audio_data.shape", first_audio_data.dtype)
    audio_data = model.generate(first_samples = first_audio_data, num_samples=1000)
    np.savetxt("audio_data.txt", audio_data.numpy().astype(int), fmt='%d')
    # for i, file_path in enumerate(files):
    # file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    audio = utils.mu_law_decode_numpy(audio_data.cpu().numpy(), model.n_out_channels)
    audio = utils.MAX_WAV_VALUE * audio
    print("audio: ", audio)
    wavdata = audio.astype('int16')
    write(output_path, 16000, wavdata)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file_path", default=None)
    parser.add_argument('-c', "--checkpoint_path", required=True)
    parser.add_argument('-o', "--output_path", required=True)
    
    args = parser.parse_args()
    main(args.file_path, args.checkpoint_path, args.output_path)
