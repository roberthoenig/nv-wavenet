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

def chunker(seq, size):
    """
    https://stackoverflow.com/a/434328
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def main(mel_files, model_filename, output_dir):
    mel_files = utils.files_to_list(mel_files)
    model = torch.load(model_filename)['model']
    # wavenet = nv_wavenet.NVWaveNet(**(model.export_weights()))
    
    for file_path in mel_files:
        # mels = []
        # for file_path in files:
        #     print(file_path)
        #     mel = torch.load(file_path)
        #     mel = utils.to_gpu(mel)
        #     mels.append(torch.unsqueeze(mel, 0))
        # cond_input = model.get_cond_input(torch.cat(mels, 0))
        # audio_data = wavenet.infer(cond_input, implementation)
        audio_data = model.generate(num_samples=6000)

        # for i, file_path in enumerate(files):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        audio = utils.mu_law_decode_numpy(audio_data.cpu().numpy(), model.n_out_channels)
        audio = utils.MAX_WAV_VALUE * audio
        print("audio: ", audio)
        wavdata = audio.astype('int16')
        write("{}/{}.wav".format(output_dir, file_name),
                16000, wavdata)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', "--checkpoint_path", required=True)
    parser.add_argument('-o', "--output_dir", required=True)
    
    args = parser.parse_args()
    main(args.filelist_path, args.checkpoint_path, args.output_dir)
