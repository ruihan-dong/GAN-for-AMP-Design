# load trained GAN generator to produce seqs

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch

from utils import *
from model import Generator

def main(fasta_path=None, output_root=None, batch_size=None, epoch=None, step=None):
    batch_size = batch_size or 128
    epoch = epoch or 10000
    step = step or 100
    latent_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table_path = './data/AAF.txt'
    model_path = './model/generator.pkl'

    table = get_conversion_table(table_path, norm=False)

    # generator = Generator(latent_size, 5, 64).to(device)
    # generator.load_state_dict(torch.load(model_path, map_location=device))
    generator = torch.load(model_path, map_location=device)
    generator.eval()

    noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
    generated_seqs = generate_seqs(generator, table, noise)
    write_fasta_remove(generated_seqs, "./output/b" + str(batch_size) + "_generated_seqs.fasta")

def remove_polyA(seq):
    for idx in range(len(seq)):
        if seq[idx:idx+3] == 'AAA':
            seq = seq[:idx]
    while seq[-1] == 'A':
        seq = seq[:-1]
    return seq

def write_fasta_remove(seqs, path):
    with open(path, "w") as output:
        for name, seq in seqs.items():
            seq = remove_polyA(seq)
            output.write(">{}\n{}\n".format(name, seq))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This program would train the GAN model with user-provided "
                                     "sequences and save the result in thr output folder. "
                                     "Use -h to get more help.")
    parser.add_argument("-f","--fasta_path",
                        help="The path of the peptide file in FASTA format")
    parser.add_argument("-o","--output_root",
                        help="The path of the folder where result is saved")
    parser.add_argument("-b","--batch_size",type=int,
                        help="The batch size of the data. The default value is 128.")
    parser.add_argument("-e","--epoch",type=int,
                        help="The epoch of the training process. The default value is 10000.")
    parser.add_argument("-s","--step",type=int,
                        help="The number of epoch to save the temporary result. The default value is 100.")
    args = vars(parser.parse_args())
    
    main(**args)
