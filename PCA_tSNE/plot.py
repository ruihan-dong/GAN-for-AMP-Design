import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from Bio import SeqIO

def get_conversion_table(path, norm=True):
    table_input = pd.read_csv(path, sep=" ", index_col=0)
    index = list(table_input.index)
    if norm:
        scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(table_input)
        table = {}
        for index, aa in enumerate(index):
            table[aa] = np.array(scaled[index])
    else:
        table = {}
        table_input = np.array(table_input.values)
        for index, aa in enumerate(index):
            table[aa] = np.array(table_input[index])
    table["X"] = [0] * 5
    return table


def read_fasta(fasta_path):
    fasta = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        fasta[str(record.id)] = str(record.seq)
    return fasta


def padding_seqs(seqs, length=None, pad_value=None):
    length = length or 30
    pad_value = pad_value or "X"
    data = {}
    for key, seq in seqs.items():
        if len(seq) <= length:
            data[key] = seq + pad_value * (length - len(seq))
        else:
            raise Exception("Length exceeds {}".format(length))
    return data


def encode(fasta, table):
    encoded_seqs = {}
    for key, seq in fasta.items():
        encoded_seqs[key] = [table[aa] for aa in list(seq)]
    return encoded_seqs


def get_encoded_seqs(seqs, table):
    seqs = padding_seqs(seqs, 30)
    encoded_seqs = encode(seqs, table)
    output_vectors = []
    for value in encoded_seqs.values():
        value = list(np.array(value).flatten())
        output_vectors.append(value)
    
    return output_vectors


def draw_tsne(ex, label, name, perp):
	tsne = TSNE(n_components=2, perplexity=perp, init='random', n_iter=1000, random_state=1234)
	tsne_ex = tsne.fit_transform(ex)
	colors = ['red', 'lightskyblue']

	plt.figure(figsize=(12,8))
	plt.xlim(tsne_ex[:,0].min() - 10, tsne_ex[:,0].max() + 10)
	plt.ylim(tsne_ex[:,1].min() - 10, tsne_ex[:,1].max() + 10)
	for i in range(len(tsne_ex)):
		plt.scatter(tsne_ex[i, 0], tsne_ex[i, 1], c=colors[int(label[i])], alpha=0.6)
	plt.tick_params(labelsize=20)
	labels = ['generate', 'real']
    # produce a legend with the unique colors from the scatter
	patches = [plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i], label="{:s}".format(labels[i]))[0] for i in range(len(colors))]
	plt.legend(handles=patches)

	plt.savefig(name + ".png", dpi=300)
	print("t-SNE figure saved:" + name)

def draw_pca(ex, label, name):
	pca = PCA(n_components=2)
	pca_ex = pca.fit_transform(ex)
	colors = ['red', 'lightskyblue']

	plt.figure(figsize=(12,8))
	plt.xlim(pca_ex[:,0].min() - 2, pca_ex[:,0].max() + 2)
	plt.ylim(pca_ex[:,1].min() - 2, pca_ex[:,1].max() + 2)
	for i in range(len(pca_ex)):
		plt.scatter(pca_ex[i, 0], pca_ex[i, 1], c=colors[int(label[i])], alpha=0.6)
		plt.legend()
	plt.tick_params(labelsize=20)
	labels = ['generate', 'real']
    # produce a legend with the unique colors from the scatter
	patches = [plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i], label="{:s}".format(labels[i]))[0] for i in range(len(colors))]
	plt.legend(handles=patches)

	plt.savefig(name + ".png", dpi=300)
	print("PCA figure saved:" + name)


def tsne_each(ex, label, name, perp):
    tsne = TSNE(n_components=2, perplexity=perp, init='random', n_iter=1000, random_state=1234)
    tsne_ex = tsne.fit_transform(ex)
    colors = ['lightskyblue', 'royalblue', 'violet', 'limegreen', 'darkorange', 'red']

    plt.figure(figsize=(12,8))
    plt.xlim(tsne_ex[:,0].min() - 10, tsne_ex[:,0].max() + 10)
    plt.ylim(tsne_ex[:,1].min() - 10, tsne_ex[:,1].max() + 10)
    for i in range(len(tsne_ex)):
        plt.scatter(tsne_ex[i, 0], tsne_ex[i, 1], c=colors[int(label[i])], alpha=0.6)
    plt.tick_params(labelsize=20)
    labels = ['real', 'generate at epoch 1', 'generate at epoch 100', 'generate at epoch 200', 'generate at epoch 400', 'generate at epoch 800']
    # produce a legend with the unique colors from the scatter
    patches = [plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i], label="{:s}".format(labels[i]))[0] for i in range(len(colors))]
    plt.legend(handles=patches, fontsize=15)

    plt.savefig(name + ".png", dpi=300)
    print("t-SNE figure saved:" + name)

def pca_each(ex, label, name):
    pca = PCA(n_components=2)
    pca_ex = pca.fit_transform(ex)
    colors = ['lightskyblue', 'royalblue', 'violet', 'limegreen', 'darkorange', 'red']

    plt.figure(figsize=(12,8))
    plt.xlim(pca_ex[:,0].min() - 2, pca_ex[:,0].max() + 2)
    plt.ylim(pca_ex[:,1].min() - 2, pca_ex[:,1].max() + 2)
    for i in range(len(pca_ex)):
        plt.scatter(pca_ex[i, 0], pca_ex[i, 1], c=colors[int(label[i])], alpha=0.6)
        plt.legend()
    plt.tick_params(labelsize=20)
    labels = ['real', 'generate at epoch 1', 'generate at epoch 100', 'generate at epoch 200', 'generate at epoch 400', 'generate at epoch 800']
    # produce a legend with the unique colors from the scatter
    patches = [plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i], label="{:s}".format(labels[i]))[0] for i in range(len(colors))]
    plt.legend(handles=patches, fontsize=15)

    plt.savefig(name + ".png", dpi=300)
    print("PCA figure saved:" + name)



if __name__ == "__main__" :
    table_path = "AAF.txt" #AAF.txt
    table = get_conversion_table(table_path, norm=False)
    seqs = read_fasta('pepvae.fasta')
    generated_seqs = read_fasta('generated_seq.fasta')
    
    train_seqs = {}
    for id_, seq in seqs.items():
        if len(seq) < 30:
            train_seqs[id_] = seq
    train_encoded = get_encoded_seqs(train_seqs, table)
    '''
    generate_encoded = get_encoded_seqs(generated_seqs, table)

    labels = list(np.concatenate((np.ones(len(train_seqs)), np.zeros(len(generated_seqs))), axis=0))
    seqlist = train_encoded + generate_encoded

    draw_tsne(seqlist, labels, 'tsne-30-new', 30)
    draw_pca(seqlist, labels, 'pca-new')
    '''
    # draw for each 200 epoches
    epoches = ['1', '100', '200', '400', '800']
    seqlist = train_encoded
    labels = list((np.zeros(len(train_seqs))))
    for i in range(len(epoches)):
        seq = read_fasta('epoch_'+epoches[i]+'_generated_seq.fasta')
        encoded_seq = get_encoded_seqs(seq, table)
        seqlist = seqlist + encoded_seq
        label =  (i + 1) * np.ones(len(encoded_seq))
        labels = labels + list(label)

    tsne_each(seqlist, labels, 'tsne-30-each-all', 30)
    pca_each(seqlist, labels, 'pca-each-all')
  
