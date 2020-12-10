from dataset import RainforestDataset
import matplotlib.pyplot as plt
import sys

def main(manifest, audio_dir):
    manifest = manifest
    audio_dir = audio_dir
    chunk = 100
    dataset = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=True, feat_type="fbank",  num_mel_bins=40, chunk=chunk)

    frame_shift = 0.01 # second
    species_id_count = {}
    species_chunk_length_count = {}
    for i in range(len(dataset)):
        _, feats, y = dataset[i]
        y = y.tolist()
        chunk_length = feats.shape[0] * chunk * frame_shift
        species_id = [i for i in range(len(y)) if y[i] != 0][0]

        try:
            species_id_count[species_id] += 1
        except:
            species_id_count[species_id] = 1
        
        try:
            species_chunk_length_count[species_id] += chunk_length
        except:
            species_chunk_length_count[species_id] = chunk_length


    keys = sorted(species_id_count.keys())
    species_id = [species_id_count[i] for i in keys ]
    chunk_length = [species_chunk_length_count[i] for i in keys ]
    
    fig, ax = plt.subplots(1, 3)
    ax[0].pie(species_id, labels=keys, autopct="%1.1f%%")
    ax[0].axis("equal")
    ax[0].set_title("Species distribution")
    ax[1].bar(keys, species_id)
    ax[1].set_title("Species count")
    ax[2].bar(keys, chunk_length)
    ax[2].set_title("Species audio length (chunk)")

    fig.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

