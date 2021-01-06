import sys
import torch

def main(argv):
    
    # load all dataset in dict
    dataset = []
    for idx in range(1, len(argv)-1):
        items = torch.load(argv[idx])
        items = list(items.values())
        dataset += items
    
    merge_dct = {}
    for idx in range(len(dataset)):
        merge_dct[idx] = dataset[idx]
        
    torch.save(merge_dct, argv[-1])
    

if __name__ == "__main__":
    main(sys.argv)
    