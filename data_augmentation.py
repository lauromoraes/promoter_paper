import numpy as np

class DataAug(object):
    def __init__(self):
        pass
    def switch_nucs(self, dataset, switch_ratio=.5):
        # Creates new random sequences based on a samples.
        # Switches random nucleotides.
        # Maintains same frequency distributions of nucletides
        new_seqs = []
        seq_length = len(dataset[0])
        num_switches = int(switch_ratio * seq_length)
        # Iterate over each sequence
        for i, sample in enumerate(dataset):
            nucs = [x for x in sample] # Decode characters to list
            for j in range(num_switches):
                # Select two distinct index
                i_x = np.random.randint(seq_length)
                i_y = np.random.randint(seq_length)
                while i_x == i_y:
                    i_y = np.random.randint(seq_length)
                # Switch nucs
                nucs[i_x], nucs[i_y] = nucs[i_y], nucs[i_x]
            new_seq = ''.join(nucs) # Encode back to str
            new_seqs.append(new_seq)
        return new_seqs
