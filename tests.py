from ml_data import (SequenceNucsData, SequenceNucHotvector)

organism = 'Bacillus'
npath, ppath = './fasta/{}_neg.fa'.format(organism), './fasta/{}_pos.fa'.format(organism)

s1 = SequenceNucHotvector(npath, ppath)
s2 = SequenceNucsData(npath, ppath, k=2)

X1, y1 = s1.getX(), s1.getY()
X2, y2 = s1.getX(), s1.getY()
