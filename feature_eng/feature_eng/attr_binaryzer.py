from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Set

class AttrBinaryzer:
    def __init__(self, codes):
        self.mlb = None
        self.createBinaryRepresentation(codes)
        

    def createBinaryRepresentation(self, codes):
        self.mlb = MultiLabelBinarizer(classes=codes)
        self.mlb.fit([codes])
    
    def transform(self, codes):    
        if isinstance(codes, Set):
            codes = list(codes)    
        # Trasforma i livelli in un vettore binario
        return self.mlb.transform([codes])[0]

