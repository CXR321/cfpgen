import pickle

with open("data-bin/uniprotKB/cfpgen_general_dataset/test.pkl", "rb") as f:
    test_data = pickle.load(f)

with open("data-bin/uniprotKB/cfpgen_general_dataset/test.fasta", 'a') as fp_save:
    for index, row in enumerate(test_data):

        sequence = row['sequence']
        seq_id = row['uniprot_id']


        seq = sequence.replace(" ", "")
        fp_save.write(f">SEQUENCE_ID={seq_id}_L={len(seq)}\n")
        fp_save.write(f"{seq}\n")