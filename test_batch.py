import requests

uids = ["A7F996", "P81031", "P04637"]
query = " OR ".join([f"accession:{u}" for u in uids])
url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&fields=accession,xref_pdb&size=500"

res = requests.get(url)
print(res.json())
