import requests

uids = ["A7F996"] * 200
query = " OR ".join([f"accession:{u}" for u in uids])
url = f"https://rest.uniprot.org/uniprotkb/search"
params = {
    "query": query,
    "fields": "accession,xref_pdb",
    "size": 500
}
res = requests.get(url, params=params)
print(res.status_code)
print(res.text)
