import requests
uids = [f"P8103{i}" for i in range(200)]
query = "accession:(" + " OR ".join(uids) + ")"
url = f"https://rest.uniprot.org/uniprotkb/search"
params = {
    "query": query,
    "fields": "accession,xref_pdb",
    "size": 500
}
res = requests.get(url, params=params)
print(res.status_code)
if res.status_code != 200:
    print(res.text)
else:
    print("Success! Number of results:", len(res.json().get('results', [])))
