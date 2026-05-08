import re

regex = r"(esm.encoder.layer.[0-9]*.attention.(self.query|self.key|self.value|output.dense).*|esm.encoder.layer.[0-9]*.(intermediate|output).dense.*)"
name = "base_model.model.esm.encoder.layer.0.attention.self.query.weight"

if re.search(regex, name):
    print("MATCH!")
else:
    print("NO MATCH!")

name2 = "base_model.model.esm.encoder.layer.0.attention.self.query.lora_A.default.weight"
if re.search(regex, name2):
    print("MATCH 2!")
    
name3 = "esm.encoder.layer.0.attention.output.LayerNorm.weight"
if re.search(regex, name3):
    print("MATCH 3 (should not match)!")
else:
    print("NO MATCH 3 (correct)!")
