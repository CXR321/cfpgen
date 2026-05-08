import re
lora_target_module = "(esm.encoder.layer.[0-9]*.attention.(self.query|self.key|self.value|output.dense).*|esm.encoder.layer.[0-9]*.(intermediate|output).dense.*)"

names = [
    "base_model.model.esm.encoder.layer.0.attention.self.query.base_layer.weight",
    "base_model.model.esm.encoder.layer.0.attention.self.query.lora_A.default.weight",
    "base_model.model.esm.encoder.layer.0.attention.self.query.lora_B.default.weight",
    "base_model.model.esm.encoder.layer.0.attention.output.LayerNorm.weight",
    "base_model.model.esm.encoder.layer.0.attention.output.dense.base_layer.weight",
    "base_model.model.esm.encoder.layer.0.attention.output.dense.lora_A.default.weight"
]

for name in names:
    if 'lora_' in name:
        print(f"TRUE (lora): {name}")
    elif re.search(lora_target_module, name):
        print(f"FALSE (base): {name}")
    else:
        print(f"TRUE (other): {name}")
