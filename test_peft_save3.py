from safetensors.torch import load_file
st = load_file("test_peft_out/adapter_model.safetensors")
for k in st.keys():
    print("  ", k)
