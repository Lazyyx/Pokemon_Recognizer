from datasets import load_dataset

ds = load_dataset("keremberke/pokemon-classification", name="full")
example = ds['train'][0]
print(example)