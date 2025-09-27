from bioscan_dataset import BIOSCAN5M
from transformers import AutoTokenizer, AutoModel
import numpy as np
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bioscan-ml/BarcodeBERT", trust_remote_code=True
)

# Load the model
model = AutoModel.from_pretrained("bioscan-ml/BarcodeBERT", trust_remote_code=True)

def dnafeature(dna_seq):
    # Tokenize
    input_seq = tokenizer(dna_seq, return_tensors="pt")["input_ids"]

    # Pass through the model
    output = model(input_seq.unsqueeze(0))["hidden_states"][-1]

    # Compute Global Average Pooling
    features = output.mean(1)

    return features.detach().numpy()

def m():
    dataset = BIOSCAN5M(
        root="/mnt/big/datasets/bioscan5m",
        download=True,
        target_type="species",
        target_format="text",
    )

    # n = 500
    embeddings = []
    for i_ in range(1000):
        image, dna_barcode, label = dataset[i_]
        image.save(f"/mnt/big/datasets/bioannflux/images/image_{i_}.jpg")
        print(image, label)
        embeddings.append(dnafeature(dna_barcode))
        # image.show()
        # n += 1
        pass

    x = np.vstack(embeddings)
    print(x.shape)
    np.savez("/mnt/big/datasets/bioannflux/features/embeddings.npz", lastFull=x)


if __name__ == "__main__":
    m()
