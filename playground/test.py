import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import random


# Function to generate random biological questions and answers
def generate_bio_qa():
    questions = [
        "What is the function of DNA?",
        "How does photosynthesis work?",
        "What is the role of mitochondria in cells?",
        "Explain the process of protein synthesis.",
        "What is the difference between prokaryotic and eukaryotic cells?",
        "How do enzymes catalyze chemical reactions?",
        "What is the structure of a phospholipid bilayer?",
        "Describe the stages of mitosis.",
        "What is the role of ATP in cellular metabolism?",
        "How does natural selection contribute to evolution?"
    ]

    answers = [
        "DNA stores and transmits genetic information, directing cellular functions and inheritance.",
        "Photosynthesis converts light energy into chemical energy, producing glucose from CO2 and water.",
        "Mitochondria are the powerhouses of cells, generating ATP through cellular respiration.",
        "Protein synthesis involves transcription of DNA to mRNA and translation of mRNA to amino acids.",
        "Prokaryotes lack a nucleus, while eukaryotes have a membrane-bound nucleus and organelles.",
        "Enzymes lower activation energy, increasing reaction rates without being consumed.",
        "Phospholipid bilayers consist of hydrophilic heads and hydrophobic tails, forming cell membranes.",
        "Mitosis stages: prophase, metaphase, anaphase, and telophase, followed by cytokinesis.",
        "ATP stores and transfers energy for cellular processes like metabolism and biosynthesis.",
        "Natural selection favors beneficial traits, leading to adaptation and speciation over time."
    ]

    return random.choice(questions), random.choice(answers)


def generate_bio_qa_2():
    qa_pairs = [
        ("What is the function of DNA?",
         """DNA (Deoxyribonucleic Acid) has several key functions:
         1. Storage of genetic information
         2. Transmission of genetic information to offspring
         3. Directing the synthesis of proteins
         4. Regulation of gene expression
         DNA's structure allows it to replicate and pass on genetic traits."""),

        ("How does photosynthesis work?",
         """Photosynthesis is a complex process that can be summarized in these steps:
         1. Light absorption by chlorophyll
         2. Excitation of electrons in chlorophyll
         3. Electron transport chain and ATP production
         4. Carbon fixation in the Calvin cycle
         5. Production of glucose from CO2 and water
         This process converts light energy into chemical energy stored in glucose.""")
    ]
    return random.choice(qa_pairs)


# The rest of the code remains the same

# Generate dataset
num_samples = 1000
questions = []
answers = []

for _ in range(num_samples):
    q, a = generate_bio_qa_2()
    questions.append(q)
    answers.append(a)

# Create DataFrame
df = pd.DataFrame({
    'question': questions,
    'answer': answers
})

# Save as Parquet file
df.to_parquet('biology_qa_dataset.parquet')

# Read back the Parquet file to verify
read_df = pd.read_parquet('biology_qa_dataset.parquet')

# Display first few rows
print(read_df.head())

# Display info about the dataset
print(read_df.info())

