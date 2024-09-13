import random
import pandas as pd


DATA_ROOT_TARGET = "./data_raw"
DATA_FILE_NAME = "train.parquet"
TEST_DATA_FILE_NAME = "test.parquet"

DATA_FULL_PATH = DATA_ROOT_TARGET + '/' + DATA_FILE_NAME
TEST_DATA_FULL_PATH = DATA_ROOT_TARGET + '/' + TEST_DATA_FILE_NAME

qa_pairs =[]
for num in range(0, 4000):
    qa_pairs.append(["question "+str(num),"answer "+str(num)])

train_questions = []
train_answers = []

test_questions = []
test_answers = []

index = 0
for pairs in qa_pairs:
    test_indecies = random.sample(range(0, len(qa_pairs) - 1), int(len(qa_pairs) *0.1))
    if index in test_indecies:
        test_questions.append(pairs[0])
        test_answers.append(pairs[1])
    else:
        train_questions.append(pairs[0])
        train_answers.append(pairs[1])
    index = index + 1

# Create Train DataFrame
train_df = pd.DataFrame({
    'question': train_questions,
    'answer': train_answers
})

print("Saving " + str(len(train_questions)) + " data entries to: " + DATA_FULL_PATH)
# Save as Parquet file
train_df.to_parquet(DATA_FULL_PATH)

# Read back the Parquet file to verify
read_df = pd.read_parquet(DATA_FULL_PATH)

# Display first few rows
print(read_df.head())

# Display info about the dataset
print(read_df.info())

# Create Test DataFrame
test_df = pd.DataFrame({
    'question': test_questions,
    'answer': test_answers
})

print("Saving " + str(len(test_questions)) + " data entries to: " + TEST_DATA_FULL_PATH)
# Save as Parquet file
test_df.to_parquet(TEST_DATA_FULL_PATH)

# Read back the Parquet file to verify
read_df = pd.read_parquet(TEST_DATA_FULL_PATH)

# Display first few rows
print(read_df.head())

# Display info about the dataset
print(read_df.info())