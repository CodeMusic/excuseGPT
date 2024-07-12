import os
import random
import numpy as np
import pickle
import tiktoken
import argparse

char_encodings_dir = os.path.join(os.path.dirname(__file__), 'chats_char')
token_encodings_dir = os.path.join(os.path.dirname(__file__), 'chats_token')
memoryData = ""

def gatherchats(directoryPath, shuffle):
    
    allchats = ""
    fileList = os.listdir(directoryPath)
    if shuffle:
        print("Shuffling files...")
        random.shuffle(fileList)  # Shuffle the list of files randomly
    else:
        print("Sorting files...")
        fileList = sorted(fileList)

    for fileName in fileList:
        if fileName.endswith(".txt"):
            filePath = os.path.join(directoryPath, fileName)
            with open(filePath, 'r', encoding='utf-8') as file:
                allchats += file.read() + "\n"

    return allchats


def save_bin_files(encoding_ids, directory, filename):
    
    encoding_ids = np.array(encoding_ids, dtype=np.uint16)
    encoding_ids.tofile(os.path.join(directory, filename))
    print(f"Saved {len(encoding_ids):,} tokens to {os.path.join(directory, filename)}")

def process_token_encodings(train_data, val_data):
    enc = tiktoken.get_encoding("gpt2") #("gpt2")

    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)

    save_bin_files(train_ids, token_encodings_dir, 'train.bin')
    save_bin_files(val_ids, token_encodings_dir, 'val.bin')

    print(f"");
    print(f"Token encoding - Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")

def process_char_encodings(train_data, val_data):
    chars = sorted(list(set(train_data + val_data)))
    stoi = {ch: i for i, ch in enumerate(chars)} # stoi = string to index
    itos = {i: ch for i, ch in enumerate(chars)} # itos = index to string

    train_ids = [stoi[c] for c in train_data] # train_ids = train_data encoded as indices
    val_ids = [stoi[c] for c in val_data] # val_ids = val_data encoded as indices

    save_bin_files(train_ids, char_encodings_dir, 'train.bin') 
    save_bin_files(val_ids, char_encodings_dir, 'val.bin')

    meta = {'vocab_size': len(chars), 'itos': itos, 'stoi': stoi} # meta = dictionary containing the vocab size, itos, and stoi
    with open(os.path.join(char_encodings_dir, 'meta.pkl'), 'wb') as f: # save the meta information to a pickle file
        pickle.dump(meta, f)

    print(f"");
    print(f"Character Vocab: {''.join(chars)}")
    print(f"Character encoding - Vocab size: {len(chars):,}, Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids)}")

def updateEncodings(shuffle):

    os.makedirs(char_encodings_dir, exist_ok=True)
    os.makedirs(token_encodings_dir, exist_ok=True)
    
    chatsDirectory = "chats"
    memoryData = gatherchats(chatsDirectory, shuffle)

    n = len(memoryData)
    train_data = memoryData[:int(n*0.9)] # 90% of the data for training
    val_data = memoryData[int(n*0.9):] # 10% of the data for validation

    process_char_encodings(train_data, val_data)
    process_token_encodings(train_data, val_data)
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and encode text data.")
    parser.add_argument('-shuffle', action='store_true', help='Shuffle the files before processing')
    args = parser.parse_args()

    updateEncodings(args.shuffle)

