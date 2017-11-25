"""
Preprocesses the AISHELL-1 dataset by converting all characters to pinyin
"""
import argparse
import os

from tqdm import tqdm
import xpinyin

def get_txt_files(directory):
    files = []
    for fname in os.listdir(directory):
        if not fname.endswith("txt"):
            continue
        files.append(os.path.join(directory, fname))
    return files

def preprocess(directory):
    files = []
    for fname in os.listdir(directory):
        fname = os.path.join(directory, fname)
        if os.path.isfile(fname):
            continue
        for fname2 in os.listdir(fname):
            fname2 = os.path.join(fname, fname2)
            if os.path.isfile(fname2):
                continue
            files.extend(get_txt_files(fname2))
    
    translator = xpinyin.Pinyin()
    for fname in tqdm(files):
        with open(fname) as f:
            content = f.read().replace("\n", "")
        pinyin = translator.get_pinyin(content, " ", show_tone_marks=True)
        with open(fname.split(".")[0] + ".pinyin", "w") as f:
            f.write(pinyin)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data")
    args = parser.parse_args()
    preprocess(args.dir)

if __name__ == "__main__":
    main()