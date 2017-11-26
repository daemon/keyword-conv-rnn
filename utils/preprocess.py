"""
Preprocesses the AISHELL-1 dataset by converting all characters to pinyin
"""
import argparse
import os
import re

from tqdm import tqdm
import xpinyin

def preprocess(transcript, output):
    data = {}
    translator = xpinyin.Pinyin()
    with open(transcript) as f:
        lines = f.readlines()
    for line in tqdm(lines):
        key, chars = line.split(" ", 1)
        pinyin = translator.get_pinyin(chars.replace("\n", ""), " ", show_tone_marks=True)
        pinyin = re.sub("\s+", " ", pinyin.strip())
        data[key] = pinyin

    with open(output, "w") as f:
        for k, v in data.items():
            f.write("{} {}\n".format(k, v))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="aishell_transcript_v0.8.txt")
    parser.add_argument("--output", type=str, default="pinyin_transcript.txt")
    args = parser.parse_args()
    preprocess(args.file, args.output)

if __name__ == "__main__":
    main()