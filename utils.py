import re

def clean_utterance(utterance):
    utterance = utterance.lower()
    utterance = re.sub(r'[^a-z\s]', '', utterance)
    utterance = ' '.join(utterance.split())
    return utterance


def split_utterance(utterance):
    parts = re.split(r'[,.]|--|-', utterance)
    parts = [clean_utterance(elem) for elem in parts]
    parts = [elem for elem in parts if elem]
    return parts
