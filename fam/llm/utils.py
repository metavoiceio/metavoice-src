import re


def normalize_text(text: str) -> str:
    unicode_conversion = {
        8175: "'",
        8189: "'",
        8190: "'",
        8208: "-",
        8209: "-",
        8210: "-",
        8211: "-",
        8212: "-",
        8213: "-",
        8214: "||",
        8216: "'",
        8217: "'",
        8218: ",",
        8219: "`",
        8220: '"',
        8221: '"',
        8222: ",,",
        8223: '"',
        8228: ".",
        8229: "..",
        8230: "...",
        8242: "'",
        8243: '"',
        8245: "'",
        8246: '"',
        180: "'",
        2122: "TM",  # Trademark
    }

    text = text.translate(unicode_conversion)

    non_bpe_chars = set([c for c in list(text) if ord(c) >= 256])
    if len(non_bpe_chars) > 0:
        non_bpe_points = [(c, ord(c)) for c in non_bpe_chars]
        raise ValueError(f"Non-BPE single token characters found: {non_bpe_points}")

    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = text.replace("*", " ")
    text = text.strip()
    text = re.sub("\s\s+", " ", text)  # remove multiple spaces
    return text
