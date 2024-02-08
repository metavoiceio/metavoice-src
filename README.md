# MetaVoice-1B

MetaVoice-1B is a 1.2B parameter base model trained on 100K hours of speech for TTS (text-to-speech). It has been built with the following priorities:
* **Emotional speech rhythm and tone** in English. No hallucinations.
* **Zero-shot cloning for American & British voices**, with 30s reference audio.
* Support for (cross-lingual) **voice cloning with finetuning**.
  * We have had success with as little as 1 minute training data for Indian speakers.
* Support for **long-form synthesis**.

We’re releasing MetaVoice-1B under the Apache 2.0 license, *it can be used without restrictions*.

Try out the [demo](https://ttsdemo.themetavoice.xyz/)!

## Installation
```bash
# install ffmpeg
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz.md5
md5sum -c ffmpeg-git-amd64-static.tar.xz.md5
tar xvf ffmpeg-git-amd64-static.tar.xz
sudo mv ffmpeg-git-*-static/ffprobe ffmpeg-git-*-static/ffmpeg /usr/local/bin/
rm -rf ffmpeg-git-*

pip install -r requirements.txt
pip install -e .
```

## Usage
1. Download it and use it anywhere (including locally) with our [reference implementation](/fam/llm/sample.py),
```bash
python fam/llm/sample.py --huggingface_repo_id="metavoiceio/metavoice-1B-v0.1" --spk_cond_path="assets/bria.mp3"
```

2. Deploy it on any cloud (AWS/GCP/Azure), using our [inference server](/fam/llm/serving.py)
```bash
python fam/llm/serving.py --huggingface_repo_id="metavoiceio/metavoice-1B-v0.1"
```

3. Use it via [HuggingFace](https://huggingface.co/metavoiceio)

## Soon
- Long form TTS
- Fine-tuning code

## Architecture
We predict EnCodec tokens from text, and speaker information. This is then diffused up to the waveform level, with post-processing applied to clean up the audio.

* We use a causal GPT to predict the first two hierarchies of EnCodec tokens. Text and audio are part of the LLM context. Speaker information is passed via conditioning at the token embedding layer. This speaker conditioning is obtained from a separately trained speaker verification network.
  - The two hierarchies are predicted in a "flattened interleaved" manner, we predict the first token of the first hierarchy, then the first token of the second hierarchy, then the second token of the first hierarchy, and so on.
  - We use condition-free sampling to boost the cloning capability of the model.
  - The text is tokenised using a custom trained BPE tokeniser with 512 tokens.
  - Note that we've skipped predicting semantic tokens as done in other works, as we found that this isn't strictly necessary.
* We use a non-causal (encoder-style) transformer to predict the rest of the 6 hierarchies from the first two hierarchies. This is a super small model (~10Mn parameters), and has extensive zero-shot generalisation to most speakers we've tried. Since it's non-causal, we're also able to predict all the timesteps in parallel.
* We use multi-band diffusion to generate waveforms from the EnCodec tokens. We noticed that the speech is clearer than using the original RVQ decoder or VOCOS. However, the diffusion at waveform level leaves some background artifacts which are quite unpleasant to the ear. We clean this up in the next step.
* We use DeepFilterNet to clear up the artifacts introduced by the multi-band diffusion. 

## Optimizations
The model supports: 
1. KV-caching via Flash Decoding 
2. Batching (including texts of different lengths)

## Contribute
- See all [active issues](https://github.com/metavoiceio/metavoice-src/issues)!

## Acknowledgements
We are grateful to Together.ai for their 24/7 help in marshalling our cluster. We thank the teams of AWS, GCP & HuggingFace for support with their cloud platforms.
