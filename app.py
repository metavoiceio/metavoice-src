import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import gradio as gr
from huggingface_hub import snapshot_download

from fam.llm.sample import (
    InferenceConfig,
    SamplingControllerConfig,
    build_models,
    get_first_stage_path,
    get_second_stage_path,
    sample_utterance,
)
from fam.llm.utils import check_audio_file

#### setup model
sampling_config = SamplingControllerConfig(
    huggingface_repo_id="metavoiceio/metavoice-1B-v0.1", spk_cond_path=""
)  # spk_cond_path added later
model_dir = snapshot_download(repo_id=sampling_config.huggingface_repo_id)
first_stage_ckpt_path = get_first_stage_path(model_dir)
second_stage_ckpt_path = get_second_stage_path(model_dir)

config_first_stage = InferenceConfig(
    ckpt_path=first_stage_ckpt_path,
    num_samples=sampling_config.num_samples,
    seed=sampling_config.seed,
    device=sampling_config.device,
    dtype=sampling_config.dtype,
    compile=sampling_config.compile,
    init_from=sampling_config.init_from,
    output_dir=sampling_config.output_dir,
)

config_second_stage = InferenceConfig(
    ckpt_path=second_stage_ckpt_path,
    num_samples=sampling_config.num_samples,
    seed=sampling_config.seed,
    device=sampling_config.device,
    dtype=sampling_config.dtype,
    compile=sampling_config.compile,
    init_from=sampling_config.init_from,
    output_dir=sampling_config.output_dir,
)

sampling_config.max_new_tokens *= 2  # deal with max_new_tokens for flattened interleaving!

# define models
smodel, llm_first_stage, llm_second_stage = build_models(
    config_first_stage,
    config_second_stage,
    model_dir=model_dir,
    device=sampling_config.device,
    use_kv_cache=sampling_config.use_kv_cache,
)

#### setup interface
RADIO_CHOICES = ["Preset voices", "Upload target voice (atleast 30s)"]
MAX_CHARS = 220
PRESET_VOICES = {
    # female
    "Bria": "https://cdn.themetavoice.xyz/speakers%2Fbria.mp3",
    # male
    "Alex": "https://cdn.themetavoice.xyz/speakers/alex.mp3",
    "Jacob": "https://cdn.themetavoice.xyz/speakers/jacob.wav",
}


def denormalise_top_p(top_p):
    # returns top_p in the range [0.9, 1.0]
    return round(0.9 + top_p / 100, 2)


def denormalise_guidance(guidance):
    # returns guidance in the range [1.0, 3.0]
    return 1 + ((guidance - 1) * (3 - 1)) / (5 - 1)


def _check_file_size(path):
    if not path:
        return
    filesize = os.path.getsize(path)
    filesize_mb = filesize / 1024 / 1024
    if filesize_mb >= 50:
        raise gr.Error(f"Please upload a sample less than 20MB for voice cloning. Provided: {round(filesize_mb)} MB")


def _handle_edge_cases(to_say, upload_target):
    if not to_say:
        raise gr.Error("Please provide text to synthesise")

    if len(to_say) > MAX_CHARS:
        gr.Warning(
            f"Max {MAX_CHARS} characters allowed. Provided: {len(to_say)} characters. Truncating and generating speech...Result at the end can be unstable as a result."
        )

    if not upload_target:
        return

    check_audio_file(upload_target)  # check file duration to be atleast 30s
    _check_file_size(upload_target)


def tts(to_say, top_p, guidance, toggle, preset_dropdown, upload_target):
    try:
        d_top_p = denormalise_top_p(top_p)
        d_guidance = denormalise_guidance(guidance)

        _handle_edge_cases(to_say, upload_target)

        to_say = to_say if len(to_say) < MAX_CHARS else to_say[:MAX_CHARS]
        return sample_utterance(
            to_say,
            spk_cond_path=PRESET_VOICES[preset_dropdown] if toggle == RADIO_CHOICES[0] else upload_target,
            spkemb_model=smodel,
            first_stage_model=llm_first_stage,
            second_stage_model=llm_second_stage,
            enhancer=sampling_config.enhancer,
            guidance_scale=(d_guidance, 1.0),
            max_new_tokens=sampling_config.max_new_tokens,
            temperature=sampling_config.temperature,
            top_k=sampling_config.top_k,
            top_p=d_top_p,
            first_stage_ckpt_path=None,
            second_stage_ckpt_path=None,
        )
    except Exception as e:
        raise gr.Error(f"Something went wrong. Reason: {str(e)}")


def change_voice_selection_layout(choice):
    if choice == RADIO_CHOICES[0]:
        return [gr.update(visible=True), gr.update(visible=False)]

    return [gr.update(visible=False), gr.update(visible=True)]


title = """
<picture>
  <source srcset="https://cdn.themetavoice.xyz/banner_light_transparent.png" media="(prefers-color-scheme: dark)" />
  <img alt="MetaVoice logo" src="https://cdn.themetavoice.xyz/banner_light_transparent.png" style="width: 20%; margin: 0 auto;" />
</picture>

\n# TTS by MetaVoice-1B
"""

description = """
<strong>MetaVoice-1B</strong> is a 1.2B parameter base model for TTS (text-to-speech). It has been built with the following priorities:
\n
* <strong>Emotional speech rhythm and tone</strong> in English.
* <strong>Zero-shot cloning for American & British voices</strong>, with 30s reference audio.
* Support for <strong>voice cloning with finetuning</strong>.
  * We have had success with as little as 1 minute training data for Indian speakers.
* Support for <strong>long-form synthesis</strong>.

We are releasing the model under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). See [Github](https://github.com/metavoiceio/metavoice-src) for details and to contribute.
"""

with gr.Blocks(title="TTS by MetaVoice") as demo:
    gr.Markdown(title)

    with gr.Row():
        gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            to_say = gr.TextArea(
                label=f"What should I say!? (max {MAX_CHARS} characters).",
                lines=4,
                value="This is a demo of text to speech by MetaVoice-1B, an open-source foundational audio model by MetaVoice.",
            )
            with gr.Row(), gr.Column():
                # voice settings
                top_p = gr.Slider(
                    value=5.0,
                    minimum=0.0,
                    maximum=10.0,
                    step=1.0,
                    label="Speech Stability - improves text following for a challenging speaker",
                )
                guidance = gr.Slider(
                    value=5.0,
                    minimum=1.0,
                    maximum=5.0,
                    step=1.0,
                    label="Speaker similarity - How closely to match speaker identity and speech style.",
                )

                # voice select
                toggle = gr.Radio(choices=RADIO_CHOICES, label="Choose voice", value=RADIO_CHOICES[0])

            with gr.Row(visible=True) as row_1:
                preset_dropdown = gr.Dropdown(
                    PRESET_VOICES.keys(), label="Preset voices", value=list(PRESET_VOICES.keys())[0]
                )
                with gr.Accordion("Preview: Preset voices", open=False):
                    for label, path in PRESET_VOICES.items():
                        gr.Audio(value=path, label=label)

            with gr.Row(visible=False) as row_2:
                upload_target = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Upload a clean sample to clone. Sample should contain 1 speaker, be between 30-90 seconds and not contain background noise.",
                )

            toggle.change(
                change_voice_selection_layout,
                inputs=toggle,
                outputs=[row_1, row_2],
            )

        with gr.Column():
            speech = gr.Audio(
                type="filepath",
                label="MetaVoice-1B says...",
            )

    submit = gr.Button("Generate Speech")
    submit.click(
        fn=tts,
        inputs=[to_say, top_p, guidance, toggle, preset_dropdown, upload_target],
        outputs=speech,
    )


demo.queue()
demo.launch(
    favicon_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/favicon.ico"),
    server_name="0.0.0.0",
    server_port=7861,
)
