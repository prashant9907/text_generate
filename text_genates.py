import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = "cpu"
tokenizer_para = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model_para = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

st.title("Story Generation")

# Create an input field to enter the model name or path
model_name = st.text_input("Model Name or Path", "gpt2")

# Load the selected model
@st.cache(allow_output_mutation=True)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained("Prashant-karwasra/GPT2_story_generation_model")
    model = AutoModelForCausalLM.from_pretrained("Prashant-karwasra/GPT2_story_generation_model")
    return tokenizer, model

def paraphrase(
        question,
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=1,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128
):
    input_ids = tokenizer_para(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids

    outputs = model_para.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )
    res = tokenizer_para.batch_decode(outputs, skip_special_tokens=True)
    return res


if model_name:
    tokenizer, model = load_model(model_name)
    st.write(f"Model '{model_name}' loaded successfully.")

    # Create a text input widget for user input
    user_input = st.text_area("Enter the prompt here...")

    # Create a button to trigger text generation
    if st.button("Generate Text"):
        if user_input:
            input_ids = tokenizer.encode(user_input, return_tensors="pt")
            output = model.generate(input_ids, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2)

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            st.write("Generated Text:")
            st.write(generated_text)

            text = generated_text
            text = text.split(".")
            # paraphrase(text)
            output_text = ""
            for sentence in text:
                temp_text = paraphrase(sentence)
                output_text = output_text + " " + temp_text[0]

            st.write("Paraphrased Text: ")
            st.write(output_text)

        else:
            st.write("Please enter a prompt.")
else:
    st.write("Please enter a valid model name or path.")

st.write("Powered by GenAI")
