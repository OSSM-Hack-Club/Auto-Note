import os.path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from sys import argv

file_path = argv[1]

if len(argv) != 2:
    print("Improper implementation!")
    print("Usage: python3 main.py <file_path>")
    exit()

if not os.path.exists(file_path):
    print("File not found!")
    exit()

striped_name = os.path.splitext(os.path.basename(file_path))[0]

transcript_file = None
if not os.path.exists(f"transcripts/{striped_name}_transcript.txt"):
    transcript_file = open(f"transcripts/{striped_name}_transcript.txt", "x+")
else:
    transcript_file = open(f"transcripts/{striped_name}_transcript.txt", "w")


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    chunk_length_s=30,
    batch_size=16,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

result = pipe(file_path, return_timestamps=True)

for chunk in result["chunks"]:
    transcript_file.write(str(chunk["timestamp"][0]))
    transcript_file.write(": ")
    transcript_file.write(chunk["text"])
    transcript_file.write("\n")
    transcript_file.write("\n")

transcript_file.close()