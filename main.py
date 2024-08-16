import whisper
from whisper.utils import get_writer

srt_writer = get_writer(output_format='srt', output_dir='transcripts/')
json_writer = get_writer(output_format='json', output_dir='transcripts/')
text_writer = get_writer(output_format='txt', output_dir='transcripts/')
vtt_writer = get_writer(output_format='vtt', output_dir='transcripts/')

# Set SRT Line and words width
word_options = {
    "highlight_words": False,
    "max_line_count": 1,
    "max_line_width": 42        # Standard subtitles
}

def speech_to_text(video, model, srt_writer, json_writer, text_writer, vtt_writer, word_options):
    print('speech_to_text')
    model = whisper.load_model(model, device='cpu')
    result = model.transcribe(audio=video, fp16=False, word_timestamps=False)    # word_timestamps=True for applying word_options
    srt_writer(result, video, word_options)
    json_writer(result, video, word_options)
    text_writer(result, video, word_options)
    vtt_writer(result, video, word_options)

    return result

speech_to_text('reduced_audio_files/lecture.ogg', 'medium', srt_writer, json_writer, text_writer, vtt_writer, word_options)