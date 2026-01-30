from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("xxx.m4a", beam_size=5)

print(
    "Detected language '%s' with probability %f, duration %.2f sec, after VAD %.2f sec"
    % (info.language, info.language_probability, info.duration, info.duration_after_vad)
)

with open("transcription.txt", "w", encoding="utf-8") as f1:
    with open("transcription_verbose.txt", "w", encoding="utf-8") as f2:
        for segment in segments:
            f1.write(segment.text + " ")
            f2.write(
                "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
            )
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
