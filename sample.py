from politeness import is_speaker_done

sample_input = ("under what name is the order", "its for")

previous, current = sample_input
result = is_speaker_done(previous, current)
print(f"\nPrevious: {previous}")
print(f"Current: {current}")
print(f"Speaker is done: {result}")