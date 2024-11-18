class AnsweringConfig:
    model_name = "accounts/fireworks/models/llama-v3p1-70b-instruct"
    system_prompt = "answering_prompt.txt"
    context_size = 8192
    top_p = 0.95
    temperature = 0.05
    max_tokens = 1024
    n = 1
    frequency_penalty = 1.15
    seed = 42
