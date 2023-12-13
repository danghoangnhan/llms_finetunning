
import transformers

def general_config():
    generation_config = transformers.GenerationConfig(
        do_sample = True,
        temperature = 0.3,
        top_p = 0.1,
        top_k = 80,
        repetition_penalty = 1.5,
        max_new_tokens = 100    
    )
    return generation_config