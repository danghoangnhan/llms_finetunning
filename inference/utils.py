def generate_prompt(instruction, input=None):
    if input:
        return ("Dưới đây là một chỉ thị mô tả nhiệm vụ, cùng với thông tin đầu vào liên quan đến nhiệm vụ. Hãy soạn một câu trả lời phù hợp để hoàn thành chỉ thị này.\n\n"
        f'### Chỉ thị:\n{instruction}\n\n### Đầu vào:\n{input}\n\n### Trả lời:')
    else:
        return ("Dưới đây là một chỉ thị mô tả nhiệm vụ. Hãy soạn một câu trả lời phù hợp để hoàn thành chỉ thị này.\n\n"
        f'### Chỉ thị:\n{instruction}\n\n### Trả lời:\n')