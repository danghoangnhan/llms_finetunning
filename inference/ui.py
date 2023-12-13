import gradio as gr
from model import ModelServe
import click

@click.command()

@click.option("--base_model","base_model",type=str,default="meta-llama/Llama-2-7b-chat-hf",)
@click.option("--model_type", "model_type", type=str, default="llama")
@click.option("--output_dir","output_dir",type=str,default="finetuned/meta-llama/Llama-2-7b-chat-hf",)

def main(
    base_model: str,
    model_type: str,
    output_dir: str,
):  
    print(
        f"Inference parameters: \n"
        f"base_model: {base_model}\n"
        f"model_type: {model_type}\n"
        f"output_dir: {output_dir}\n"
    )

    model = ModelServe(load_8bit=True,model_type=model_type,base_model=base_model,finetuned_weights=output_dir)
    demo = gr.Interface(
        fn=model.generate,
        inputs=[
            gr.components.Textbox(
                lines=2, label="Instruction", placeholder="Tell me about alpacas."
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
        ],
        outputs=[
            gr.Textbox(
                lines=10,
                label="Output",
            )
        ],
        title="🦙🌲 llama-demo",
        description="llama-demo interface.",
    )
    demo.queue()
    demo.launch(max_threads=3)
if __name__ == "__main__":
    main()



