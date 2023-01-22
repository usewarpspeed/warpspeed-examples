from typing import Tuple
from galaxybrain.summarizers import DriverSummarizer
from galaxybrain.workflows import CompletionStep, Workflow, Memory
from galaxybrain.drivers import OpenAiDriver
from galaxybrain.prompts import Prompt
import galaxybrain.rules as rules
import gradio as gr


chat_rules = [
    rules.conversation.be_truthful(),
    rules.conversation.your_name_is("GalaxyGPT")
]

driver = OpenAiDriver(temperature=0.5, user="demo")
memory = Memory(summarizer=DriverSummarizer(driver=driver))
workflow = Workflow(rules=chat_rules, driver=driver, memory=memory)

with gr.Blocks() as demo:
    def conversation_history() -> str:
        return workflow.to_string()

    def conversation_summary() -> str:
        return workflow.memory.summary

    def question_answer(question: str) -> Tuple[str, str]:
        workflow.add_step(
            CompletionStep(input=Prompt(question))
        )

        workflow.resume()

        return conversation_history(), conversation_summary()

    gr.Markdown("# GalaxyGPT")

    with gr.Row():
        with gr.Column():
            user_input = gr.components.Textbox(label="Your Input")
            translate_btn = gr.Button(value="Submit")
        with gr.Column():
            output = gr.Textbox(label="Conversation History", value=conversation_history)
            summary = gr.Textbox(label="Conversation Summary", value=conversation_summary)

    translate_btn.click(question_answer, inputs=user_input, outputs=[output, summary])

demo.launch()
