from typing import Tuple
from galaxybrain.summarizers import CompletionDriverSummarizer
from galaxybrain.workflows import CompletionStep, Workflow
from galaxybrain.workflows.memory import SummaryMemory
from galaxybrain.drivers import OpenAiCompletionDriver
from galaxybrain.prompts import Prompt
import galaxybrain.rules as rules
import gradio as gr


# Define a few basic rules for the workflow.
chat_rules = [
    rules.meta.speculate(),
    rules.meta.your_name_is("GalaxyGPT")
]

# We'll use SummaryMemory in order to support very long conversations.
driver = OpenAiCompletionDriver(temperature=0.5, user="demo")
memory = SummaryMemory(summarizer=CompletionDriverSummarizer(driver=driver))
workflow = Workflow(rules=chat_rules, completion_driver=driver, memory=memory)

# Gradio magic begins!
with gr.Blocks() as demo:
    def conversation_history() -> str:
        return workflow.memory.to_conversation_string()

    def conversation_summary() -> str:
        return workflow.memory.summary

    # This is where we add new steps to the workflow and "resume" it on
    # every received question.
    def ask_question(question: str) -> Tuple[str, str]:
        workflow.add_step(
            CompletionStep(input=Prompt(question))
        )

        workflow.resume()

        # Here we return full conversation history and conversation summary.
        # We output both to demonstrate how the summary changes with every asked
        # question.
        return conversation_history(), conversation_summary()

    # Finally, let's setup a UI with Gradio primitives.
    gr.Markdown("# GalaxyGPT")

    with gr.Row():
        with gr.Column():
            user_input = gr.components.Textbox(label="Your Input")
            translate_btn = gr.Button(value="Submit")
        with gr.Column():
            output = gr.Textbox(label="Conversation History", value=conversation_history)
            summary = gr.Textbox(label="Conversation Summary", value=conversation_summary)

    translate_btn.click(ask_question, inputs=user_input, outputs=[output, summary])

demo.launch()
