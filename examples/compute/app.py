import galaxybrain.rules as rules
from galaxybrain.drivers import OpenAiCompletionDriver
from galaxybrain.prompts import Prompt
from galaxybrain.summarizers import CompletionDriverSummarizer
from galaxybrain.workflows import CompletionStep, Workflow, Memory, ComputeStep

chat_rules = [
    rules.meta.your_name_is("GalaxyGPT")
]

driver = OpenAiCompletionDriver(temperature=0.5, user="demo")
memory = Memory(summarizer=CompletionDriverSummarizer(driver=driver))
workflow = Workflow(rules=chat_rules, completion_driver=driver, memory=memory)

workflow.add_step(
    ComputeStep(
        input=Prompt(f"identity 3x3 matrix multiplied by a 3x3 matrix with random values")
    )
)
workflow.add_step(
    CompletionStep(
        input=Prompt(f"generate and output latex code based on the previous result: matrix1 x matrix2 = result")
    )
)

workflow.start()

print(workflow.last_step().output.value)
