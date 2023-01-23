import streamlit as st
import logging
from galaxybrain.drivers import OpenAiCompletionDriver
from galaxybrain.prompts import Prompt
from galaxybrain.rules import Rule, Validator
import galaxybrain.rules.json as json_rules
from galaxybrain.workflows import Workflow, CompletionStep
from lib.parser import to_dataframe


# We want OpenAI to extract analytics data and return it as a valid JSON object,
# so let's defined some rules
rules = [
    json_rules.return_valid_json(),
    Rule("act as an analyst and extract quantitative data from your inputs")
]

# Now, let's define a default workflow driver that will be automatically used
# to make requests to GPT.
driver = OpenAiCompletionDriver(temperature=0.5, user="demo")

# Finally, setup the workflow
workflow = Workflow(rules=rules, completion_driver=driver)

st.session_state.is_data_processed = False

st.title("GPT Analyst")

# Ask for user input
with st.expander("Raw Data", expanded=True):
    raw_text = st.text_area(
        "Add analytics text",
        height=400
    )

# Once the "Process" button is clicked, we kick off the magic.
if st.button("Process Raw Data") or st.session_state.is_data_processed:
    with st.spinner("Please wait..."):
        try:
            # Add a completion step to the workflow
            step = workflow.add_step(
                CompletionStep(input=Prompt(raw_text))
            )

            # Start the workflow. This will execute out only Step defined above
            workflow.start()

            # Only proceed if the LLM result was validated against our rules
            validator = Validator(step.output, rules)

            if validator.validate():
                st.session_state.is_data_processed = True

                processed_text = step.output.value

                # Add a text field for the OpenAI output that the user can tweak.
                # When this field is updated all charts and tables are automatically
                # refreshed.
                with st.expander("Processed Data", expanded=True):
                    edited_data = st.text_area(
                        "Edit processed data",
                        processed_text,
                        height=400
                    )

                # Finally, render default Streamlit visualizations
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["DataFrame", "Bar Chart", "Line Chart", "Map"]
                )

                with tab1:
                    try:
                        st.dataframe(to_dataframe(edited_data), use_container_width=True)
                    except Exception as err:
                        st.error(err)
                with tab2:
                    try:
                        st.bar_chart(to_dataframe(edited_data))
                    except Exception as err:
                        st.error(err)
                with tab3:
                    try:
                        st.line_chart(to_dataframe(edited_data))
                    except Exception as err:
                        st.error(err)
                        pass
                with tab4:
                    try:
                        st.map(to_dataframe(edited_data))
                        pass
                    except Exception as err:
                        st.error(err)
            else:
                failed_rules = "\n".join([rule.value for rule in validator.failed_rules()])

                st.error(f"The following rules failed: {failed_rules}")

        except Exception as err:
            st.error(err)
            logging.error(err.with_traceback, exc_info=True)
