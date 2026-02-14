from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    template=""""
    Summarize the research paper '{paper_input}' in a {style_input} style with a maximum length of {length_input} words.""",
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True,
)

template.save('template.json')