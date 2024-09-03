from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os
from langchain.schema import Document as DocumentSchema
from langchain_core.documents.base import Document as DocumentCore
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma as ChromaVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader

import bs4
import requests
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain_text_splitters import RecursiveCharacterTextSplitter # Could not pip install, maybe diff python version
# Load, chunk and index the contents of the blog.
from langchain_community.document_loaders import PyPDFLoader
import langsmith
import pandas as pd
import xml.etree.ElementTree as ET
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAI
from langchain import PromptTemplate
from langchain.agents import initialize_agent 
import mock_db

# OpenAI Key
os.environ["OPENAI_API_KEY"] = ""
os.environ["LANGSMITH_API_KEY"] = ""

##### CREATING MOCK DATABASE #####
db_queue = mock_db.queue.Queue()
db_path = 'obsplane_db.sqlite'

# Start the database worker thread
db_thread = mock_db.threading.Thread(target=mock_db.db_worker, args=(db_queue, db_path))
db_thread.start()

# Create tables
mock_db.create_tables(db_queue)

@tool
def checkSQLMock(sql_code: str) -> str:
    """Returns a string that states if the SQL query is valid"""
    try:
        print("Running SQL query...")
        # Attempt to execute the SQL command
        mock_db.run_query(db_queue, sql_code)
        return "Successful! " + sql_code
    except Exception as e:
        return "SQL query" + sql_code +f"failed with error: {str(e)}"

#### RAG Tools ####
def load_docs(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Ensure it's a file
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                # Create a document object
                document = DocumentSchema(page_content=text, metadata={"filename": filename})
                documents.append(document)
    return documents

def split_docs(documents, separators=[","]):
    split_documents = []
    for document in documents:
        parts = document.page_content.split(separators[0])
        for part in parts:
            split_documents.append(DocumentCore(page_content=part.strip(), metadata=document.metadata))
    return split_documents

valid_columns_directory = 'tools/NRCResources/validColumns'
valid_columns_documents = load_docs(valid_columns_directory)
valid_columns_docs = split_docs(valid_columns_documents)
valid_columns_vector_db = ChromaVectorStore.from_documents(valid_columns_docs, OpenAIEmbeddings(), collection_name="valid_columns")

@tool
def checkColumns(columns: str) -> str:
    """Accepts a string of columns separated by comas and checks which table the columns belong to. It returns a string representation of a dictionary where the key is the column and the value is either {column does not exist, In Observation table, In Plane table} """
    print("checked columns")
    columns = columns.split(",")
    columns = [column.strip() for column in columns]
    obs_columns_set = {'telescope_name', 'target_type', 'type', 'metaReadGroups', 'target_moving', 'collection', 'environment_ambientTemp', 'target_name', 'target_keywords', 'metaChecksum', 'environment_wavelengthTau', 'telescope_geoLocationZ', 'instrument_keywords', 'obsID', 'proposal_title', 'environment_tau', 'targetPosition_coordinates_cval2', 'typeCode', 'metaProducer', 'observationID', 'proposal_project', 'telescope_geoLocationX', 'proposal_pi', 'targetPosition_coordsys', 'accMetaChecksum', 'instrument_name', 'lastModified', 'requirements_flag', 'target_targetID', 'sequenceNumber', 'targetPosition_coordinates_cval1', 'environment_humidity', 'intent', 'algorithm_name', 'maxLastModified', 'telescope_keywords', 'environment_elevation', 'target_standard', 'telescope_geoLocationY', 'members', 'targetPosition_equinox', 'environment_seeing', 'metaRelease', 'proposal_id', 'target_redshift', 'environment_photometric', 'observationURI', 'proposal_keywords'}
    plane_columns_set= {'metaReadGroups', 'time_bounds_samples', 'energy_freqWidth', 'position_bounds_samples', 'dataProductType', 'position_dimension_naxis2', 'provenance_runID', 'lastModified', 'energy_resolvingPower', 'provenance_name', 'position_sampleSize', 'energy_bounds_width', 'position_dimension_naxis1', 'energy_sampleSize', 'metrics_magLimit', 'time_resolutionBounds', 'provenance_producer', 'energy_resolvingPowerBounds', 'energy_bounds_upper', 'provenance_inputs', 'provenance_keywords', 'time_bounds_upper', 'polarization_states', 'energy_restwav', 'position_timeDependent', 'energy_dimension', 'metaProducer', 'quality_flag', 'custom_bounds_upper', 'energy_transition_transition', 'energy_bounds', 'time_bounds_width', 'provenance_version', 'planeID', 'custom_bounds_samples', 'planeURI', 'creatorID', 'metaRelease', 'observable_ucd', 'custom_bounds', 'provenance_reference', 'energy_bounds_lower', 'publisherID', 'metrics_background', 'polarization_dimension', 'custom_dimension', 'metaChecksum', 'time_resolution', 'provenance_project', 'obsID', 'metrics_backgroundStddev', 'dataRelease', 'accMetaChecksum', 'time_sampleSize', 'position_resolutionBounds', 'time_bounds', 'time_dimension', 'calibrationLevel', 'energy_transition_species', 'custom_ctype', 'metrics_sourceNumberDensity', 'energy_bounds_samples', 'energy_emBand', 'position_resolution', 'metrics_fluxDensityLimit', 'position_bounds_size', 'dataReadGroups', 'custom_bounds_width', 'custom_bounds_lower', 'productID', 'energy_freqSampleSize', 'time_exposure', 'energy_bandpassName', 'position_bounds', 'maxLastModified', 'provenance_lastExecuted', 'time_bounds_lower', 'energy_energyBands'}

    result = dict()
    for column in columns:
        if column not in obs_columns_set and column not in plane_columns_set:
            query = "What is the most similar to " + column + "?"
            matching_docs = valid_columns_vector_db.similarity_search(query)
            page_contents = [doc.page_content for doc in matching_docs[:10]]
            result[column] = "column does not exist, try one of these" + str(page_contents)
            continue
        if column in obs_columns_set:
            result[column] = "In Observation table"
        else:
            result[column] = "In Plane table"
    print(str(result))
    return str(result)

map_columns_directory = 'tools/NRCResources/columnmappings'

map_columns_documents = load_docs(map_columns_directory)
map_columns_docs = split_docs(map_columns_documents)

map_columns_vector_db = Chroma.from_documents(documents=map_columns_docs, embedding=OpenAIEmbeddings(), collection_name="map_columns")

# Retrieve and generate using the relevant snippets of the blog.
retriever_col = map_columns_vector_db.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

map_columns_llm= OpenAI()


map_columns_rag_chain = (
    {"context": retriever_col | format_docs, "question": RunnablePassthrough()}
    | prompt
    | map_columns_llm
    | StrOutputParser()
)

@tool
def alternateColumn(column: str) -> str:
    """Checks to see if a column needs to be changed to an alternate column name"""
    obs_columns_set = {'telescope_name', 'target_type', 'type', 'metaReadGroups', 'target_moving', 'collection', 'environment_ambientTemp', 'target_name', 'target_keywords', 'metaChecksum', 'environment_wavelengthTau', 'telescope_geoLocationZ', 'instrument_keywords', 'obsID', 'proposal_title', 'environment_tau', 'targetPosition_coordinates_cval2', 'typeCode', 'metaProducer', 'observationID', 'proposal_project', 'telescope_geoLocationX', 'proposal_pi', 'targetPosition_coordsys', 'accMetaChecksum', 'instrument_name', 'lastModified', 'requirements_flag', 'target_targetID', 'sequenceNumber', 'targetPosition_coordinates_cval1', 'environment_humidity', 'intent', 'algorithm_name', 'maxLastModified', 'telescope_keywords', 'environment_elevation', 'target_standard', 'telescope_geoLocationY', 'members', 'targetPosition_equinox', 'environment_seeing', 'metaRelease', 'proposal_id', 'target_redshift', 'environment_photometric', 'observationURI', 'proposal_keywords'}
    plane_columns_set= {'metaReadGroups', 'time_bounds_samples', 'energy_freqWidth', 'position_bounds_samples', 'dataProductType', 'position_dimension_naxis2', 'provenance_runID', 'lastModified', 'energy_resolvingPower', 'provenance_name', 'position_sampleSize', 'energy_bounds_width', 'position_dimension_naxis1', 'energy_sampleSize', 'metrics_magLimit', 'time_resolutionBounds', 'provenance_producer', 'energy_resolvingPowerBounds', 'energy_bounds_upper', 'provenance_inputs', 'provenance_keywords', 'time_bounds_upper', 'polarization_states', 'energy_restwav', 'position_timeDependent', 'energy_dimension', 'metaProducer', 'quality_flag', 'custom_bounds_upper', 'energy_transition_transition', 'energy_bounds', 'time_bounds_width', 'provenance_version', 'planeID', 'custom_bounds_samples', 'planeURI', 'creatorID', 'metaRelease', 'observable_ucd', 'custom_bounds', 'provenance_reference', 'energy_bounds_lower', 'publisherID', 'metrics_background', 'polarization_dimension', 'custom_dimension', 'metaChecksum', 'time_resolution', 'provenance_project', 'obsID', 'metrics_backgroundStddev', 'dataRelease', 'accMetaChecksum', 'time_sampleSize', 'position_resolutionBounds', 'time_bounds', 'time_dimension', 'calibrationLevel', 'energy_transition_species', 'custom_ctype', 'metrics_sourceNumberDensity', 'energy_bounds_samples', 'energy_emBand', 'position_resolution', 'metrics_fluxDensityLimit', 'position_bounds_size', 'dataReadGroups', 'custom_bounds_width', 'custom_bounds_lower', 'productID', 'energy_freqSampleSize', 'time_exposure', 'energy_bandpassName', 'position_bounds', 'maxLastModified', 'provenance_lastExecuted', 'time_bounds_lower', 'energy_energyBands'}

    if column in obs_columns_set or column in plane_columns_set:
      return "No alternate column needed, " + column + " exists in the " + ("observation" if column in obs_columns_set else "plane") + " table."
    result = map_columns_rag_chain.invoke("Check to see if this word is mapped to: " + column +". This mapping only works left to right.")
    print("used alternate column")
    print(result)
    return result

### Synonyms ###
value_synonyms_directory = 'tools/NRCResources/valueSynonymMappings'

value_synonyms_documents = load_docs(value_synonyms_directory)
value_synonyms_docs = split_docs(value_synonyms_documents)

value_synonyms_vector_db = Chroma.from_documents(documents=value_synonyms_docs, embedding=OpenAIEmbeddings(), collection_name="value_synonyms")

# Retrieve and generate using the relevant snippets of the blog.
retriever_val = value_synonyms_vector_db.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

value_synonyms_llm= OpenAI()


value_synonyms_rag_chain = (
    {"context": retriever_val | format_docs, "question": RunnablePassthrough()}
    | prompt
    | value_synonyms_llm
    | StrOutputParser()
)

@tool
def alternateValue(value: str) -> str:
    """Checks to see if a value needs to be changed to an alternate value name"""
    result = value_synonyms_rag_chain.invoke("Check to see if this word maps to another: " + value +". This mapping only works left to right.")
    print("used alternate value tool")
    print(result)
    print("value passed in")
    print(value)
    return result

#### Schemas ####
pd.set_option('display.max_rows', None)
def get_table_schema(table_name: str) -> pd.DataFrame:
    """Parses the VOTABLE XML response to extract table schema information."""
    headers = {'Accept': 'application/x-votable+xml'}
    response = requests.get(f"https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync?LANG=ADQL&QUERY=SELECT%20*%20FROM%20{table_name}%20LIMIT%200", headers=headers)
    xml_response = response.content.decode('utf-8')  # Decode the response content

    root = ET.fromstring(xml_response)

    columns = []
    types = []
    descriptions = []

    for field in root.findall(".//{http://www.ivoa.net/xml/VOTable/v1.3}FIELD"):
        columns.append(field.attrib['name'])
        types.append(field.attrib['datatype'])
        descriptions.append(field.find("{http://www.ivoa.net/xml/VOTable/v1.3}DESCRIPTION").text)

    schema_df = pd.DataFrame({
        'Column': columns,
        'Type': types,
        'Description': descriptions
    })

    return schema_df

def format_schemas(*schema_dfs) -> str:
    schema_str = "Table Schema:\n"
    for schema_df in schema_dfs:
        table_name = schema_df['Table'].iloc[0]
        schema_str += f"Table: {table_name}\n"
        for index, row in schema_df.iterrows():
            schema_str += f"  Column: {row['Column']} - Type: {row['Type']} - Description: {row['Description']}\n"
    schema_str += "Join Condition: caom2.Plane.obsID = caom2.Observation.obsID\n"
    return schema_str

# get the two table schemas
obs_schema = get_table_schema("caom2.Observation")
plane_schema = get_table_schema("caom2.Plane")
# name the tables
obs_schema['Table'] = 'observation' # Changed table names
plane_schema['Table'] = 'plane'
# format schemas into a string to pass to sql generator
combined_schema_str = format_schemas(obs_schema, plane_schema)

#### React Prompt ####
def get_react_prompt_template():
    # Get the react prompt template
    return PromptTemplate.from_template("""Answer the following questions as accurately as possible. Your primary task is to enhance the user's input by ensuring it is complete, accurate, and consistent with the database schema. You will need to augment and improve the input to add the correct table names to the prompt, change or swap out any incorrect column names and change any values that have an alternate value. To achieve this, follow these instructions carefully:
It is important to know that there are only two tables in the database: observation and plane. They are joined on the obsID column. When you enhance the user's input mention the table names along with the corresponding column names and values that the user is requesting. If the user is requesting data from both tables enhance the user's input to include the join condition.

**Instructions:**
You must use the tools in the following order. Do not pass a column into checkColumns without before passing it into alternateColumn to see if it needs to be changed.
1. **alternateColumn**: Always start by verifying if any column names in the user's input need to be updated or corrected based on the database schema. Use this tool to find the correct column names if needed.
2. **checkColumns**: Next, ensure that all mentioned columns exist in the specified table(s). Confirm which columns belong to which table(s) and that the columns in the user's input are valid.
3. **alternateValue**: Finally, check if any values (like specific names or IDs) in the user's input need to be corrected. Use this tool to verify and update values as necessary.

**Important:** You must use these tools in the order provided for every query.
NOTE: Do not write any sql for your answer. Just enhance the users input by ensuring the values and columns are mapped to the correct values using the tools you have.

    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """)

#### Creating Agent ####
# Define a prompt template to generate SQL from natural language questions
template = """
You are an expert SQL query generator. Given a natural language question and a table schema, generate an appropriate SQL query.
Only return the SQL query and nothing else.
{schema}
Question: {question}
SQL Query:
"""

prompt = PromptTemplate(input_variables=["schema", "question"], template=template)

# Initialize the OpenAI LLM (replace with your actual API key)
llm = ChatOpenAI(model="gpt-4o-mini")

# Define a function that uses the LLM to generate SQL
def generate_sql(question: str, schema: str) -> str:
    response = llm.invoke(prompt.format(schema=schema, question=question))
    sql_query = str(response.content).strip()
    sql_query = sql_query.replace("```sql\n", "").replace("```", "").strip()

    print("generated sql")
    print(sql_query)

    print("____________")
    return sql_query

class SQLAgent:
    def __init__(self, sql_tool, sql_generator):
        self.sql_tool = sql_tool
        self.sql_generator = sql_generator
        self.max_retries = 3
        self.preprocessAgent = self.createPreprocessAgent()

    def createPreprocessAgent(self):
        # Choose the LLM to use
        llm2 = ChatOpenAI(model="gpt-4o-mini")

        # Get the react prompt template
        prompt_template = get_react_prompt_template()


        # set the tools
        tools = [alternateColumn,checkColumns, alternateValue]

        # Construct the ReAct agent
        agent = create_react_agent(llm2, tools, prompt_template)

        # Create an agent executor by passing in the agent and tools
        return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    def preprocess(self, input: str) -> str:
        improved_input = self.preprocessAgent.invoke({"input": input})
        return improved_input

    def regenerate_sql(self, question: str, schema: str, error_messages: list, previous_queries: list) -> str:
        # Create a new prompt that includes the error messages and previous queries to guide the LLM in fixing the query
        regenerate_prompt = """
        You are an expert SQL query generator. Given a natural language question, a table schema, previous error messages, and previous SQL queries, generate a corrected SQL query.
        Only return the SQL query and nothing else. Do not include a semicolon at the end of the query.

        Schema:
        {schema}

        Question:
        {question}

        Error messages:
        {error_messages}

        Previous SQL queries:
        {previous_queries}

        Corrected SQL Query:
        """
        error_messages_str = '\n'.join(error_messages)
        previous_queries_str = '\n'.join(previous_queries)
        prompt = PromptTemplate(input_variables=["schema", "question", "error_messages", "previous_queries"], template=regenerate_prompt)
        response = llm.invoke(prompt.format(schema=schema, question=question, error_messages=error_messages_str, previous_queries=previous_queries_str))
        response_content = str(response.content)

        sql_query = response_content
        return sql_query

    def run(self, question: str) -> pd.DataFrame:
        schema = combined_schema_str
        error_messages = []
        previous_queries = []
        print("orignal question")
        print(question)
        question = self.preprocess(question)["output"]
        print("augmented question")
        print(question)
        for attempt in range(self.max_retries):
            try:
                # Generate the SQL query from the question and schema
                sql_query = self.sql_generator(question, schema)
                print(f"Attempt {attempt+1}: Generated query: {sql_query}")
                previous_queries.append(sql_query)
                # Run the generated SQL query using the tool
                return self.sql_tool.invoke(sql_query)
            except Exception as e:
                error_message = str(e)
                print(f"Attempt {attempt+1}: Error encountered: {error_message}")
                error_messages.append(error_message)
                # Regenerate the SQL query with the error message and previous queries
                sql_query = self.regenerate_sql(question, schema, error_messages, previous_queries)
        raise Exception("Failed to generate a correct SQL query after " + str(self.max_retries) + " attempts.")
# Initialize the agent
agent = SQLAgent(checkSQLMock, generate_sql)
