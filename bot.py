from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os
from langchain.schema import Document as DocumentSchema
from langchain_core.documents.base import Document as DocumentCore
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma as ChromaVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
import re
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
def load_doc(filepath):
   documents = []
   filename = os.path.basename(filepath)
   filepath = os.path.join(filepath)


   # Ensure it's a file
   if os.path.isfile(filepath):
       with open(filepath, 'r', encoding='utf-8') as file:
           text = file.read()
           # Create a document object
           document = DocumentSchema(page_content=text, metadata={"filename": filename})
           documents.append(document)


   return documents

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

def strip_non_alphanumeric(s: str) -> str:
    # Use regex to remove non-alphanumeric characters from the start and end
    return re.sub(r'^\W+|\W+$', '', s)

@tool
def checkColumns(columns: str) -> str:
    """Accepts a string of columns separated by comas and checks which table the columns belong to. It returns a string representation of a dictionary where the key is the column and the value is either {column does not exist, In Observation table, In Plane table} """
    print("checked columns")
    print("columns")
    print(columns)

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
    result = map_columns_rag_chain.invoke("return the sentence where " + column +" is mentioned")
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
    result = value_synonyms_rag_chain.invoke("return the sentence where " + value +" is mentioned. If it is not mentioned return 'there is no mapping'")
    print("used alternate value tool")
    print(result)
    print("value passed in")
    print(value)
    return result

calibrationLevel = 'tools/NRCResources/valid_values/calibrationLevel.txt'
calibrationLevel = load_doc(calibrationLevel)
calibrationLevel = split_docs(calibrationLevel)
calibrationLevel = Chroma.from_documents(calibrationLevel, OpenAIEmbeddings(),collection_name="calibrationLevel")


collection = 'tools/NRCResources/valid_values/collection.txt'
collection = load_doc(collection)
collection = split_docs(collection)
collection = Chroma.from_documents(collection, OpenAIEmbeddings(),collection_name="collection")


dataProductType = 'tools/NRCResources/valid_values/dataProductType.txt'
dataProductType = load_doc(dataProductType)
dataProductType = split_docs(dataProductType)
dataProductType = Chroma.from_documents(dataProductType, OpenAIEmbeddings(),collection_name="dataProductType")




energy_bandpassName = 'tools/NRCResources/valid_values/energy_bandpassName.txt'
energy_bandpassName = load_doc(energy_bandpassName)
energy_bandpassName = split_docs(energy_bandpassName)
energy_bandpassName = Chroma.from_documents(energy_bandpassName, OpenAIEmbeddings(),collection_name="energy_bandpassName")




energy_emBand = 'tools/NRCResources/valid_values/energy_emBand.txt'
energy_emBand = load_doc(energy_emBand)
energy_emBand = split_docs(energy_emBand)
energy_emBand = Chroma.from_documents(energy_emBand, OpenAIEmbeddings(),collection_name="energy_emBand")




instrument_name = 'tools/NRCResources/valid_values/instrument_name.txt'
instrument_name = load_doc(instrument_name)
instrument_name = split_docs(instrument_name)
instrument_name = Chroma.from_documents(instrument_name, OpenAIEmbeddings(),collection_name="instrument_name")




intent = 'tools/NRCResources/valid_values/intent.txt'
intent = load_doc(intent)
intent = split_docs(intent)
intent = Chroma.from_documents(intent, OpenAIEmbeddings(),collection_name="intent")




telescope_name = 'tools/NRCResources/valid_values/telescope_name.txt'
telescope_name = load_doc(telescope_name)
telescope_name = split_docs(telescope_name)
telescope_name = Chroma.from_documents(telescope_name, OpenAIEmbeddings(),collection_name="telescope_name")






types = 'tools/NRCResources/valid_values/type.txt'
types = load_doc(types)
types = split_docs(types)


types = Chroma.from_documents(types, OpenAIEmbeddings(),collection_name="types")




column_dbs = {"calibrationLevel": calibrationLevel, "collection": collection,"dataProductType": dataProductType,"energy_bandpassName":energy_bandpassName,"energy_emBand":energy_emBand,"instrument_name":instrument_name,"intent":intent, "telescope_name":telescope_name, "type":types}

@tool
def checkValuesForCommonColumns(columnAndValue: str) -> str:
   """Accepts a string of a column name and value name separated by a comma. This tool checks if the value exists in the column and returns a string indicating if it exists or and if not returns a value that does exist that is the most similar. This tool is only valid for the following columns:instrument_name,calibrationLevel,type,energy_emBand,collection,energy_bandpassName,telescope_name,dataProductType,intent} """
   column = columnAndValue.split(",")[0]
   value = columnAndValue.split(",")[1]
   print("checkValuesForCommonColumns")
   print("column")
   print(column)
   print("value")
   print(value)

   distinct_values_dict = {'dataProductType': {'cube', 'event', 'http://www.opencadc.org/caom2/DataProductType#catalog', 'image', 'measurements', 'spectrum', 'timeseries', 'visibility'}, 'energy_bandpassName': {'F814W/CLEAR_HRC/F606W/F475W', 'G230M;G430L;G750L;G750M', 'CH4-H6.5L + CH4-H6.5S', 'F435W/F550M/F775W', 'E140M;G160M', 'F140W/F125W/F098M', 'F110W/F190N/F095N/F145M/F160W', 'F550M/F660N/F658N/F435W/F814W', 'F284M', 'F336W/F439W/F255W/F555W/F160BW', 'F502N/F658N/F547M/F656N', 'F785LP/F850LP/F1042M/F622W/F791W', 'E230M;G430L', 'F606W/F555W/F702W', 'F220W/F344N/F775W/F502N/F850LP/F', 'F110W/F222M/F237M/F160W', 'F814W/F380W', 'F656N/F475W/F555W/F814W', 'F502N/F373N/F631N', 'SN1', 'U_new', 'F658N/F606W', 'F814W/F275W/F606W', 'Kcont + J', 'F125LP/F115LP', 'Wash_C', 'W-S-ZB', 'F350LP/F814W/F200LP/F502N', 'g.MP9401', 'F110W/F160W/F145M/F170M', 'F187N;MASKRND', 'K Cont (5342)', 'G200M', '5007/55-4x4', 'F110W/F222M/F160W/F187N', 'F187N;WLM8', 'FR388N', 'F128N/F140W/F098M/F105W/F167N/F1', 'F555W;CLEAR2L', 'PRISM2;CLEAR2;F190M;CLEAR4', 'F850LP/F791W', '1.644(FeII)[MK]', 'CO', 'GG455 + r', 'Lya395', 'JWL38', 'MIRRORB', 'Kcon(209)', 'F108N/F190N/F187N', 'F658N/F502N/F555W', 'VR Bernstein k1040', 'F850LP/F250W', 'CLEAR1L;FR914M', 'F215N/F216N', 'F656N/F555W/F658N/F469N/F502N/F4', 'F250M;MASKBAR', 'none', 'F380M', 'V+R', 'F185W', 'F336W/F275W/F438W/F390W/F814W/F8', 'F110W/F207M/F160W', 'F255W/F439W/F555W', 'F187N;F405N_F444W', 'F555W/F673N/F658N/F814W/F631N/F6', 'POL60;CLEAR2;F1ND;F346M', 'F439W/F547M/F469N/F502N', 'F439W/F502N/F555W/F673N/F658N/F6', 'W016', 'F180M/F237M/F204M/F212N/F171M/F1', 'F336W/F814W/F656N/F555W/F439W', 'F110W/F205W/F207M/F187W/F160W', 'GR150R;F140M', 'F850LP/F606W/F892N', 'F656N/F218W/F502N', 'F410M;GRISMC', 'F139M/F153M', 'CLEAR1S;F660N', 'F164N/F127M/F110W', 'KP1582_5289/256', 'F300W;F814W', 'HartmannA + i', 'F6ND;F342W;F1ND;CLEAR4', 'F344N/F550M/F502N/F892N/F660N', 'E230H;G140L', 'F185W/F122M/F380W/F343N/F375N/F3', 'F300W/F547M/F380W', '6563-75 c656375', 'I128B2', 'F300W/F218W/F547M/F439W', 'C2', 'OPAQUE;G235H', 'F390W/F775W', 'Kprime + Block', 'F8ND;CLEAR2;F165W;CLEAR4', 'F814W/F555W/F450W', 'Dark', 'F115W;F405N_F444W', 'gunn i', 'F110W/F160W/F125W/F140W', 'F656N/F439W/F555W/F675W', 'CH4-H1L + J', 'F212N/F215N/F216N', 'CaT + HartmannD', 'CLEAR1;F220W;F210M;CLEAR4', 'W', 'O3 k1590', 'F171M/F204M/F187W', 'F225W/F555W', 'F344N/F550M/F658N/F502N/F892N', 'E140M;E230M;G430M;G750M', 'Si5-11.7um + N', 'i2', 'E230H;G130M', 'F550M/F660N/F502N/F344N', 'F467M', 'F555W;F122M', 'F098M/F140W/F160W/F127M/F110W/F1', 'F560W/MRS', 'gunn r', 'F108N/F187N', 'G800L;FR459M', 'F606W/F775W', 'F814W/F450W/F675W', 'F439W/F555W', 'F300W/F439W/F555W/F814W', 'I Harris W5', 'FR533P15;F502N', 'PRISM1;F175W;CLEAR3;CLEAR4', 'BLOCK3', 'F438W/F625W', 'V Harris c6026', 'G140M;G430L;G430M;G750M', 'CH2-SHORT', 'INVALID&J', 'B Harris', 'CLEAR1;CLEAR2;F165W;CLEAR4', 'F182M;MASKRND', 'F814W/F255W/F606W', 'F150W;F480M', 'N-B-L711', 'F487N/F469N', 'F588N;FR533N33', 'F435W/F814W/F330W/F555W', 'F336W/F814W', 'F300W/F673N/F487N/F658N/F547M/F5', 'F327M', 'Off OIII+30nm k1015', 'h-alpha k1009', 'FR533N18', 'FR680P15/F814W/F547M', 'VR k1040', 'F113N/F097N', 'O2 OII c6012', 'F187N;F405N', 'F8ND;F547M', 'H (5209)', 'FR533N18/FQUVN-D/FR533N/FR533P15', 'CLEAR1S;F220W', 'F336W/F547M/F791W/F502N', 'F212N/F110W/F160W/F205W', 'F207M/F160W', 'F255W/F656N', 'F785LP;F122M', 'POL120;CLEAR2;F1ND;F410M', 'F128N/F164N', 'F814W/F275W/F475W', 'F814W/F555W', 'F487N/F343N/F437N/F469N/F390N', 'F1042M;F437N', 'F165M/F110M/F110W/F108N/F160W', 'PR130L', 'F814W/F555W/F250W', 'D51 DDO 51 c6008', 'F160W/F110W/F139M/F125W/F098M', 'G430L;G750L;G750M', 'W015_6570/73', 'PR110L', 'F492M;F122M', 'KP 1585 Sloan R', 'F6ND;CLEAR2;CLEAR3;F470M', 'F139M', 'F165M/F110W/F205W/F187W/F160W', 'CLEAR1;F275W;CLEAR3;CLEAR4', 'wh', 'SDSS i', 'F222M/F187N/F212N/F190N/F216N/F1', 'F330W/F555W/F250W', 'F750W;F750W', 'FR533N/FR868N/FR418N', 'F547M;F122M', 'FR680P15/FR680N/FR533N/FR680N18', 'L', 'F435W/F658N/F850LP/F775W/F555W', 'F4ND;F275W;F1ND;F278M', 'F165M/F171M/F207M/F180M', 'F127M/F139M/F153M', 'F502N/F658N/F547M/F656N/F631N', 'F435W/F658N', 'SII_Ha16 W37', 'r SDSS k1018', '1350um', 'F336W/F814W/F555W/F675W/F450W', 'F2550W/MRS', 'F814W/F225W/F475W', 'F182M;WLP8', 'NOTCMDED;NOTCMDED', 'F110W/F160W/F187W', 'F090W;F430M', 'H2(1-0)', 'FR462N', '337 BATC k1051', '4578', 'F336W/F343N/F438W', 'F300W/F555W', 'F375N/F953N/F439W/F673N/F656N/F4', 'F336W/F656N/F555W/F673N/F814W/F5', 'F375N;FR418N18', 'F250M;NONE', 'V', 'SDSS r prime', 'U Harris_solid', 'F110M/F165M/F110W/F108N/F160W', 'CLEAR1;CLEAR2;F210M;CLEAR4', 'POL120;F320W;F1ND;CLEAR4', 'F631N;FR680P15', 'WashM_KP1581', 'F140W/F105W/F098M/F160W/F127M/F1', 'E230H;G140M', 'F430M;GRISMR', 'POL60;F122M', 'F140W/F105W/F139M/F160W/F110W/F1', 'F110M/F165M/F145M', '850um', 'I-A-L550', 'F090W;F444W', 'F439W/F170W/F555W/F814W', 'O3 5023 40 cO3 5023 40', 'F656N/F502N/F631N', 'G140M', 'F237M/F216N/F187N/F212N/F190N/F2', 'F336W/F225W/F547M/F673N/F438W', '454 BATC k1054', 'F606W/F475W', 'F171M', 'F622W/F450W', 'SDSS_g', 'F502N/F658N/F656N/F673N/F631N', 'F220W/F555W', 'PRISM2;F342W;CLEAR3;CLEAR4', 'F850LP;CLEAR2S', 'U Harris k1001', 'F373N/F469N/F673N/F953N', 'F336W/F300W/F170W/F255W/F555W', 'sloan_i', 'F555W/F675W', 'POL0;F547M', 'F164N/F110W/F098M/F105W/F140W/F1', 'PK50', 'F100LP/G140M', 'F775W/F850LP/F555W/F435W/F606W/F', 'rd DECam k1036', 'Harris_R', 'F100LP;G140H', 'F300W;POLQ', 'KPHOT', 'SDSS z prime', 'F547M/F450W/F675W', 'Ha', 'W013', 'F336W/F275W/F438W/F775W/F300X/F3', '0', 'F127M', 'F390W/F814W/F475W/F625W', 'F487N/F502N/F656N/F673N', 'PRISM1;F140W;CLEAR3;CLEAR4', 'F336W/F439W/F170W/F255W/F555W', 'F4ND;F220W;F1ND;F253M', 'F439W/F555W/F658N/F502N/F656N/F6', 'F336W/F814W/F439W/F555W', 'F658N;CLEAR2S', 'F390N/F410M', 'F105W/F127M/F098M', 'F212N/F215N/F190N/F216N/F187N', 'BrG.WC8305', 'SDSS_Z', 'F775W/F475W/F850LP/F555W/F435W/F', 'F336W/F814W/F555W/F657N', 'F814W/F390M/F475W/F555W/F625W', 'Block + J', 'F336W/F438W/F502N/F673N/F775W', 'r_gunn', 'F140W/F105W/F098M/F160W/F110W/F1', '6563/78', 'FR680P15/F631N', 'F200W;F356W', 'F1042M/F160BW', 'I-A-L624', 'Stromgen_b', 'F375N/F656N/F631N/F487N/F658N/F5', 'F400LP', 'F665N/F547M', 'FR868N33', 'I105B53', 'F343N/F438W', 'F250M;GRISMR', 'F814W/F439W/F555W/F675W/F380W', 'F336W/F814W/F547M/F675W/F502N', 'Br(alpha)', 'F547M/F791W/F656N/F502N', 'F110W/F160W/F175W', 'F336W/F255W/F410M/F170W/F343N', '-1.800', 'E140M;G130M', 'FR914M', 'F115W;F250M', 'CLEAR1;F140W;F1ND;CLEAR4', 'F108N/F160W/F187N', 'F2ND;F275W;F1ND;F278M', 'F165M/F110M/F090M', 'i', 'F330W/F555W', 'W.WC8105', 'F2100W/MRS', 'F547M/F502N/F658N/F656N/F673N', 'H2v=2-1s1', 'F487N/F656N/F547M/F502N', 'F336W/F170W/F439W/F555W', 'F2100W', 'K-long', 'F290LP;G395M', 'E140H;E230H;E230M', 'W5007 55 w5007', 'BrGamma', 'F164N/F110W/F160W/F108N', 'F6ND;CLEAR2;CLEAR3;F550M', 'F702W/F555W/F450W', 'K_(order_3)', 'F467M/F275W/F410M', 'F112B21', 'F475W/F555W/F775W', 'F380W/F255W/F390N', 'F656N', 'F8ND;CLEAR2;CLEAR3;F346M', 'SDSS AURA 9704', 'F128N/F160W/F125W', 'F110W/F187W/F187N', 'F467M/F665N', 'FQ508N/F547M', 'donats', 'F127M/F153M/F139M/F098M', 'FR680N33;F631N', 'U_Harris', 'PRISM1;CLEAR2;F140M;CLEAR4', 'K-blue', 'F250M;MASKRND', 'F380W/F791W/F439W/F555W/F702W', 'F2ND;F220W;F1ND;CLEAR4', 'F8ND;F220W;F1ND;CLEAR4', 'Strom y cStrom y', 'Sloan_r', 'F350LP/F814W/F200LP/F502N/F555W', 'F658N/F625W', 'F110W/F098M/F139M/F125W/F160W', 'G130M;G160M;G230L;G430L', 'F237M', 'F160BN15', 'O3 4990', 'F606W;POL60V/F606W;POL0V/F606W;P', 'F673N;FR680N18', 'F444W;GRISMC', 'FR656N/F555W/F814W', 'F140X;MIRROR', 'G130M;G160M;G230L;G750L', 'F336W/F275W/F502N/F555W/F814W/F2', 'OIII', 'F2ND;CLEAR2;CLEAR3;F346M', 'rd DES k1036', 'G185M', 'F750W', 'NB0515', 'F502N;F660N', 'F775W;POL120V', 'FeII', 'F225W', 'BG39', 'Kprime', 'F300W/F656N/F606W/F450W/F814W', '12', 'Cu SO4', 'OVI + g', 'F502N;CLEAR2S', 'Ha5 7240', 'F467M/FQ906N/F547M/F814W/F606W/F', 'LowOH1', 'F196N/F150W/F175W', 'POL0;F220W;CLEAR3;CLEAR4', 'G230LB;G230MB', 'CLEAR1;CLEAR2;CLEAR3;F130LP', 'Red', 'F195W;PRISM2', 'POL0UV;F330W', 'F212N/F215N/F110W/F160W', 'F110W/F160W/F187W/F205W', 'PRISM2;F140W;CLEAR3;CLEAR4', 'F200W;MASKBAR', 'I-A-L598', 'F550M/F814W/F435W', 'F4ND;CLEAR2;F1ND;F550M', 'HeII', 'W-S-G+', 'CLEAR1;CLEAR2', 'F814W/F606W/F656N', 'F115W;F277W', 'H2(2-1)', 'F128N/F130N/F110W', 'F673N/F588N', 'gd DECcam k1035', 'G130M;G140L;G430M', 'F164N/F190N/F095N', 'F170M/F145M', 'U c6001', 'CLEAR;F150W', 'F450W/F675W', 'ha H-alpha+16nm k1013', 'F487N/F502N/F547M', 'I103B10', 'POL0;F342W;CLEAR3;CLEAR4', 'Si5-11.7um', 'F1042M/F439W/F673N/F953N/F791W', 'IHW_(17-19)', 'MIRROR-A2', 'F439W/F555W/F702W', 'Kprime + H2-1-0-S1', 'F160AW', 'Ooff OIII+30nm k1031', 'FR418N/FR533N', 'F220W/F555W/F435W/F330W/F814W/F6', 'F175W;F275W', 'G140L;G230L;G430L', 'F606W/F658N/F502N', 'F550M/F658N', 'F164N;F405N', 'NBF 403', 'F791W/F673N/F1042M/F850LP', 'i.MP9703', 'E230M;G140M', 'F656N/F410M/F547M/F1042M', 'F435W/F606W', 'FQ492N', 'FR868N18', 'CH4-H4S + H2-1-0-S1', 'RG780 + i', 'F2ND;CLEAR2;F1ND;F437M', 'F502N/F673N', 'BC', 'F300W/F656N/F606W/F814W/F502N/F4', 'F475W/F775W/F625W', 'VR Bernstein k 1040 0', 'F547M/F469N/F439W/F656N', 'F2ND;CLEAR2;F231M;CLEAR4', 'z SDSS c6020', 'F360M;MASKBAR', 'FR418P15', 'F336W/F255W/F170W/F439W/F555W', 'SDSS i k1586', 'F336W/F656N/F675W', 'F555W/FQUVN', 'Ha6 7288', 'F185W/F122M/F380W/F300W/F375N/F3', 'F128B2', 'F190N', 'tio.MP7701', 'G230LB;G230M', 'SDSS-z k1587', 'F2ND;CLEAR2;CLEAR3;F372M', 'G096', 'COnar.WC8399', 'F360M;GRISMC', 'F336W/F555W/F606W/F814W/F439W/F6', 'FQ634N', 'F502N/F547M', 'W5135 90 w5135', 'F1065C', 'FR868N33/FR868N18/FR868P15/FR868', 'F502N/F547M/F673N/F657N', 'r Sloan KP1585', 'F4ND;F275W;CLEAR3;CLEAR4', 'F070LP;MIRROR', 'CaT + HartmannC', 'F212N/F215N/F222M/F160W', 'F336W/F225W/F656N', 'F430M;MASKRND', 'F185B9', 'F212N;F250M', 'F128N/F164N/F130N/F167N/F110W/F1', 'F336W/F170W/F255W', 'K-red', 'F204M', 'G140L;G230L;G230MB', 'F212N;F470N', 'F336W/F170W/F255W/F555W/F814W/F4', 'F814W/F275W/F438W', 'FR418N33;F437N', 'F2ND;F342W;F1ND;CLEAR4', 'F550M/F606W;POL60V/F606W;POL0V/F', 'F785LP/F850LP/F953N/F1042M/F622W', 'F850LP/F555W/F450W', 'POL120;CLEAR2;CLEAR3;F437M', 'F395N/F280N/F656N', 'CLEAR1S;PR200L', 'F658N/F791W/F547M', 'HeI', 'ha16', 'Z + z', 'F187N;F430M', 'I_Hrris', 'Bess-B sBessB', 'F6ND;F342W;F1ND;F372M', 'VR Supermacho c 6027', 'Stromgren V', 'F130N/F110W/F132N', 'F180M/F237M/F222M/F160W/F204M/F1', 'F336W/F814W/F438W', 'F657N/F645N/F775W/F625W', 'W-J-U', 'MIRCUV', 'N-B-L497', 'u DECam c0006 3500.0 1000.0', 'F160BN15;F130LP', 'F658N/F606W/F555W/F625W', 'G750L', 'F656N/F502N', 'o2 OII c6012', 'BG38', 'BLOCK4', 'FR680P15/F814W/F547M/FR533N', 'H Br(gamma)', 'F675W', 'F110M/F108N', 'F502N/F657N', 'F6ND;F320W;CLEAR3;CLEAR4', 'F487N', 'FQCH4N/F410M/F255W/F953N', 'F275W', 'F336W/F487N/F438W/F555W', 'F475W/F606W/F555W/F625W', 'F410M/F656N/F1042M/F588N', 'F336W/F275W/F225W', 'F547M/F502N', 'F200W;F430M', 'POL0UV;F660N', 'F105W/F160W/F125W/F140W', 'F212N/F215N', 'F622W/F656N', 'F220W/F550M/F850LP/F330W/F435W/F', 'F205W', 'F375N', 'F145M', 'F160W/F125W/F098M', 'F814W;F122M', 'M', 'F237M/F205W/F187W/F207M', 'F125W/F140W/F105W/F160W/F110W/F1', 'F2ND;CLEAR2;F1ND;F502M', 'CH4-H4L + J', 'r-SDSS', 'CLEAR1;F140W;F152M;CLEAR4', 'F150W2;F322W2', 'CH4-H4L + Block', 'y Stromgren', 'F439W/F170W/F555W/F658N/F814W/F6', 'FQCH4P15/FQCH4N/F673N/F953N', 'F182M;F335M', 'F8ND;F342W;CLEAR3;F307M', 'wrc4 WR CIV k1024', 'z DECam SDSS c0004 9260.0 1520.0', 'FR418N/FR868N/FR533N', 'F569W/F791W/F656N/F673N', 'F656N/F469N', 'F200W/GR150R', 'F160W/F175W', 'F555W/F606W/F814W/F439W/F450W/F6', 'POL120;F555W', 'CLEAR1;CLEAR2;CLEAR3;F278M', 'F4ND;F430W;CLEAR3;CLEAR4', 'X', 'PRISM1;CLEAR2;F152M;CLEAR4', 'F8ND;F889N', 'F555W/F435W/F550M/F814W/F475W/F6', 'F105W', 'CLEAR1;F175W;F152M;CLEAR4', 'F814W/F656N/F555W/F475W', 'F814W/F606W/F450W/F656N', 'F336W/F814W/F555W/F160BW', 'POL0;F320W;F1ND;CLEAR4', 'FR680P15/F791W', 'F164N/F160W/F125W/F110W/F098M/F1', 'F212N/F215N/F110W/F205W', 'FR680N;F631N', 'HartmannA + r', 'F225W/F218W', 'F150W/GR150R', 'PRISM3;F130LP', 'KP1513 Osmer/Green 7500', 'F814W/F330W/F606W/F475W', 'oiii_pfw', 'F487N/F343N/F469N/F390N', 'F555W;F606W/F814W;F791W', 'F336W/F656N', 'F555W/F437N', 'F145M;F145M', 'Lprime + Br-gamma', 'GG 495', 'CLEAR1;F342W;CLEAR3;CLEAR4', 'F814W/FR680N/F555W', 'OIIIoff k1015', 'W16', 'F336W/F255W/F170W', 'F220W/F502N', 'F070W;F430M', 'F814W/F502N/F555W', 'F4ND;CLEAR2;F210M;CLEAR4', 'Si3-9.7um + N', 'F110W/F160W/F140W', 'F305LP;F275W', 'hydrocarb', 'F110W/F167N/F160W', 'F128N/F164N/F130N/F167N/F126N/F1', 'F550M/F660N', 'F605W', 'OG570new', '4.0(cont)', 'tres filtros rojos + free hole', 'CH4-H1L + FeII', 'G130M;G160M;G430M', 'F212N/F215N/F205W', 'G130M;G430L', 'F375N/F673N/F953N', 'F814W/FR680N/F547M/FR533N', 'F335M;GRISMC', 'W013_Ha_wide', 'F656N/F673N/F675W/F702W', 'F438W/F625W/F657N', 'K-continuum', 'H-alpha+ 8nm c6011', 'YPHOT', 'F115W;MASKRND', 'I-A-L445', 'F166N/F222M', 'QuotaU test', 'F502N/F547M/F673N/F656N/F814W', 'POL120;CLEAR2;F170M;CLEAR4', 'F108N', 'F2ND;CLEAR2;CLEAR3;F410M', 'F336W/F438W/FQ508N/F775W/F680N', 'POL60;F320W;CLEAR3;CLEAR4', 'F475W/F658N/F606W/F502N/F814W', 'W21', 'NOCHANGE;F650W', 'W12', 'F375N/F502N/F656N/F953N', 'CH4-H1Sp + CH4-H1Sp', 'F240M', 'G230L;G430L;G430M', 'N-B-L973', 'F210M;WLP8', 'CH12-LONG', 'CaT + z', 'F814W/FR680N/F547M/FR533P15', 'F336W;POLQN18', 'F160BW;F606W', 'F164N/F166N', '7799', 'W15', 'z SDSS 1020', 'DDS 51 c6008', 'F150LP', 'CLEAR1;CLEAR2;CLEAR3;F550M', 'F336W/F438W', 'SN5', 'VR', 'CLEAR;F115W', 'SII', 'POL60;F675W', 'F718M', 'ha8 H-alpha+8nm c6011', 'POL120UV;F250W', 'F150W;F275W', 'F620W', 'F625W;POL60V', 'F336W/F487N/F547M/F814W/F502N/F6', 'K1566_6709_71', 'ZJ', 'F212N;F300M', 'F336W/F255W/F631N', 'F547M/F814W/F170W', 'E140H;E140M', 'F410M/F953N/FQCH4P15/F555W/F673N', 'F375N/F656N/F502N/F437N', 'ha16 H-alpha+16nm k1013', 'PRISM1;CLEAR2;F120M;CLEAR4', 'CN.MP7803', 'F439W/F555W/F658N/F469N/F502N/F6', 'F814W/F656N/F555W/F675W', 'PRISM2;CLEAR2;F195W;CLEAR4', '255', 'F255W/F702W', 'Y', 'sii_pfw', 'F555W;POLQN33', 'F336W/F814W/F606W', 'F814W/CLEAR_HRC/F555W', 'FQ937N', 'F110W/F170M', 'F187N', 'F435W/F814W/F555W/F625W', 'F115W/GR150R', 'F673N;FR680N33', 'F165M/F095N/F190N/F145M/F090M', 'GG385', 'F892N;CLEAR2L', 'F182M;NONE', 'F814W/F702W', 'F814W/F438W/F606W/F657N', 'JPHOT', 'F953N/F656N/F791W/F673N/F658N/F5', 'I c6028', 'F175W;CLEAR2', 'F390M', 'I-A-L464', 'unknown', 'NeII_ref2-13.1um', 'F1280W', 'F336W/F673N/F588N/FQUVN', 'F180M/F222M/F190N/F165M/F160W/F1', 'Blue-u', 'E230M;G130M;G160M;G185M', 'F8ND;F675W', 'F648M;F122M', 'IRAS-12um', 'Ca HK', 'F110W/F132N', 'G230M', 'WG360', 'F164N/F125W/F098M/F105W/F140W/F1', 'F150W2;F162M', 'F2ND;CLEAR2;F1ND;CLEAR4', 'GRISM2', 'F130N/F125W', 'DECODERR', 'F336W/F438W/F606W', 'F1800W', 'F547M/F656N/F673N', 'F375N/F502N/F656N', '[SII]', 'PRISM2;CLEAR2;F165W;CLEAR4', 'CLEAR1;CLEAR2;F1ND;F253M', 'FR647M', 'F390W/F475X/F600LP', 'F656N/F673N/F631N/F658N/F547M/F5', 'F2ND;F480LP;F1ND;CLEAR4', 'G140L;G140M;G230L;G430L', 'F105W/F110W/F160W/F125W/F140W', 'F336W/F255W/F439W/F555W/F160BW', 'F110M/F165M/F090M', 'F336W/F255W/F170W/F343N', 'Us solid U k1044', 'F467M/F665N/F814W', 'macho_r', 'F350LP/F336W/F275W', 'GRAT_ERR', 'Harris V', 'gri', 'BrG', '493 BATC k1055', 'F110W/F170M/F090M', 'Blue-g', 'F128N/F164N/F126N', 'F225W/F218W/F606W/F438W', 'OPAQUE;MIRROR', 'Johnson I', 'F814W/F438W/F555W', 'F4ND;F370LP;CLEAR3;CLEAR4', 'F606W/F656N', 'F814W/F850LP', 'F435W/F625W', 'CH4-H1L + Block', 'F622W/F850LP', 'W016 6618/72', 'HaleBopp CN 1', 'CLEAR1;F430W', 'F200W/GR150C', 'HETG', 'F220W/F550M', 'H2O_ice', 'FeII (5211)', 'F150W/F175W', 'Kcont + CH4-H4S', 'POL120S', 'POL0;F140W;CLEAR3;CLEAR4', 'g SDSS k1017', 'F164N/F110W/F127M', 'FQCH4P15/F850LP/F673N/F1042M', 'F673N/F625W', 'F194W', 'F336W/F439W/F555W/F656N/F814W/F5', 'F375N/F343N/F555W/F487N/F658N/F4', 'F953N/F469N/F673N/F373N', 'F631N;FR680N18', 'F656N/F658N/F502N/F588N', 'green_pfw', 'F460M;GRISMC', 'F814W/F225W', 'F502N/F673N/F631N/F487N/F658N/F6', 'F502N;FR533N18', 'F814W/F606W/F450W/F656N/F300W', 'CLEAR1;PRISM2', 'F275W/F225W', 'V_Harris', 'N502', 'F547M/F656N/F658N/F502N/F673N', '9440', 'F550M;POL0V', 'B_Harris_W2', 'F656N/F675W/F631N', 'F207M', 'F850LP/F606W', 'IRAS-100um', 'F237M/F216N/F187N/F190N/F215N/F2', 'Jlow', 'NOCHANGE;F605W', 'FQ232N', 'F665N/F625W', 'w15', 'U Harris (solid)', 'F718M;F122M', 'F555W/F450W/F702W', 'F435W/F850LP/F606W/F775W', 'OG570 kOG570', 'F205W/F187N', 'F2550W', 'F430M;F140M', 'F122M;F675W', 'F467M/F547M/F953N', 'F187N/F212N/F166N/F190N/F164N/F2', 'F140W/F098M/F167N/F110W/F125W/F1', 'F230W', 'WLP4;F444W', 'Hb-on', 'F128N/F160W/F140W/F105W/F098M/F1', 'F547M/F656N/F502N/F673N/F814W', 'F814W/F850LP/F702W', 'Qone-17.8um', 'KP1565', 'F128N/F140W/F105W/F167N/F098M/F1', 'F550M/F435W/F850LP/F775W', 'F368M;F122M', 'F220W/F435W/F330W/F475W/F250W', 'Qbranch', 'F770W', 'F187N;MASKBAR', 'F336W/F300W/F170W/F160BW/F255W', 'F160W/F167N', 'F785LP/F850LP/F1042M/F953N/F622W', 'E140M;G230L;G430L', 'CLEAR1L;F814W', 'F222M/F166N/F190N/F240M/F160W/F2', 'F487N/F547M/F502N', 'E230M;G140L;G140M;G430L', 'F656N/F673N/F487N/F658N/F547M/F5', '3290', 'F150W;F444W', 'F814W/F606W/F555W/F675W', 'F4ND;F501N;F1ND;CLEAR4', 'n.a.', 'FR680P15/F791W/F547M/FR533N', 'F8ND;F439W', 'CLEAR1;F220W;CLEAR3;CLEAR4', 'POL60UV;F435W', 'F8ND;CLEAR2;F195W;CLEAR4', 'Cousins R', 'F814W/F475W/F555W', 'CH4', 'F658N;POL120V', 'F187N;F470N_F444W', 'Ks', 'F438W/F606W', 'F200W;F335M', 'CO.WC8306', 'F606W/F656N/F450W/F814W', 'F150W;F360M', 'CLEAR1;CLEAR2;F152M;CLEAR4', 'F547M/F218W/F439W/F300W', 'F095N/F166N/F190N/F113N/F145M', 'Rc', 'F850LP/F775W/F625W', 'F390N/F658N/F656N/F673N', 'F160W/F090M', 'F675W;POLQ', 'F410M;POLQP15', 'F164N/F108N/F113N', 'F430M;F158M', 'G-A-R030', 'F588N;FR533N', 'F4ND;F220W;CLEAR3;F278M', 'F814W/F555W/F673N', 'Nprime', 'F180M/F110W/F222M/F207M', 'F673N/F702W', 'F435W/F814W/F330W/F850LP/F555W', 'JH', 'F105W/F098M/F160W/F140W', 'H-alpha narrow', 'G130M;G160M;G185M;G285M', 'KP1496', 'F547M/F673N', 'F140M;WLP8', 'F658N/F791W/F502N', 'SDSS-z cSDSSz', 'F218W', 'C_Washington c6006', 'F850LP/F702W', 'NB921', 'HartmannA', 'W08', 'F212N;F466N', 'F336W/F439W/F170W/F555W/F502N/F8', 'F467M/F622W/F588N/F631N', 'H1', 'F6ND;CLEAR2;F140M;CLEAR4', 'F237M/F222M/F187N/F212N/F190N/F2', 'Unknown', 'F164N/F108N/F222M/F113N', 'F212N;F335M', 'F195W;CLEAR2', 'F105W/F160W/F125W', 'H2_2-1_S1', 'Y.WC8002', 'Harris B', '6.456', 'F255W/F170W/F555W', 'F210M;F335M', 'ECAS x', 'F502N;FR533N', 'F502N/F673N/F657N', 'HeI(2p2s)', 'F336W/F410M/F953N/FQCH4P15/F255W', 'F225W/F275W/F218W', 'GR150C;F150W', 'F658N/F656N/F673N/F675W', 'F814W/F657N/F502N/F555W/F673N', 'F763M/F845M/F621M/F555W/F689M', 'F475W/F606W/F555W/F814W', 'detection', 'G160M', 'F785LP/F791W/F1042M/F622W/F850LP', 'F336W/F547M', 'CH4-H4L + CH4-H4S', 'F336W/F255W/F555W/F160BW', 'F4ND;F220W;F1ND;F278M', 'F300W/F814W/F656N/F450W/F555W', 'I-A-L797', 'F128N/F110W/F160W/F098M', 'G160M;G185M;G225M', 'F170W/F1042M', 'F336W/F275W/F280N/F410M/F373N/F3', 'F550M/F850LP/F606W', 'F814W/F475W/F625W', 'F763M/F845M', 'r SDSS c6018', 'F105W/F110W/F160W', 'POL60;F275W;CLEAR3;CLEAR4', 'JX', 'Donats', 'CLEAR1;PRISM1', 'M Washington k1007', 'g DECam SDSS c0001 4720.0 1520.0', 'F875M;F122M', 'Mp', 'W-J-VR', 'F115W/GR150C', 'F622W/F785LP', 'F555W/F953N', 'F212N;F460M', 'Z + HartmannD', 'F814W/F439W/F555W/F675W', 'F702W/F450W/F555W', 'FQUVN/F555W', 'F437N;F1042M', 'F336W/F814W/F439W/F555W/F656N', 'F200W;MASKRND', 'PAH1-8.6um', 'F210M;F250M', 'Cousins I', 'F336W/F275W/F438W/F390W/F475W/F8', 'F814W/F555W/F588N', 'F190N/F095N/F145M', 'F164N/F167N', 'FR459M', ...}, 'telescope_name': {'ACP->ASA N-8 Dark Ridge Observatory', 'ACP->TheSky Cerro Tololo Inter-American Observatory', 'ACP->TheSky Dark Ridge Observatory', 'ALMA-12m', 'ALMA-7m', 'ALMA-TP', 'ASKAP', 'Apache Point 1.0m', 'Apache Point 2.5m', 'BLAST', 'BRITE-AUSTRIA', 'BRITE-HEWELIUSZ', 'BRITE-LEM', 'BRITE-Toronto', 'CFHT 3.6m', 'CTIO-1.5m', 'Chandra X-ray Observatory', 'DAO 1.2-m', 'DAO 1.8-m', 'DAO Skycam', 'DRAO-ST', 'EVLA', 'FCRAO', 'FUSE', 'Gemini-North', 'Gemini-South', 'HST', 'IRAS', 'JCMT', 'JWST', 'MOST', 'Mt. Stromlo Observatory 50 inch', 'NEOSSat', 'No Value', 'None', 'OMM-1.6', 'OMM-1.6m', 'SUBARU', 'Subaru', 'TESS', 'UKIRT', 'UniBRITE', 'VLA', 'WIYN', 'XMM-Newton', 'bok23m', 'ct09m', 'ct13m', 'ct15m', 'ct1m', 'ct4m', 'gscp8m', 'kp09m', 'kp21m', 'kp4m', 'kpcf', 'soar', 'unknown', 'wiyn'}, 'intent': {'calibration', 'science'}, 'collection': {'ALMA', 'APASS', 'BLAST', 'BRITE-Constellation', 'CFHT', 'CFHTMEGAPIPE', 'CFHTTERAPIX', 'CFHTWIRWOLF', 'CGPS', 'CHANDRA', 'DAO', 'DAOCADC', 'DAOPLATES', 'DRAO', 'FUSE', 'GEMINI', 'GEMINICADC', 'HST', 'HSTHLA', 'IRIS', 'JCMT', 'JCMTLS', 'JWST', 'MACHO', 'MOST', 'NEOSSAT', 'NGVS', 'NOAO', 'OMM', 'POSSUM', 'RACS', 'SDSS', 'SUBARU', 'SUBARUCADC', 'TESS', 'UKIRT', 'VGPS', 'VLASS', 'WALLABY', 'XMM'}, 'type': {'ACQUIRE', 'ACQUISITION', 'ALIGN', 'ARC', 'ASTAR', 'BIAS', 'BPM', 'CAL', 'CALIB', 'COMP', 'COMPARISON', 'CORONAGRAPHIC', 'DARK', 'DATACUBE', 'DIM', 'DOME-FLAT', 'DOMEFLAT', 'DOME_FLAT', 'ENG', 'EXT', 'FIELD_ACQ', 'FLAT', 'FLATFIELD', 'FOCUS', 'FOCUSING', 'FRINGE', 'GUIDE', 'ILLUM', 'IMAGING', 'INDEF', 'INTERNAL', 'LED', 'MASK', 'NULL', 'OBJECT', 'PARALLEL_COORDINATED', 'PARALLEL_PURE', 'PINHOLE', 'PRIME', 'PSFSTANDARD', 'READOUT', 'REF', 'RONCHI', 'SCIENCE', 'SKY', 'SKYFLAT', 'SKY_FLAT', 'SLIT', 'SPECTROSCOPIC', 'STANDARD', 'STANDARD_OB', 'STANDARD_ST', 'STANDARD_STAR', 'STARING DARK', 'STARING FLAT', 'STARING OBJECT', 'TEST', 'TSTPAT', 'UNKNOWN', 'VVOBJECT', 'WEIGHT', 'ZERO', '_none_', 'acquire', 'arc', 'bias', 'calibration', 'comp', 'comparison', 'dark', 'dream', 'eng', 'flat', 'flatfield', 'focus', 'fringe', 'grid', 'guider', 'illum', 'jiggle', 'junk', 'noise', 'object', 'pointing', 'pupil', 'red', 'reduc', 'reject', 'remap', 'scan', 'setup', 'sky', 'skydip', 'standard', 'stare', 'targetacq', 'test', 'unknown'}, 'energy_emBand': {'Infrared', 'Infrared|Optical', 'Infrared|Optical|UV', 'Infrared|Optical|UV|EUV', 'Infrared|Optical|UV|EUV|X-ray|Gamma-ray', 'Millimeter', 'Millimeter|Infrared', 'Optical', 'Optical|UV', 'Optical|UV|EUV', 'Optical|UV|EUV|X-ray', 'Optical|UV|EUV|X-ray|Gamma-ray', 'Radio', 'UV', 'UV|EUV', 'UV|EUV|X-ray', 'X-ray'}, 'calibrationLevel': {'-1', '0', '1', '2', '3', '4'}, 'instrument_name': {'1872 RETICON', '90prime', 'ACIS-I', 'ACIS-S', 'ACS', 'ACS/HRC', 'ACS/SBC', 'ACS/WFC', 'ALAIHI-ACSIS', 'AOSC', 'APOGEE Spectrograph', 'ARCOIRIS', 'ASKAP', 'AWEOWEO-ACSIS', 'Alopeke', 'Apogee USB/Net', 'BEAR', 'BLAST', 'BOSS Spectrograph', 'BRITE-AUSTRIA', 'BRITE-HEWELIUSZ', 'BRITE-LEM', 'BRITE-Toronto', 'Band 3', 'Band 4', 'Band 5', 'Band 6', 'Band 7', 'CBE', 'CCD', 'CFH12K MOSAIC', 'CFHTIR', 'CGS3', 'CGS4', 'CIAO', 'CIRPASS', 'COS', 'COS-STIS', 'COS/FUV', 'COS/NUV', 'COUDE_F8', 'CPAPIR', 'Cassegrain Spectrograph', 'Cassegrain Spectropolarimeter', 'DAS', 'DRAO-ST', 'Direct image', 'EMOS1', 'EMOS2', 'EPIC PN', 'ESPaDOnS', 'F2', 'FCRAO', 'FCS', 'FES', 'FGS', 'FGS/FGS1', 'FGS/FGS2', 'FILAO', 'FLAMINGOS', 'FOC/288', 'FOC/48', 'FOC/96', 'FOCAM', 'FOCAS', 'FOS/BL', 'FOS/RD', 'FTS', 'FTS2-SCUBA-2', 'FUV', 'Fabry image', 'GECKO', 'GHOST', 'GMOS', 'GMOS-N', 'GMOS-S', 'GNIRS', 'GPI', 'GRACES', 'GRIF', 'GSAOI', 'Guide star', 'HARP-ACSIS', 'HERZBERG', 'HRC-I', 'HRC-S', 'HRCAM', 'HRS', 'HRS/1', 'HRS/2', 'HSC', 'HSP/PMT/VIS', 'HSP/POL', 'HSP/UNK/PMT', 'HSP/UNK/POL', 'HSP/UNK/UV1', 'HSP/UNK/UV2', 'HSP/UNK/VIS', 'HSP/UV1', 'HSP/UV2', 'HSP/UV2/UV1', 'HSP/VIS', 'HSP/VIS/PMT', 'Hokupaa+QUIRC', 'IFD', 'IGRINS', 'IRAS', 'IRCAM3', 'ISIS', 'KUNTUR-ACSIS', 'MACHO', 'MARLIN', 'MARVELS Spectrograph', 'MIRI', 'MIRI/CORON', 'MIRI/IFU', 'MIRI/IMAGE', 'MIRI/SLIT', 'MIRI/SLITLESS', 'MIRI/TARGACQ', 'MOCAM', 'MOS', 'MOSFP', 'MPIRXE-DAS', 'McKellar Spectrograph', 'MegaPrime', 'Michelle', 'NEOSSat_Science', 'NEOSSat_StarTracker', 'NICI', 'NICMOS', 'NICMOS/NIC1', 'NICMOS/NIC2', 'NICMOS/NIC3', 'NIFS', 'NIRCAM', 'NIRCAM/CORON', 'NIRCAM/GRISM', 'NIRCAM/IMAGE', 'NIRCAM/TARGACQ', 'NIRI', 'NIRISS/AMI', 'NIRISS/IFU', 'NIRISS/IMAGE', 'NIRISS/MSA', 'NIRISS/SLIT', 'NIRISS/SOSS', 'NIRISS/WFSS', 'NIRSPEC', 'NIRSPEC/IFU', 'NIRSPEC/IMAGE', 'NIRSPEC/MSA', 'NIRSPEC/SLIT', 'NULL', 'Newtonian Imager', 'OASIS', 'OSCIR', 'OSIS', 'Optical Monitor', 'PALILA', 'PHOENIX', 'POL-HARP-ACSIS', 'POL-RXA3-ACSIS', 'POL-RXWB-ACSIS', 'POL2-FTS2-SCUBA-2', 'POL2-SCUBA-2', 'PUEO', 'PUMA', 'PYTHIAS', 'Photometer', 'REDEYE', 'RGS1', 'RGS2', 'RXA', 'RXA2-DAS', 'RXA3-ACSIS', 'RXA3-DAS', 'RXA3M-ACSIS', 'RXB2', 'RXB2-DAS', 'RXB3-DAS', 'RXB3I-DAS', 'RXC', 'RXC2-DAS', 'RXWB-ACSIS', 'RXWC-DAS', 'RXWD-DAS', 'RXWD2-ACSIS', 'S-WIDAR', 'SCUBA', 'SCUBA-2', 'SDSS Camera', 'SDSS Spectrograph', 'SIS', 'SISFP', 'SITELLE', 'SPIRou', 'STIS', 'STIS/CCD', 'STIS/FUV-MAMA', 'STIS/NUV-MAMA', 'ScienceCCD', 'Sky Camera', 'Suprime-Cam', 'TEXES', 'TIGER', 'TReCS', 'UFTI', 'UH8K', 'UH8K MOSAIC CAMERA', 'UIST', 'UKT14', 'UU-ACSIS', 'UVPRIME', 'UniBRITE', 'VLA', 'WFC3', 'WFC3/IR', 'WFC3/UVIS', 'WFCAM', 'WFPC/PC', 'WFPC/WFC', 'WFPC2', 'WFPC2/PC', 'WFPC2/WFC', 'WIRCam', 'WVM', 'Zorro', 'andicam', 'aobir', 'aobvis', 'arcoiris', 'bHROS', 'bench', 'ccd_imager', 'ccd_spec', 'chiron', 'cosmos', 'cpapir', 'decam', 'echelle', 'flamingos', 'goodman', 'hrwfs', 'hydra', 'ir_imager', 'ispi', 'kosmos', 'michelle', 'mosaic3', 'mosaic_1', 'mosaic_1_1', 'mosaic_2', 'newfirm', 'optic', 'osiris', 'sami', 'soi', 'spartan', 'whirc', 'wttm', 'y4kcam'}}
   obs_columns_set = {'telescope_name', 'target_type', 'type', 'metaReadGroups', 'target_moving', 'collection', 'environment_ambientTemp', 'target_name', 'target_keywords', 'metaChecksum', 'environment_wavelengthTau', 'telescope_geoLocationZ', 'instrument_keywords', 'obsID', 'proposal_title', 'environment_tau', 'targetPosition_coordinates_cval2', 'typeCode', 'metaProducer', 'observationID', 'proposal_project', 'telescope_geoLocationX', 'proposal_pi', 'targetPosition_coordsys', 'accMetaChecksum', 'instrument_name', 'lastModified', 'requirements_flag', 'target_targetID', 'sequenceNumber', 'targetPosition_coordinates_cval1', 'environment_humidity', 'intent', 'algorithm_name', 'maxLastModified', 'telescope_keywords', 'environment_elevation', 'target_standard', 'telescope_geoLocationY', 'members', 'targetPosition_equinox', 'environment_seeing', 'metaRelease', 'proposal_id', 'target_redshift', 'environment_photometric', 'observationURI', 'proposal_keywords'}
   plane_columns_set= {'metaReadGroups', 'time_bounds_samples', 'energy_freqWidth', 'position_bounds_samples', 'dataProductType', 'position_dimension_naxis2', 'provenance_runID', 'lastModified', 'energy_resolvingPower', 'provenance_name', 'position_sampleSize', 'energy_bounds_width', 'position_dimension_naxis1', 'energy_sampleSize', 'metrics_magLimit', 'time_resolutionBounds', 'provenance_producer', 'energy_resolvingPowerBounds', 'energy_bounds_upper', 'provenance_inputs', 'provenance_keywords', 'time_bounds_upper', 'polarization_states', 'energy_restwav', 'position_timeDependent', 'energy_dimension', 'metaProducer', 'quality_flag', 'custom_bounds_upper', 'energy_transition_transition', 'energy_bounds', 'time_bounds_width', 'provenance_version', 'planeID', 'custom_bounds_samples', 'planeURI', 'creatorID', 'metaRelease', 'observable_ucd', 'custom_bounds', 'provenance_reference', 'energy_bounds_lower', 'publisherID', 'metrics_background', 'polarization_dimension', 'custom_dimension', 'metaChecksum', 'time_resolution', 'provenance_project', 'obsID', 'metrics_backgroundStddev', 'dataRelease', 'accMetaChecksum', 'time_sampleSize', 'position_resolutionBounds', 'time_bounds', 'time_dimension', 'calibrationLevel', 'energy_transition_species', 'custom_ctype', 'metrics_sourceNumberDensity', 'energy_bounds_samples', 'energy_emBand', 'position_resolution', 'metrics_fluxDensityLimit', 'position_bounds_size', 'dataReadGroups', 'custom_bounds_width', 'custom_bounds_lower', 'productID', 'energy_freqSampleSize', 'time_exposure', 'energy_bandpassName', 'position_bounds', 'maxLastModified', 'provenance_lastExecuted', 'time_bounds_lower', 'energy_energyBands'}


   if column not in obs_columns_set and column not in plane_columns_set:
       return column + "column does not exist in the database"
   if column not in distinct_values_dict:
       return column + "column does exist but is not valid for this tool"


   vector_db = column_dbs[column]


   if value not in distinct_values_dict[column]:
       query = "What is the most similar to " + value + "?"
       matching_docs = vector_db.similarity_search(query)
       page_contents = [doc.page_content for doc in matching_docs[:5]]
       return value + " is not a value that exists in the " + column +" column, try one of these" + str(page_contents)


   else:
       if column in obs_columns_set:
         return value + " is a value that exists in the " + column + " column in the observation table"
       else:
         return value + " is a value that exists in the " + column + " column in the plane table"
   return str(result)


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

You must never generate sql code. Only enhance the user's input.
**Instructions:**
You must use the tools in the following order. Do not pass a column into checkColumns without before passing it into alternateColumn to see if it needs to be changed.
1. **alternateColumn**: Always start by verifying if any column names in the user's input need to be updated or corrected based on the database schema. Use this tool to find the correct column names if needed.
2. **checkColumns**: Next, ensure that all mentioned columns exist in the specified table(s). Confirm which columns belong to which table(s) and that the columns in the user's input are valid.
3. **alternateValue**: Use this tool to check if any values (like specific names or IDs) in the user's input need to be corrected. Use this tool to verify and update values as necessary.
4. **checkValuesForCommonColumns**: Finally use this tool to check to see if a value exists in a column. The output will indicate if the value exists or if it does not exist.
**Important:** You must use these tools in the order provided for every query.


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
        tools = [alternateColumn,checkColumns, alternateValue, checkValuesForCommonColumns]

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
