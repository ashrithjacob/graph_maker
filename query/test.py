import sys 
# append the path of the parent directory
sys.path.append("..")

from graph_maker import GraphMaker, Ontology, GroqClient, OpenAIClient
from graph_maker import Document
import re
import os
from openai import OpenAI
from graph_maker import Document
import pandas as pd
import pyvis
from pyvis.network import Network
from graph_maker import GraphMaker, Ontology, GroqClient, OpenAIClient
from falkordb import FalkorDB
from langchain_core.prompts import PromptTemplate

client = OpenAI()

class ProcessConversation:
    CORESOLUTION_SYSTEM_PROMPT = """
                                    You are a coresolution resolver, which is to say you replace all pronouns referencing an object
                                    with it's proper noun based on the context.
                                    Ensure to also infer who 'I' refers to based on the context and replace all Is witht the proper noun.
                                    Example input:
                                    `Elon Musk was born in South Africa. There, he briefly attended classes at the University of Pretoria`

                                    Example output:
                                    `Elon Musk was born in South Africa. In South Africa, Elon briefly attended classes at the University of Pretoria`'

                                    Keep the same structure of the text as passed to you in the text body.
                                """
    SPEAKER_MAPPING = {"SPEAKER_00": "Jeff Bezos", "SPEAKER_01": "Lex Fridman"}

    #extract text from file
    @classmethod
    def _text_from_file(cls, file):
        with open(file) as f:
            text = f.read()
        return text

    @classmethod
    def create_outfile_path(cls, infile_file_path):
        return os.path.splitext(infile_file_path)[0] + "_renamed.txt"
    
    # split text into sentences
    @classmethod
    def sentence_splitter(cls, text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return sentences

    #create Document format
    @classmethod
    def create_docs(cls, sentences):
        docs = [Document(text=sentence["text"], metadata={"num": str(idx), "speaker":sentence["speaker"]}) for (idx, sentence) in enumerate(sentences)]
        return docs

    # run coreference resolution
    @classmethod
    def coreference_resolution(cls, text):
        #TODO: entire text is dumped for the time being, need to split it into chunks if it exceeds the limit
        response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": cls.CORESOLUTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"{text}"},
        ]
        )
        return response.choices[0].message.content

    @classmethod
    # replace dummy mappings with actual names
    def _replace_speaker_with_mapping(cls, line):
        for key, value in cls.SPEAKER_MAPPING.items():
            if key in line:
                line = line.replace(key, value)
        return line
    
    @classmethod
    #store texts and speakers in a list of dictionaries
    def _extract_speaker_and_text(cls, sentences):
        speaker_text_store = []
        for input_string in sentences:
            print(input_string)
            speaker_match = re.search(r'speaker:(.*?)\|', input_string).group(1).strip()
            text_match = re.search(r'text:(.*)', input_string).group(1).strip()
            speaker_text_store.append({"text":text_match, "speaker":speaker_match})
        return speaker_text_store
    
    @classmethod
    #clump sentences together based on speaker and chunk limit, whichever comes first
    def clump_sentences(cls, sentences, chunk_limit):
        current_speaker = sentences[0]["speaker"]
        text = ""
        chunk_number = 0
        clumped_sentences = []
        for sentence in sentences:
            if sentence["speaker"] == current_speaker and chunk_number < chunk_limit:
                text += sentence["text"]
                chunk_number += 1
            elif sentence["speaker"] != current_speaker or chunk_number >= chunk_limit:
                clumped_sentences.append({"text": text, "speaker":current_speaker})
                current_speaker = sentence["speaker"]
                text = sentence["text"]
                chunk_number = 1
        else:
            clumped_sentences.append({"text": text, "speaker":current_speaker})
        return clumped_sentences

    @classmethod
    def process_sentences(cls, sentences, chunk_limit):
        speaker_text_store = cls._extract_speaker_and_text(sentences)
        sentences_clumped = cls.clump_sentences(speaker_text_store, chunk_limit)
        return sentences_clumped

    @classmethod
    #write conversation with correct speaker names to file
    def write_conversation_to_file(cls):
        # Replace speaker in conversation:
        if not os.path.exists(cls.outfile_file_path):
            with open(cls.infile_file_path, 'r') as infile, open(cls.outfile_file_path, 'w') as outfile:
                # Read the lines from the input file
                for line in infile.readlines():
                    processed_line = cls._replace_speaker_with_mapping(line)
                    outfile.write(processed_line)

    @classmethod
    #run the entire preprocessing pipeline
    def run(cls, infile_file_path):
        cls.infile_file_path = infile_file_path
        cls.outfile_file_path = cls.create_outfile_path(infile_file_path)
        cls.write_conversation_to_file()
        text = cls._text_from_file(cls.outfile_file_path)
        coresolved_text = cls.coreference_resolution(text)
        sentences = cls.sentence_splitter(coresolved_text)
        sentences_processed = cls.process_sentences(sentences, chunk_limit=3)
        docs = cls.create_docs(sentences_processed)
        return docs


class CreateGraph:
    GRAPH_CLEANER_PROMPT = """ 
                            You are an expert at generating open cypher queries.
                            You are provided with a list of cypher queries for creation of new nodes and edges.
                            Your duty is to simplify the queries so that the relationships fall into categories preferably.
                            Feel free to rearrange or add new nodes if required to ensure the best relationships are captured.
                            Do not change or remove any details as provided by the query.
                            Ensure to use metadata in the relationship elements to capture any details that cannot be captured otherwise

                            Examples:
                            CREATE (charlie:Person:Actor {name: 'Charlie Sheen'})-[:ACTED_IN {role: 'Bud Fox'}]->(wallStreet:Movie {title: 'Wall Street'})<-[:DIRECTED]-(oliver:Person:Director {name: 'Oliver Stone'})

                            Notice how in the relationship 'ACTED_IN' the role of 'Bud fox' is present.
                            """

    ontology = Ontology(
        # labels of the entities to be extracted. Can be a string or an object, like the following.
        labels=[
            {"Person": "Person name without any adjectives, Remember a person may be references by their name or using a pronoun"},
            {"Object": "Do not add the definite article 'the' in the object name"},
            {"Event": "Event event involving multiple people. Do include qualifiers or verbs like gives, leaves, works etc."},
            "Place",
            "Document",
            "Organisation",
            "Action",
            {"Miscellanous": "Any important concept can not be categorised with any other given label"},
        ],
        # Relationships that are important for your application.
        # These are more like instructions for the LLM to nudge it to focus on specific relationships.
        # There is no guarentee that only these relationships will be extracted, but some models do a good job overall at sticking to these relations.
        relationships=[
            "Relation between any pair of Entities",
            ],
    )
    model = "gpt-4-turbo"
    llm = OpenAIClient(model=model, temperature=0.1, top_p=0.5)
    graph_maker = GraphMaker(ontology=ontology, llm_client=llm, verbose=False)
    net = Network(notebook=True, cdn_resources='in_line')

    @classmethod
    def create_graph(cls, documents):
        graph = cls.graph_maker.from_documents(
            list(documents),
            delay_s_between=10 ## delay_s_between because otherwise groq api maxes out pretty fast.
            )
        if graph:
            print("Graph creatd with total number of Edges:", len(graph))
            return graph
        else:
            return None
        
    # Function to check if a node exists
    @classmethod
    def node_exists(cls, net, node_id):
        return any(node['id'] == node_id for node in net.nodes)

    # Function to create a joined string
    @classmethod
    def create_joined_string(cls, input_string):
        modified_string = input_string.replace(' ', '_')
        modified_string = modified_string.replace("'", "")
        return modified_string

    @classmethod
    def build_graph_image(cls, graph):
        for chunk in graph:
            labels = [chunk.node_1.label, chunk.node_2.label]
            ids = [chunk.node_1.name, chunk.node_2.name]
            relationship = chunk.relationship
            print(f"{labels[0]}: {ids[0]} --> {labels[1]}:{ids[1]}; rel:{relationship}")

            for label,id in zip(labels,ids):
                if not cls.node_exists(cls.net, node_id=id):
                    cls.net.add_node(id, label=f"{label}:{id}")

            cls.net.add_edge(ids[0], ids[1], title=relationship, arrows='to')

    @classmethod
    def save_graph_image(cls, filename):
        cls.net.show_buttons(filter_=['physics'])
        if filename.endswith('.html'):
            cls.net.show(filename)
        else:
            print("Filename should have a .html extension")
        cls.net.show(filename)

    @classmethod
    def get_create_graph_query(cls, graph):
        relationship_list = []
        query = "CREATE\n"
        for chunk in graph:
            labels = [chunk.node_1.label, chunk.node_2.label]
            ids = [chunk.node_1.name, chunk.node_2.name]
            relationship = cls.create_joined_string(chunk.relationship.split('.')[0])
            relationship_list.append(relationship)
            query += f"""(:{labels[0]} {{name:"{ids[0]}"}})-[:{relationship}]->(:{labels[1]} {{name:"{ids[1]}"}}),\n"""
        query =  query[:-2]
        return query
    
    @classmethod
    def clean_create_graph_query(cls, query):
        response = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0.0,
        messages=[
            {"role": "system", "content": cls.GRAPH_CLEANER_PROMPT},
            {"role": "user", "content": f"{query}"},
        ]
        )
        return response.choices[0].message.content
    
    @classmethod
    def run(cls, docs, graph_name):
        # Connect to FalkorDB
        db = FalkorDB(host='localhost', port=6379)
        g = db.select_graph(graph_name)
        graph = cls.create_graph(documents=docs)
        #cls.build_graph_image(graph=graph)
        #cls.save_graph_image(filename="graph.html")
        query = cls.get_create_graph_query(graph=graph)
        print("FIRST CREATION QUERY:",query)
        modified_query = cls.clean_create_graph_query(query=query)
        print("MODIFIED QUERY:",modified_query)
        g.query(modified_query)
        return g
    
class RetrieveFromGraph:
    def __init__(self, graph_name) -> None:
        self.db = FalkorDB(host='localhost', port=6379)
        self.graph = self.db.select_graph(graph_name)
        self.QUESTION_GEN_PROMPT = """
        You’re a Cypher expert, with access to the following graph:
        The knowledge graph schema is as follows:
        The graph contains the following node labels:
        {node_text}
        the Module label has {node_length} number of nodes.
        The graph contains the following relationship types:
        {relationship_text}
        This is the end of the knowledge graph schema description.
        Ensure to generate the cypher query for the question passed to you
        """
        self.question_prompt_template = PromptTemplate.from_template(self.QUESTION_GEN_PROMPT)

    def create_query(self, question):
        response = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0.0,
        messages=[
            {"role": "system", "content": self.question_prompt_template.format(
                node_text=self.view_nodes(), 
                node_length=self.get_total_number_of_nodes(), 
                relationship_text=self.view_edges())},
            {"role": "user", "content": f"{question}"},
        ]
        )
        return response.choices[0].message.content

    def view_nodes(self):
        node_text= ""
        nodes = self.graph.query("""MATCH (n) RETURN n""")
        for node in nodes.result_set:
            node_text += str(node[0]) + "\n"
        return (node_text)
    
    def get_total_number_of_nodes(self):
        return len(self.graph.query("""MATCH (n) RETURN n""").result_set)
    
    def get_unique_edge_set(self):
        edge_set = set()
        edges = self.graph.query("""MATCH ()-[r]->() RETURN type(r) AS type, properties(r) AS properties""")
        for e in edges.result_set:
            temp_e = str(e[0])
            for key, value in e[1].items():
                temp_e += str(f" {{{key}: {value}}}")
            edge_set.add(temp_e)
        return edge_set
    
    def view_edges(self):
        relationship_text = ""
        edge_set = self.get_unique_edge_set()
        for e in edge_set:
            relationship_text += str(e)+ "\n"
        return (relationship_text)
    
    def get_cypher_text_from_output(self, text):
        pattern = r'```cypher\n(.*?)\n```'
        cypher_query = re.search(pattern, text, re.DOTALL)
        return cypher_query.group(1)
    
    def get_total_number_of_edges(self):
        edge_set = self.get_unique_edge_set()
        edge_len=len(edge_set)
        return edge_len
    
    def get_answer(self, question):
        query = self.create_query(question)
        print("QUERY:",query)
        cypher_query = self.get_cypher_text_from_output(query)
        answer=self.graph.query(cypher_query)
        return answer.result_set[0]

'''
docs = ProcessConversation.run(infile_file_path="../docs/bezos.txt")
graph = CreateGraph.run(docs=docs, graph_name="Convo-1")

graph_to_query = RetrieveFromGraph(graph_name="Convo-1")
nodes  = graph_to_query.view_nodes()
edges = graph_to_query.view_edges()
print("nodes are:", nodes)
print("edges are:", edges)
ans = graph_to_query.get_answer("What ages did jeff live on the farm?")
print(ans)
'''
