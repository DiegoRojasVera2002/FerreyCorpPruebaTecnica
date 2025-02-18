import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import pandas as pd
import numpy as np
from openai import OpenAI as DirectOpenAI

# Load environment variables
load_dotenv()

def preprocess_data(df):
    """
    Función para preprocesar el dataset y hacer más claros los valores categóricos
    """
    df_processed = df.copy()
    
    # Mapeo de variables categóricas
    mappings = {
        'genero': {
            0: 'Mujer',
            1: 'Hombre'
        },
        'estado_civil': {
            0: 'Soltero',
            1: 'Casado',
            2: 'Divorciado'
        },
        'nivel_educacion': {
            0: 'Básica',
            1: 'Media',
            2: 'Superior',
            3: 'Postgrado'
        },
        'ocupacion': {
            0: 'Desempleado',
            1: 'Empleado',
            2: 'Independiente',
            3: 'Jubilado'
        },
        'incidencia_compra': {
            0: 'No compró',
            1: 'Sí compró'
        }
    }
    
    # Aplicar mapeos
    for column, mapping in mappings.items():
        if column in df_processed.columns:
            df_processed[column] = df_processed[column].map(mapping)
    
    # Crear columnas descriptivas para promociones y precios
    for i in range(1, 6):
        # Mapear promociones a texto descriptivo
        promo_col = f'promo_marca_{i}'
        if promo_col in df_processed.columns:
            df_processed[promo_col] = df_processed[promo_col].map({
                0: 'Sin promoción',
                1: 'En promoción'
            })
        
        # Añadir descripción a precios
        price_col = f'precio_marca_{i}'
        if price_col in df_processed.columns:
            df_processed[f'{price_col}_descripcion'] = df_processed[price_col].apply(
                lambda x: f'${x:.2f}'
            )
    
    return df_processed

def create_dataset_description(df_processed):
    """
    Crear una descripción detallada del dataset para el contexto del RAG
    """
    description = []
    
    # Información general
    description.append("INFORMACIÓN GENERAL DEL DATASET:")
    description.append(f"- Total de registros: {len(df_processed):,}")
    description.append(f"- Período de análisis: Día {df_processed['dia_visita'].min()} al {df_processed['dia_visita'].max()}")
    
    # Estadísticas de clientes
    description.append("\nESTADÍSTICAS DE CLIENTES:")
    description.append(f"- Rango de edades: {df_processed['edad'].min()} a {df_processed['edad'].max()} años")
    description.append(f"- Edad promedio: {df_processed['edad'].mean():.1f} años")
    description.append(f"- Distribución por género: {df_processed['genero'].value_counts().to_dict()}")
    
    # Estadísticas de compras
    description.append("\nESTADÍSTICAS DE COMPRAS:")
    compras = df_processed[df_processed['incidencia_compra'] == 'Sí compró']
    description.append(f"- Total de compras realizadas: {len(compras)}")
    description.append(f"- Promedio de cantidad por compra: {compras['cantidad'].mean():.2f}")
    
    # Información de precios y promociones
    description.append("\nINFORMACIÓN DE PRECIOS Y PROMOCIONES:")
    for i in range(1, 6):
        price_col = f'precio_marca_{i}'
        promo_col = f'promo_marca_{i}'
        if price_col in df_processed.columns and promo_col in df_processed.columns:
            description.append(f"Marca {i}:")
            description.append(f"  - Rango de precios: ${df_processed[price_col].min():.2f} - ${df_processed[price_col].max():.2f}")
            description.append(f"  - Frecuencia de promociones: {(df_processed[promo_col] == 'En promoción').mean()*100:.1f}%")
    
    return "\n".join(description)

class RAGChatbot:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGChatbot, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        self.llm = OpenAI(
            temperature=0.7,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer',
            k=5
        )
        
        self.vectorstore = self.create_vectorstore()
        
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=CUSTOM_PROMPT_TEMPLATE
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )
        
        self.initialized = True

    def create_vectorstore(self):
        try:
            # Cargar y preprocesar datos
            df = pd.read_csv(r'D:\PruebatecnicaFerreycorp\GenAI\Dataset\compras_data.csv')
            df_processed = preprocess_data(df)
            
            # Generar descripción del dataset
            dataset_description = create_dataset_description(df_processed)
            
            # Cargar diccionario
            with open(r'D:\PruebatecnicaFerreycorp\GenAI\Dataset\Diccionario de Datos.txt', 'r', encoding='utf-8') as file:
                dictionary_text = file.read()

            # Combinar toda la información
            combined_text = f"""
            DATASET DE ANÁLISIS DE COMPRAS
            
            {dataset_description}
            
            DICCIONARIO DE DATOS:
            {dictionary_text}
            
            INTERPRETACIÓN DE VALORES:
            - Género: 0 = Mujer, 1 = Hombre
            - Estado Civil: 0 = Soltero, 1 = Casado, 2 = Divorciado
            - Nivel Educación: 0 = Básica, 1 = Media, 2 = Superior, 3 = Postgrado
            - Ocupación: 0 = Desempleado, 1 = Empleado, 2 = Independiente, 3 = Jubilado
            - Incidencia de compra: 0 = No compró, 1 = Sí compró
            - Promociones: 0 = Sin promoción, 1 = En promoción
            
            MUESTRA DE DATOS PROCESADOS:
            {df_processed.head().to_string()}
            """

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=100,
                length_function=len
            )
            texts = text_splitter.split_text(combined_text)

            return FAISS.from_texts(texts, self.embeddings)
        except Exception as e:
            print(f"Error al crear el vectorstore: {str(e)}")
            raise

    def process_question(self, question: str) -> str:
        try:
            if not question.strip():
                return "Por favor, hazme una pregunta sobre los datos de compras."

            result = self.chain.invoke({
                "question": question
            })
            
            return result.get("answer", "Lo siento, no pude generar una respuesta.")
            
        except Exception as e:
            return f"Lo siento, ocurrió un error al procesar tu pregunta: {str(e)}"

# Singleton instance
chatbot_instance = None

def get_chatbot_instance():
    global chatbot_instance
    if chatbot_instance is None:
        chatbot_instance = RAGChatbot()
    return chatbot_instance

# Template del prompt actualizado con el contexto del preprocesamiento
CUSTOM_PROMPT_TEMPLATE = """Eres un asistente especializado en analizar datos de compras y comportamiento de clientes.
Utiliza la siguiente información recuperada para responder la pregunta del usuario.
SI EL USUARIO TE PREGUNTA COMO ESTAS? COMO TE VA? QUE TAL? RESPONDE DE MANERA AMABLE Y SE UNA PERSONA MUY AMABLE Y CONVERSACIONAL 
SI DAS UNA RESPUESTA PONLE ESTADISTICAS CON PORCENTAJES, NO DES RESPUESTAS DUDOSAS

Los datos han sido preprocesados y contienen las siguientes interpretaciones:
- Género: 0 = Mujer, 1 = Hombre
- Estado Civil: 0 = Soltero, 1 = Casado, 2 = Divorciado
- Nivel Educación: 0 = Básica, 1 = Media, 2 = Superior, 3 = Postgrado
- Ocupación: 0 = Desempleado, 1 = Empleado, 2 = Independiente, 3 = Jubilado

Contexto recuperado: {context}

Historial de la conversación:
{chat_history}

Pregunta del usuario: {question}

Instrucciones adicionales:
- Responde en español
- Sé conciso pero informativo
- Utiliza las interpretaciones de los valores proporcionadas
- Si no tienes suficiente información para responder, indícalo
- Si la pregunta no está relacionada con los datos proporcionados, puedes responder de manera general pero apropiada
- Si te preguntan por estadísticas, proporciona números precisos cuando estén disponibles

Tu respuesta:"""