# 01 - IMPORTAÇÕES
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, DallETool
from langchain_openai import ChatOpenAI

# 02 - CARREGAMENTO DAS VARIÁVEIS DE AMBIENTE
load_dotenv()

# 03 - CONFIGURAÇÃO DO LLM
gpt_4o_mini = ChatOpenAI(model_name='gpt-4o-mini')    

# 04 - TOOLS
# Ferramenta de pesquisa na internet
search_tool = SerperDevTool()

# Ferramenta DALL-E com especificação de tamanho 16:9
dalle_tool = DallETool(size='1792x1024')  # Formato 16:9

# 05 - AGENTES
## AGENTE PESQUISADOR
pesquisador = Agent(
    role='Pesquisador de Tema',
    goal='Pesquisar informações detalhadas sobre {tema} na internet',
    backstory=(
        "Você é um especialista em pesquisa, com habilidades aguçadas para "
        "encontrar informações valiosas e detalhadas na internet."
    ),
    verbose=True,
    memory=True,
    tools=[search_tool],
    llm=gpt_4o_mini
)

tarefa_pesquisa = Task(
    description=(
        "Pesquisar informações detalhadas e relevantes sobre o tema: {tema}. "
        "Concentre-se em aspectos únicos e dados importantes que podem enriquecer o vídeo. "
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output='Um documento com as principais informações e dados sobre {tema}.',
    tools=[search_tool],
    agent=pesquisador
)

## AGENTE ESCRITOR DE TÍTULO
escritor_titulos = Agent(
    role='Escritor de Títulos de Vídeo',
    goal='Criar títulos atraentes e otimizados para vídeos sobre {tema}',
    verbose=True,
    memory=True,
    backstory=(
        "Você tem uma habilidade especial para criar títulos que capturam a "
        "essência de um vídeo e atraem a atenção do público."
    ),
    llm=gpt_4o_mini
)

tarefa_titulos = Task(
    description=(
        "Criar títulos atraentes e otimizados para o vídeo sobre o tema: {tema}. "
        "Certifique-se de que o título seja cativante e esteja otimizado para SEO. "
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output='Um título otimizado para o vídeo sobre {tema}.',
    agent=escritor_titulos,
)

## AGENTE ROTEIRISTA
escritor_roteiro = Agent(
    role='Escritor de Roteiro',
    goal='Escrever um roteiro detalhado e envolvente para um vídeo sobre {tema}',
    verbose=True,
    memory=True,
    backstory=
        f"""Você é um contador de histórias talentoso, capaz de transformar
        informações em narrativas cativantes para vídeos."""
    ,
    llm=gpt_4o_mini
)

tarefa_roteiro = Task(
    description=(
        "Escrever um roteiro detalhado para o vídeo sobre o tema: {tema}. "
        "O roteiro deve ser envolvente e fornecer um fluxo lógico de informações. "
        "Se necessário, especifique imagens que podem enriquecer o conteúdo. "
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output='Um roteiro completo e bem estruturado para o vídeo sobre {tema}.',
    agent=escritor_roteiro,
)

## AGENTE ESPECIALISTA EM SEO
especialista_seo = Agent(
    role='Especialista em SEO para YouTube',
    goal='Otimizar o roteiro e o título para que o vídeo tenha uma alta performance no YouTube',
    verbose=True,
    memory=True,
    backstory=(
        "Você é um expert em SEO, com um profundo entendimento das melhores "
        "práticas para otimizar conteúdo para o YouTube."
    ),
    llm=gpt_4o_mini
)

tarefa_seo = Task(
    description=(
        "Otimizar o título e o roteiro do vídeo para que tenha uma alta performance no YouTube. "
        "Incorporar as melhores práticas de SEO para garantir uma boa classificação e visibilidade. "
        "Crie hashtags, palavras-chaves e tags de vídeo para youtube. "
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output='Roteiro e título otimizados para o vídeo sobre {tema}, prontos para publicação no YouTube.',
    agent=especialista_seo,
)

## AGENTE ESCRITOR DE PROMPT PARA DALL-E
criador_prompt_dalle = Agent(
    role='Criador de Prompts para DALL-E',
    goal='Escrever um prompt para gerar uma imagem usando DALL-E com base no tema do vídeo {tema}',
    verbose=True,
    memory=True,
    backstory=(
        "Você é um especialista em criar descrições detalhadas e imaginativas que "
        "permitem ao DALL-E gerar imagens impressionantes com base em textos."
    ),
    llm=gpt_4o_mini
)

tarefa_criacao_prompt_dalle = Task(
    description=(
        "Criar um prompt detalhado para gerar uma imagem no DALL-E "
        "conforme descrito no roteiro."
    ),
    expected_output='Um prompt de texto detalhado para geração de imagem no DALL-E.',
    agent=criador_prompt_dalle,
)

## AGENTE GERADOR DE IMAGENS COM O DALL-E
gerador_imagens = Agent(
    role='Gerador de Imagens com DALL-E',
    goal='Gerar uma imagem usando DALL-E com o prompt fornecido pelo Criador de Prompts para DALL-E',
    verbose=True,
    memory=True,
    backstory=(
        "Você é um mestre em transformar descrições textuais em belas imagens, utilizando o poder do DALL-E."
    ),
    tools=[dalle_tool],
    llm=gpt_4o_mini
)

tarefa_geracao_imagem = Task(
    description=(
        "Gerar uma imagem usando o DALL-E com o prompt fornecido pelo Criador de Prompts para DALL-E."
    ),
    expected_output='Uma imagem gerada pronta para uso.',
    agent=gerador_imagens,
)

## AGENTE REVISOR
revisor = Agent(
    role='Revisor de Conteúdo',
    goal='Revisar todo o conteúdo produzido, incluir links das imagens geradas e entregar a versão final ao usuário',
    verbose=True,
    memory=True,
    backstory=(
        "Você tem um olho afiado para detalhes, garantindo que todo o conteúdo "
        "esteja perfeito antes de ser entregue ao usuário."
    ),
    llm=gpt_4o_mini
)

tarefa_revisao = Task(
    description=(
        "Revisar todo o conteúdo produzido (título, roteiro, e otimização de SEO), "
        "incluir os links das imagens geradas e preparar a versão final para entrega ao usuário."
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output='Conteúdo revisado, com links das imagens inclusos, pronto para entrega ao usuário.',
    agent=revisor,
    output_file='output_teste.md'  # Configurando o output para salvar em um arquivo Markdown
)

# 06 - CREW
# Formando a crew
crew = Crew(
    agents=[
        pesquisador,
        escritor_titulos,
        escritor_roteiro,
        especialista_seo,
        criador_prompt_dalle,
        gerador_imagens,
        revisor
    ],
    tasks=
    [
        tarefa_pesquisa,
        tarefa_titulos,
        tarefa_roteiro,
        tarefa_seo,
        tarefa_criacao_prompt_dalle,
        tarefa_geracao_imagem,
        tarefa_revisao
    ],
    process=Process.sequential  # Processamento sequencial das tarefas
)

# 07 - SOLICITANDO O TEMA PARA O USUÁRIO
tema_user = input("Digite o tema do vídeo: ")
print(tema_user)

# 08 - EXECUÇÃO DA CREW
# Executando o processo com o tema escolhido
result = crew.kickoff(inputs={'tema': tema_user})

# 09 - EXIBINDO O RESULTADO
#print(result)