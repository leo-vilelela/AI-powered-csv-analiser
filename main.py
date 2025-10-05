import os
import streamlit as interface
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
import io
import base64
import json
from datetime import datetime
from dotenv import load_dotenv

# Configuração inicial
load_dotenv()

class SistemaMemoria:
    def __init__(self):
        self.historico_analises = []
        self.insights_coletados = []
        self.padroes_detectados = []
        self.conclusoes_gerais = []
        self.contador_interacoes = 0
    
    def registrar_analise(self, pergunta, resposta, metrica=None):
        registro = {
            "timestamp": datetime.now().isoformat(),
            "pergunta": pergunta,
            "resposta": resposta,
            "metrica": metrica,
            "tipo_analise": self._classificar_analise(pergunta)
        }
        self.historico_analises.append(registro)
        self.contador_interacoes += 1
        
        # Extrai insights automaticamente da resposta
        self._extrair_insights(resposta, pergunta)
    
    def _classificar_analise(self, pergunta):
        pergunta_lower = pergunta.lower()
        if any(palavra in pergunta_lower for palavra in ['distribuição', 'histograma', 'frequência']):
            return "distribuicao"
        elif any(palavra in pergunta_lower for palavra in ['correlação', 'relação', 'associação']):
            return "correlacao"
        elif any(palavra in pergunta_lower for palavra in ['tendência', 'evolução', 'tempo']):
            return "tendencia"
        elif any(palavra in pergunta_lower for palavra in ['comparação', 'diferença', 'categoria']):
            return "comparacao"
        else:
            return "geral"
    
    def _extrair_insights(self, resposta, pergunta):
        # Analisa a resposta para extrair insights importantes
        insights_potenciais = [
            "alta", "baixa", "aumento", "diminuição", "correlação", "significante",
            "outlier", "padrão", "tendência", "maior", "menor", "máximo", "mínimo"
        ]
        
        resposta_lower = resposta.lower()
        for insight in insights_potenciais:
            if insight in resposta_lower:
                insight_registro = {
                    "timestamp": datetime.now().isoformat(),
                    "insight": insight,
                    "contexto": pergunta[:100],  # Primeiros 100 caracteres
                    "resumo": self._resumir_insight(resposta, insight)
                }
                if insight_registro not in self.insights_coletados:
                    self.insights_coletados.append(insight_registro)
    
    def _resumir_insight(self, resposta, palavra_chave):
        # Cria um resumo do insight encontrado
        sentences = resposta.split('.')
        for sentence in sentences:
            if palavra_chave in sentence.lower():
                return sentence.strip()
        return resposta[:150] + "..."
    
    def adicionar_padrao(self, tipo_padrao, descricao, dados_suporte):
        padrao = {
            "tipo": tipo_padrao,
            "descricao": descricao,
            "dados_suporte": dados_suporte,
            "timestamp": datetime.now().isoformat()
        }
        self.padroes_detectados.append(padrao)
    
    def adicionar_conclusao(self, conclusao, nivel_confianca="medio"):
        conclusao_registro = {
            "conclusao": conclusao,
            "nivel_confianca": nivel_confianca,
            "timestamp": datetime.now().isoformat(),
            "base_analise": len(self.historico_analises)
        }
        self.conclusoes_gerais.append(conclusao_registro)
    
    def obter_resumo_conclusoes(self):
        if not self.conclusoes_gerais and not self.insights_coletados:
            return "Ainda não foram identificadas conclusões significativas. Faça mais análises para gerar insights."
        
        resumo = "## 📊 Conclusões e Insights Obtidos\n\n"
        
        if self.conclusoes_gerais:
            resumo += "### Principais Conclusões:\n"
            for i, conclusao in enumerate(self.conclusoes_gerais[-5:], 1):  # Últimas 5 conclusões
                resumo += f"{i}. {conclusao['conclusao']} (Confiança: {conclusao['nivel_confianca']})\n"
        
        if self.insights_coletados:
            resumo += "\n### Insights Detectados:\n"
            insights_unicos = {}
            for insight in self.insights_coletados:
                chave = insight['insight']
                if chave not in insights_unicos:
                    insights_unicos[chave] = insight['resumo']
            
            for insight, descricao in list(insights_unicos.items())[:5]:
                resumo += f"• **{insight.title()}**: {descricao}\n"
        
        resumo += f"\n*Baseado em {self.contador_interacoes} análises realizadas*"
        return resumo
    
    def obter_historico_recente(self, limite=5):
        return self.historico_analises[-limite:]
    
    def limpar_memoria(self):
        self.historico_analises.clear()
        self.insights_coletados.clear()
        self.padroes_detectados.clear()
        self.conclusoes_gerais.clear()
        self.contador_interacoes = 0

class AnalisadorDadosInteligente:
    def __init__(self, chave_api):
        self.cliente = Groq(api_key=chave_api)
        self.conjunto_dados = None
        self.memoria = SistemaMemoria()
    
    def carregar_informacoes(self, dataframe):
        self.conjunto_dados = dataframe
        self.memoria.limpar_memoria()  # Limpa memória ao carregar novo dataset
        
        # Análise inicial automática do dataset
        analise_inicial = self._realizar_analise_inicial()
        self.memoria.adicionar_conclusao(analise_inicial, "alto")
    
    def _realizar_analise_inicial(self):
        """Realiza uma análise inicial automática do dataset"""
        if self.conjunto_dados is None:
            return "Dataset não carregado"
        
        try:
            colunas_numericas = self.conjunto_dados.select_dtypes(include=['number']).columns
            colunas_categoricas = self.conjunto_dados.select_dtypes(include=['object']).columns
            
            conclusoes = []
            
            if len(colunas_numericas) > 0:
                conclusoes.append(f"Dataset contém {len(colunas_numericas)} variáveis numéricas")
                # Detecta possíveis outliers
                for coluna in colunas_numericas[:3]:  # Analisa as 3 primeiras
                    q1 = self.conjunto_dados[coluna].quantile(0.25)
                    q3 = self.conjunto_dados[coluna].quantile(0.75)
                    iqr = q3 - q1
                    outliers = self.conjunto_dados[
                        (self.conjunto_dados[coluna] < (q1 - 1.5 * iqr)) | 
                        (self.conjunto_dados[coluna] > (q3 + 1.5 * iqr))
                    ]
                    if len(outliers) > 0:
                        conclusoes.append(f"Possíveis outliers detectados em {coluna}")
            
            if len(colunas_categoricas) > 0:
                conclusoes.append(f"Dataset contém {len(colunas_categoricas)} variáveis categóricas")
            
            if len(self.conjunto_dados) > 1000:
                conclusoes.append("Dataset de grande porte - amostra significativa")
            elif len(self.conjunto_dados) < 100:
                conclusoes.append("Dataset pequeno - cuidado com generalizações")
            
            return "; ".join(conclusoes)
            
        except Exception as e:
            return f"Análise inicial básica: {len(self.conjunto_dados)} registros carregados"

    def obter_resposta(self, pergunta):
        if self.conjunto_dados is None:
            return "Por favor, carregue um arquivo CSV primeiro.", None
        
        # Prepara contexto com histórico de análises
        contexto_historico = self._preparar_contexto_historico()
        
        # Prepara contexto sobre os dados
        contexto_dados = f"""
        Informações do dataset:
        - Formato: {self.conjunto_dados.shape}
        - Colunas: {list(self.conjunto_dados.columns)}
        - Primeiras linhas:
        {self.conjunto_dados.head().to_string()}
        - Estatísticas descritivas:
        {self.conjunto_dados.describe().to_string()}
        """
        
        prompt_completo = f"""
        {contexto_historico}
        
        {contexto_dados}
        
        Pergunta atual: {pergunta}
        
        Com base no histórico de análises e nos dados atuais, forneça uma resposta completa.
        Considere padrões já identificados e conclusões anteriores.
        """
        
        try:
            # Consulta ao Groq
            resposta = self.cliente.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """Você é um especialista em análise de dados com memória contextual. 
                        Use o histórico de análises para enriquecer suas respostas.
                        Identifique padrões, tendências e relações nos dados.
                        Ao final de cada análise, sugira próximos passos ou perguntas relacionadas."""
                    },
                    {
                        "role": "user", 
                        "content": prompt_completo
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=1024
            )
            
            texto_resposta = resposta.choices[0].message.content
            
            # Registra a análise na memória
            self.memoria.registrar_analise(pergunta, texto_resposta)
            
            # Gera visualização
            visualizacao = self._criar_visualizacao(pergunta, texto_resposta)
            
            # Atualiza conclusões baseadas na nova análise
            self._atualizar_conclusoes(pergunta, texto_resposta)
            
            return texto_resposta, visualizacao
            
        except Exception as erro:
            return f"Erro ao processar pergunta: {str(erro)}", None
    
    def _preparar_contexto_historico(self):
        """Prepara o contexto do histórico de análises para o prompt"""
        historico_recente = self.memoria.obter_historico_recente(3)  # Últimas 3 análises
        
        if not historico_recente:
            return "Não há histórico anterior de análises."
        
        contexto = "### Histórico Recente de Análises:\n"
        for i, analise in enumerate(historico_recente, 1):
            contexto += f"{i}. Pergunta: {analise['pergunta']}\n"
            contexto += f"   Tipo: {analise['tipo_analise']}\n"
            contexto += f"   Resumo: {analise['resposta'][:200]}...\n\n"
        
        return contexto
    
    def _atualizar_conclusoes(self, pergunta, resposta):
        """Atualiza conclusões baseadas na nova análise"""
        # Detecta conclusões importantes na resposta
        marcadores_conclusao = [
            "concluímos que", "portanto", "dessa forma", "em resumo", 
            "isso indica que", "sugere que", "evidencia", "demonstra"
        ]
        
        resposta_lower = resposta.lower()
        for marcador in marcadores_conclusao:
            if marcador in resposta_lower:
                # Extrai a sentença contendo a conclusão
                sentences = resposta.split('.')
                for sentence in sentences:
                    if marcador in sentence.lower():
                        self.memoria.adicionar_conclusao(sentence.strip(), "medio")
                        break
    
    def _criar_visualizacao(self, pergunta, resposta):
        try:
            figura, eixo = plt.subplots(figsize=(10, 6))
            
            pergunta_lower = pergunta.lower()
            colunas_numericas = self.conjunto_dados.select_dtypes(include=['number']).columns
            
            if len(colunas_numericas) > 0:
                coluna_alvo = colunas_numericas[0]
                
                if any(palavra in pergunta_lower for palavra in ['distribuição', 'histograma', 'frequência']):
                    self.conjunto_dados[coluna_alvo].hist(ax=eixo, bins=15, alpha=0.7, color='skyblue')
                    eixo.set_title(f'Distribuição de {coluna_alvo}')
                    eixo.set_ylabel('Frequência')
                    eixo.grid(True, alpha=0.3)
                
                elif any(palavra in pergunta_lower for palavra in ['correlação', 'relação', 'associação']):
                    if len(colunas_numericas) >= 2:
                        sns.scatterplot(data=self.conjunto_dados, x=colunas_numericas[0], y=colunas_numericas[1], ax=eixo)
                        eixo.set_title(f'Relação entre {colunas_numericas[0]} e {colunas_numericas[1]}')
                        eixo.grid(True, alpha=0.3)
                    else:
                        # Fallback para histograma
                        self.conjunto_dados[coluna_alvo].hist(ax=eixo, bins=15, alpha=0.7)
                        eixo.set_title(f'Distribuição de {coluna_alvo}')
                
                elif any(palavra in pergunta_lower for palavra in ['boxplot', 'outlier', 'dispersão']):
                    sns.boxplot(data=self.conjunto_dados[coluna_alvo], ax=eixo)
                    eixo.set_title(f'Boxplot de {coluna_alvo}')
                    eixo.grid(True, alpha=0.3)
                
                else:
                    # Gráfico de barras para colunas categóricas ou linha temporal
                    colunas_categoricas = self.conjunto_dados.select_dtypes(include=['object']).columns
                    if len(colunas_categoricas) > 0:
                        coluna_cat = colunas_categoricas[0]
                        contagem = self.conjunto_dados[coluna_cat].value_counts().head(10)
                        contagem.plot(kind='bar', ax=eixo, color='lightgreen')
                        eixo.set_title(f'Top 10 Valores em {coluna_cat}')
                        eixo.tick_params(axis='x', rotation=45)
                        eixo.grid(True, alpha=0.3)
                    else:
                        # Gráfico de linha para tendências temporais
                        if len(self.conjunto_dados) > 1:
                            self.conjunto_dados[coluna_alvo].plot(ax=eixo, marker='o')
                            eixo.set_title(f'Tendência de {coluna_alvo}')
                            eixo.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Converter para base64
            buffer = io.BytesIO()
            figura.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            imagem_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(figura)
            
            return f"data:image/png;base64,{imagem_base64}"
            
        except Exception as erro:
            print(f"Erro ao criar visualização: {erro}")
            return None
    
    def obter_conclusoes(self):
        return self.memoria.obter_resumo_conclusoes()
    
    def limpar_memoria(self):
        self.memoria.limpar_memoria()

class GerenciadorConversa:
    def __init__(self):
        self.registros = []
    
    def adicionar_mensagem(self, emissor, conteudo, imagem=None):
        self.registros.append({
            "emissor": emissor,
            "conteudo": conteudo,
            "imagem": imagem,
            "timestamp": datetime.now().isoformat()
        })

# Configuração da interface
interface.set_page_config(
    page_title="Analisador Inteligente com Memória", 
    layout="wide",
    page_icon="🧠"
)

interface.title("🧠 Analisador Inteligente com Memória Contextual")

# Inicialização na sessão
if "analisador_inteligente" not in interface.session_state:
    chave_groq = os.getenv("GROQ_API_KEY")
    print(chave_groq)
    if not chave_groq:
        interface.error("GROQ_API_KEY não encontrada nas variáveis de ambiente")
    interface.session_state.analisador_inteligente = AnalisadorDadosInteligente(chave_groq)

if "gerenciador_dialogo" not in interface.session_state:
    interface.session_state.gerenciador_dialogo = GerenciadorConversa()

if "dados_carregados" not in interface.session_state:
    interface.session_state.dados_carregados = None

# Mensagem inicial
if len(interface.session_state.gerenciador_dialogo.registros) == 0:
    interface.session_state.gerenciador_dialogo.adicionar_mensagem(
        "assistente", 
        "Olá! Sou um analisador inteligente com memória. Envie um CSV e faça perguntas - vou lembrar das análises anteriores para dar respostas contextuais! 🧠"
    )

# Layout principal
col_principal, col_lateral = interface.columns([3, 1])

with col_lateral:
    interface.header("🔧 Controles")
    
    # Botão para ver conclusões
    if interface.button("📊 Ver Conclusões", use_container_width=True):
        if interface.session_state.analisador_inteligente.conjunto_dados is not None:
            conclusoes = interface.session_state.analisador_inteligente.obter_conclusoes()
            interface.session_state.gerenciador_dialogo.adicionar_mensagem("assistente", conclusoes)
    
    # Botão para limpar memória
    if interface.button("🗑️ Limpar Memória", use_container_width=True):
        interface.session_state.analisador_inteligente.limpar_memoria()
        interface.session_state.gerenciador_dialogo.adicionar_mensagem(
            "assistente", 
            "Memória limpa! Começando uma nova sessão de análise."
        )
        interface.rerun()
    
    interface.markdown("---")
    
    # Área de upload
    interface.header("📁 Carregar Dados")
    arquivo_submetido = interface.file_uploader(
        "Selecionar arquivo CSV",
        type=["csv"],
        help="Carregue um dataset para análise",
        key="uploader_csv"
    )
    
    if arquivo_submetido and interface.session_state.dados_carregados is None:
        with interface.spinner("Analisando dataset..."):
            dados = pd.read_csv(arquivo_submetido)
            interface.session_state.analisador_inteligente.carregar_informacoes(dados)
            interface.session_state.dados_carregados = dados
            
            interface.success(f"✅ Dataset carregado: {dados.shape[0]} linhas × {dados.shape[1]} colunas")
            
            # Mensagem de confirmação com análise inicial
            resumo_inicial = f"""
            Dataset '{arquivo_submetido.name}' carregado com sucesso! 

            **Resumo inicial:**
            - Dimensões: {dados.shape[0]} linhas × {dados.shape[1]} colunas
            - Memória contextual ativada
            - Análise inicial concluída

            Faça perguntas sobre os dados e eu vou me lembrar das análises anteriores!
            """
            
            interface.session_state.gerenciador_dialogo.adicionar_mensagem(
                "assistente",
                resumo_inicial
            )

    # Estatísticas da sessão
    if interface.session_state.dados_carregados is not None:
        interface.markdown("---")
        interface.header("📈 Estatísticas")
        dados = interface.session_state.dados_carregados
        analisador = interface.session_state.analisador_inteligente
        
        col1, col2 = interface.columns(2)
        col1.metric("Linhas", dados.shape[0])
        col2.metric("Colunas", dados.shape[1])
        
        interface.metric("Análises Realizadas", analisador.memoria.contador_interacoes)
        interface.metric("Insights Coletados", len(analisador.memoria.insights_coletados))

with col_principal:
    interface.header("💬 Análise Contextual com Memória")
    
    # Exibir histórico de conversa
    for mensagem in interface.session_state.gerenciador_dialogo.registros:
        with interface.chat_message(mensagem["emissor"]):
            interface.markdown(mensagem["conteudo"])
            if mensagem["imagem"]:
                interface.image(mensagem["imagem"], use_container_width=True)

    # Entrada do usuário
    entrada_usuario = interface.chat_input(
        "Digite sua pergunta sobre os dados...",
        disabled=interface.session_state.dados_carregados is None
    )

    if entrada_usuario:
        # Adiciona pergunta do usuário
        interface.session_state.gerenciador_dialogo.adicionar_mensagem("usuario", entrada_usuario)
        
        with interface.chat_message("usuario"):
            interface.markdown(entrada_usuario)
        
        # Processa a resposta
        with interface.chat_message("assistente"):
            with interface.spinner("Analisando com contexto anterior..."):
                texto_resposta, grafico = interface.session_state.analisador_inteligente.obter_resposta(entrada_usuario)
            
            interface.markdown(texto_resposta)
            if grafico:
                interface.image(grafico, use_container_width=True)
        
        # Armazena resposta no histórico
        interface.session_state.gerenciador_dialogo.adicionar_mensagem("assistente", texto_resposta, grafico)

# Rodapé com informações da memória
if interface.session_state.dados_carregados is not None:
    interface.markdown("---")
    col1, col2, col3 = interface.columns(3)
    
    analisador = interface.session_state.analisador_inteligente
    col1.metric("Interações", analisador.memoria.contador_interacoes)
    col2.metric("Conclusões", len(analisador.memoria.conclusoes_gerais))
    col3.metric("Insights", len(analisador.memoria.insights_coletados))

# Exemplos de perguntas
with col_lateral:
    interface.markdown("---")
    interface.header("💡 Perguntas Sugeridas")
    interface.markdown("""
    - Qual a distribuição dos dados?
    - Existe correlação entre variáveis?
    - Quais são os outliers?
    - Mostre tendências temporais
    - Compare categorias
    - **Quais suas conclusões?** ← Novo!
    """)