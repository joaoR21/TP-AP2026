#import "cover.typ": cover
#import "template.typ": *

#show: project

#cover("", (
  (name: "Enzo Vieira", number: "pg61518"),
  (name: "Gustavo Barros", number: "pg61527"),
  (name: "João Macedo", number: "pg60274"),
  (name: "Jorge Pereira", number: "pg60276")), 
   datetime.today().display("[month repr:long] [day], [year]")
 )

#set page(numbering: "i", number-align: center)
#counter(page).update(1)

#show outline: it => {
    show heading: set text(size: 18pt)
    it
}

#{
  show outline.entry.where(level: 1): it => {
    v(5pt)
    strong(it)
  }

  outline(
    title: [Índice], 
    indent: true, 
  )
}

#v(-0.4em)
#outline(
  title: none,
  target: figure.where(kind: "attachment"),
  indent: n => 1em,
)


#set page(numbering: "1", number-align: center)
#counter(page).update(1)

= Introdução

Com o crescimento dos modelos de linguagem de grande escala, surge a necessidade de distinguir textos gerados por máquinas de textos escritos por humanos. Para além desta distinção básica, surge um desafio adicional: identificar a origem específica do texto, ou seja, qual a empresa responsável pelo modelo que o gerou.

Neste contexto, consideramos um conjunto de textos curtos em inglês, com 80 a 120 palavras, provenientes de cinco classes distintas: *Google* (modelos Gemma e Gemini), *Anthropic* (modelo Claude), *Meta* (modelo Llama), *OpenAI* (modelos GPT) e *Human* (textos escritos por pessoas). A identificação correta da origem destes textos representa um problema complexo, sobretudo devido às características específicas do domínio e à semelhança estilística entre os textos gerados por diferentes modelos.


= Construção dos Datasets

== Dataset v1

A primeira versão do _dataset_ foi construída a partir de três fontes públicas amplamente utilizadas em investigação. O *OpenTuringBench* (MLNTeam-Unical) forneceu textos gerados por vários modelos de linguagem de grande escala, sendo utilizado para representar as classes `Google` e `Meta`. O *HC3* (Hello-SimpleAI) disponibiliza pares de perguntas com respostas humanas e respostas geradas pelo ChatGPT, permitindo a construção das classes `Human` e `OpenAI`. O _dataset_ *Anthropic/persuasion* foi utilizado para obter textos gerados por modelos Claude, representando a classe `Anthropic`.

Após a recolha dos dados, foi aplicado um processo de pré-processamento que incluiu a filtragem por comprimento entre 80 e 120 palavras, a remoção de duplicados e o balanceamento das classes com um máximo de 2000 amostras por classe. O _dataset_ final ficou composto por 5000 amostras, distribuídas em treino (70%), validação (15%) e teste (15%).

Os resultados obtidos com este _dataset_ (Submissão 1) ficaram próximos dos 32% de _accuracy_, valor equivalente a uma classificação aleatória numa tarefa de cinco classes. Este desempenho evidenciou um problema de diferença de domínio: os _datasets_ públicos incluem principalmente conversas genéricas, respostas a perguntas abertas e textos persuasivos, enquanto os textos do docente pertencem ao domínio das ciências naturais e seguem um comprimento bastante específico.

== Dataset v2

Para reduzir este problema de diferença de domínio, foi construída uma segunda versão do _dataset_ com dados gerados especificamente no mesmo formato e tema dos dados de avaliação. Foram definidos 37 tópicos de ciências naturais, incluindo áreas como química, biologia, física, biomedicina, matemática e engenharia. Para cada tópico foram criadas três variações de _prompt_, resultando em 111 combinações distintas.

Para cada classe foram gerados textos com comprimento entre 80 e 120 palavras sobre os tópicos definidos, utilizando as seguintes ferramentas:

- *Groq API*: `llama-3.3-70b` para a classe Meta e o modelo `gpt-oss-120b` para a classe OpenAI
- *Ollama* (via execução local): `Gemma-3:1b` para a classe Google
- *interface _web_ do Claude*: `Haiku 4.5` para a classe Anthropic
- *Wikipedia API*: parágrafos de artigos científicos para a classe Human

Cada combinação de tópico e _prompt_ gerou 150 textos por classe, garantindo diversidade temática e consistência estrutural. Após a geração, foi aplicado um _pipeline_ de limpeza semelhante ao utilizado na primeira versão do _dataset_, incluindo filtragem por comprimento, remoção de duplicados e balanceamento das classes até 1000 amostras por classe.

O _dataset_ final ficou composto por 5000 amostras, distribuídas em 3500 para treino, 750 para validação e 750 para teste, com divisão estratificada por classe. Esta abordagem permitiu alinhar o domínio dos dados de treino com o domínio dos dados de avaliação, aumentando a probabilidade de o modelo aprender padrões relevantes para a tarefa de classificação.

== Pré-processamento

O pré-processamento aplicado é comum a todos os modelos e inclui a conversão do texto para minúsculas, a remoção de pontuação e a filtragem por comprimento através da função `truncate_text()`. Os modelos baseados em Transformers não requerem etapas adicionais, uma vez que o tokenizador do próprio modelo realiza automaticamente a conversão do texto em _tokens_ numéricos adequados para processamento.

Para os modelos clássicos de _machine learning_ foi necessário converter os textos numa representação numérica densa. Na Submissão 1A, o modelo de regressão logística utilizou um `TFIDFVectorizer` implementado de raiz em NumPy, com unigramas e um vocabulário limitado a 10 000 _tokens_. Esta abordagem permite transformar cada texto num vetor numérico que representa a importância relativa das palavras no _corpus_, facilitando a aprendizagem de padrões estatísticos.

Na Submissão 1B, a rede densa implementada em PyTorch utilizou o `CombinedVectorizer`, que concatena duas representações TF-IDF distintas. A primeira corresponde a TF-IDF de palavras com 5 000 dimensões. A segunda corresponde a TF-IDF de _character n-grams_ que variam entre trigramas e pentagramas, com 3 000 dimensões. A concatenação destas duas representações produz vetores finais com 8 000 dimensões.

Os *character n-grams* permitem capturar padrões estilísticos sublexicais que tendem a ser consistentes dentro do mesmo modelo gerador. Estes padrões incluem sufixos, estruturas morfológicas, fragmentos de palavras, formas de pontuação e pequenas regularidades na construção das frases que não são facilmente capturadas apenas por unigramas. Esta combinação de características léxicas e sublexicais contribui para melhorar a capacidade discriminativa dos modelos clássicos.

Os vectorizadores são guardados em ficheiros `.pkl` e posteriormente carregados nos _notebooks_ de submissão, evitando a necessidade de novo treino e garantindo reprodutibilidade dos resultados.

= Modelos

== Regressão Logística

O modelo de _baseline_ foi uma regressão logística multi-classe implementada de raiz em NumPy, sem recurso a bibliotecas de _machine learning_. A implementação utiliza a função _softmax_ na camada de saída para produzir distribuições de probabilidade sobre as cinco classes, com *cross-entropy* como função de custo e regularização L2 aplicada aos pesos. O treino é realizado através de *mini-batch gradient descent*, com pesos inicializados a zero e, opcionalmente, com standardização das _features_.

Os hiperparâmetros utilizados no treino final foram *learning rate* de 0.1, regularização L2 com lambda de 1e-4, 300 _epochs_ e *batch size* de 64. As _features_ foram geradas com o `TFIDFVectorizer` implementado de raiz, utilizando unigramas e um vocabulário de 10 000 _tokens_.

Este modelo foi submetido como Submissão 1A e obteve cerca de 32% de *accuracy* no _dataset_ do docente, reflexo direto da diferença de domínio discutida na Secção 2.1.

== DNN em PyTorch

A rede densa em PyTorch, submetida como Subm1-B, utilizou o `CombinedVectorizer` como representação de entrada, produzindo vetores com 8 000 dimensões compostos por 5 000 _features_ de palavras e 3 000 _character n-grams_. A arquitetura consistia em duas camadas densas com 128 e 64 neurónios, ativação ReLU e *Dropout* de 0.5. O treino foi realizado com o optimizador Adam, *learning rate* de 0.001 e *early stopping* com _patience_ de 15 _epochs_.

Treinado com o dataset v1, o modelo obteve igualmente cerca de 32% de _accuracy_ no dataset do docente. Quando re-treinado com o dataset v2, mantendo a mesma arquitetura e configuração, a _accuracy_ interna subiu para cerca de 58%, confirmando que o _bottleneck_ era a qualidade e o domínio dos dados e não a capacidade da rede.

== DistilRoBERTa

Para a Submissão 2B foi realizado _fine-tuning_ do modelo `distilroberta-base`. O conjunto de treino combinou 5 000 amostras sintéticas do _dataset_ v2 com 100 amostras reais provenientes das _labels_ reveladas da submissão anterior, sobre-amostradas com peso 10 para aumentar a sua influência no processo de aprendizagem. O conjunto de validação utilizado durante o treino correspondeu aos 125 exemplos do ficheiro dataset-exemplos.csv disponibilizado antecipadamente pelo docente.

Os hiperparâmetros utilizados incluíram *learning rate* de 3e-5, *batch size* de 32, 15 _epochs_ com *early stopping* com _patience_ de 3, *label smoothing* de 0.1 e *warmup* de 10% dos passos totais. A função de perda incorporou *label smoothing* e *class weights* calculados a partir dos dados reais, permitindo reduzir o impacto do desbalanceamento entre dados sintéticos e dados reais.

O modelo atingiu cerca de 67% de _accuracy_ interna e aproximadamente 58% no _dataset_ de validação do docente, mostrando uma melhoria significativa face aos modelos clássicos e à rede densa.

== RoBERTa-base com Optuna

Para a Submissão 2A foi realizado _fine-tuning_ do modelo `roberta-base` com optimização automática de hiperparâmetros através de *Optuna*, utilizando 10 _trials_ e `MedianPruner` para interrupção antecipada de experiências pouco promissoras.

#table(
  columns: (auto, auto, auto),
  [*hiperparâmetro*], [*espaço de busca*], [*melhor valor*],
  [`learning rate`], [[1e-5,5e-5] log], [1.10e-5],
  [`batch size`], [{16,32}], [32],
  [`label smoothing`], [[0.05,0.2]], [0.062],
  [`warmup fraction`], [[0.05,0.2]], [16.2%],
  [`weight decay`], [[1e-3,0.1] log], [0.0151],
  [`real data weight`], [[5,20]], [9],
)

O parâmetro *real data weight* controla o número de repetições das amostras reais no dataset de treino, permitindo compensar a diferença de distribuição entre dados sintéticos e dados reais. O treino final utilizou 20 _epochs_ com *early stopping* com _patience_ de 5, *mixed precision* FP16 e *gradient clipping* de 1.0.

O modelo obteve cerca de 76% de _accuracy_ interna e 67% no _dataset_ do docente, alcançando o quarto lugar no _ranking_ da Submissão 2.

== DeBERTa-v3-base

Após a Submissão 2 foi treinado o modelo `microsoft/deberta-v3-base`, que utiliza *disentangled attention* e posições relativas, permitindo uma melhor modelação das relações entre _tokens_ e padrões estilísticos presentes nos textos. A configuração de treino seguiu a mesma abordagem do RoBERTa, incluindo optimização de hiperparâmetros com Optuna.

Os melhores hiperparâmetros encontrados foram *learning rate* de 1.15e-5, *batch size* de 16, *label smoothing* de 0.060, *warmup* de 9.7%, *weight decay* de 0.094 e *real data weight* de 14. O modelo atingiu cerca de 69.6% de _accuracy_ interna e 60% no _dataset_ do docente, apresentando desempenho competitivo mas ligeiramente inferior ao RoBERTa no domínio específico da tarefa.

== Ensemble

Para a Submissão 3 foi construído um *ensemble* ponderado entre os dois melhores modelos Transformer. Para cada amostra, cada modelo produz uma distribuição de probabilidades sobre as cinco classes. A previsão final é calculada como uma média ponderada dessas distribuições, controlada por um parâmetro `alpha` que representa o peso atribuído ao RoBERTa.

$ p_("final") = alpha dot p_("RoBERTa") + (1 - alpha) dot p_("DeBERTa") $

O valor de `alpha` foi optimizado através de varrimento no conjunto de validação com 125 amostras reais, utilizando passos de 0.05 no intervalo entre 0.0 e 1.0. O melhor valor encontrado foi 0.90, indicando que o ensemble atribui 90% do peso às previsões do RoBERTa e 10% às do DeBERTa. Apesar do peso reduzido, a contribuição do DeBERTa revelou-se positiva: o ensemble superou o RoBERTa isolado na avaliação externa (74% vs. 67%), confirmando a complementaridade dos dois modelos.


= Resultados

A tabela seguinte resume os resultados obtidos por todos os modelos ao longo das três submissões. A _accuracy_ interna foi avaliada nos 125 exemplos reais do ficheiro dataset-exemplos.csv. A _accuracy_ externa corresponde ao _dataset_ de validação completo utilizado pelo docente para gerar o _ranking_ final.

#table(
  columns: (auto, auto, auto, auto),
  [*modelo*], [*acc. interna*], [*acc. externa*], [*submissão*],
  [Regressão Logística (NumPy)], [---], [~32%], [Subm1-A],
  [DNN em PyTorch], [---], [~32%], [Subm1-B],
  [DNN re-treinada com dataset v2], [58%], [---], [---],
  [DistilRoBERTa], [67%], [58%], [Subm2-B],
  [RoBERTa-base + Optuna], [76%], [67%], [Subm2-A],
  [DeBERTa-v3-base + Optuna], [70%], [60%], [Subm3-B],
  [Ensemble RoBERTa + DeBERTa (α = 0.90)], [77%], [74%], [Subm3-A],
)

A Submissão 1 apresentou cerca de 32% de _accuracy_ externa em ambos os modelos, valor equivalente a uma classificação aleatória, reflexo direto da diferença de domínio entre o dataset v1 e os dados de avaliação. A re-validação da DNN com o dataset v2 confirmou esta hipótese: a _accuracy_ interna subiu para 58% sem qualquer alteração à arquitetura.

Com a introdução de modelos Transformer e do dataset v2, os resultados melhoraram substancialmente. O DistilRoBERTa atingiu 67% de _accuracy_ interna, mas apenas 58% de _accuracy_ externa, revelando algum _overfitting_ ao dataset sintético. O RoBERTa-base, com hiperparâmetros optimizados via Optuna e sobre-amostragem dos dados reais, reduziu essa diferença e alcançou 76% de _accuracy_ interna e 67% de _accuracy_ externa.

O ensemble ponderado entre RoBERTa e DeBERTa foi o modelo final e atingiu 74% de _accuracy_ externa. A melhoria face ao RoBERTa isolado (67% → 74%) confirma a complementaridade dos dois modelos, mesmo com um peso reduzido atribuído ao DeBERTa.

= Desafios e trabalho futuro

O principal desafio do projeto foi a diferença de domínio entre os dados públicos utilizados no _dataset_ v1 e os dados de avaliação do docente. Os _datasets_ públicos incluem maioritariamente conversas genéricas e respostas a perguntas abertas, enquanto os textos de avaliação pertencem ao domínio das ciências naturais com um comprimento específico. Este desalinhamento de domínio resultou numa _accuracy_ externa de apenas 32%, equivalente a classificação aleatória, e motivou a construção do _dataset_ v2.

O segundo desafio foi o facto de as DNNs apresentarem constantemente _overfitting_, problema que acabou por não ser totalmente resolvido com as alterações aplicadas, nem mesmo trocando a técnica utilizada.

O modelo DistilRoBERTa teve um bom desempenho inicial, mas, para obter ainda melhores resultados, decidiu-se utilizar o RoBERTa-base, que é o modelo completo e maior que o DistilRoBERTa. Esta alteração revelou-se um sucesso, permitindo obter resultados relativamente bons quando comparados com todos os outros modelos.

Para encontrar um complemento ao RoBERTa-base, foram testados _ensembles_ com outras arquiteturas, incluindo combinações com as DNNs, mas os resultados ficaram abaixo do RoBERTa isolado.

Posteriormente, tentou-se utilizar o DeBERTa, que apresenta uma arquitetura diferente do RoBERTa-base. A combinação destes dois modelos obteve um bom resultado, contudo, o _ensemble_ atribui cerca de 90% de importância ao RoBERTa-base e apenas 10% ao DeBERTa, sugerindo que a sua contribuição é complementar e não substitutiva.

Uma estratégia futura que poderia ser implementada seria a classificação em duas etapas. Primeiro, os textos seriam classificados como gerados por IA ou escritos por humanos. Na segunda etapa, apenas os textos identificados como IA seriam classificados para determinar a que modelo pertencem. Esta técnica permitiria realizar primeiro uma classificação binária entre IA e humano e, posteriormente, uma classificação mais aprofundada do modelo responsável pela geração do texto. 

= Conclusões

A principal conclusão deste trabalho é que a qualidade e o domínio dos dados de treino têm um impacto significativamente maior no desempenho do que a escolha da arquitetura do modelo. A construção do _dataset_ v2, alinhado com o domínio das ciências naturais, permitiu aumentar a _accuracy_ de valores próximos de 32% para a faixa dos 60% a 70%, evidenciando a importância do _data-centric approach_.

A utilização de modelos Transformer pré-treinados combinada com sobre-amostragem de dados reais revelou-se a estratégia mais eficaz. O `roberta-base` com optimização de hiperparâmetros via Optuna foi o modelo individual com melhor desempenho, e o _ensemble_ com `microsoft/deberta-v3-base` permitiu uma melhoria adicional na avaliação externa, atingindo 74% de _accuracy_ e o terceiro lugar no ranking final.
