# Semantic Macro-Themes in Sertanejo Universitário Lyrics: Topic Modeling with BERTopic

Este repositório reúne os materiais desenvolvidos para o artigo **“Semantic Macro-Themes in Sertanejo Universitário Lyrics: Topic Modeling with BERTopic”**, elaborado no contexto da disciplina de **Processamento de Linguagem Natural** do Programa de Pós-Graduação em Computação da UFRGS.

O trabalho investiga a organização temática interna de letras de **sertanejo universitário** por meio de modelagem de tópicos não supervisionada com **BERTopic**. O corpus analisado contém **5.037 estrofes válidas**, extraídas de **959 músicas** lançadas entre **2005 e 2025**.

## Objetivo

O objetivo do projeto é analisar como estrofes de músicas de sertanejo universitário se organizam semanticamente em tópicos e macro-tópicos interpretáveis.

A contribuição principal não está no uso isolado do BERTopic ou na comparação entre modelos de embeddings, mas na identificação de uma estrutura temática interna nas letras analisadas. Os resultados indicam a predominância de um núcleo afetivo-relacional, associado a paixão, sofrimento amoroso, término e reconciliação, articulado a dimensões como espiritualidade, comunicação, bebida, bar e sociabilidade.

## Estrutura do repositório

```text
bertopic-sertanejo-universitario/
│
├── README.md
│
└── pln-artigo/
    │
    ├── Códigos/
    │   ├── letras_sertanejo_web_scrap.py
    │   ├── rodar_bertopic_5_modelos_unificado.py
    │   ├── avaliar_metricas_bertopic_multimodelo.py
    │   └── rodar_corte_conceitual.py
    │
    ├── Dados Sertanejo Universitário/
    │   └── arquivos de entrada e dados processados utilizados no estudo
    │
    ├── Outputs Bertopic/
    │   └── saídas geradas pelos modelos BERTopic
    │
    ├── Outputs Cortes Conceituais/
    │   └── resultados da consolidação hierárquica dos tópicos
    │
    └── PLN_Enunciado do Trabalho 2026_PPGC.pdf
```

## Pipeline metodológico

O pipeline do projeto foi organizado nas seguintes etapas:

1. **Coleta das letras** por web scraping.
2. **Filtragem e preparação do corpus**, com remoção de músicas em outros idiomas, duplicidades e estrofes muito curtas.
3. **Segmentação das letras em estrofes**, adotadas como unidade principal de análise.
4. **Execução do BERTopic** com diferentes modelos de embeddings.
5. **Comparação quantitativa dos modelos**, considerando métricas como coerência, NPMI, diversidade e RBO invertido.
6. **Consolidação hierárquica dos tópicos**, agrupando tópicos semanticamente próximos em macro-tópicos.
7. **Validação humana dos rótulos**, realizada por quatro anotadores humanos nos dez macro-tópicos mais frequentes.

## Scripts principais

### `letras_sertanejo_web_scrap.py`

Script utilizado para coleta inicial das letras e metadados das músicas.

### `rodar_bertopic_5_modelos_unificado.py`

Executa o BERTopic com cinco modelos de embeddings aplicados ao mesmo corpus de estrofes.

Modelos comparados:

* `MPNet-multilingual`
* `BERT-ASSIN2`
* `ptbr-e5-small`
* `multilingual-e5-base`
* `Serafim-900M`

### `avaliar_metricas_bertopic_multimodelo.py`

Calcula métricas quantitativas para comparação dos modelos, incluindo:

* coerência C_V;
* NPMI;
* diversidade lexical;
* RBO invertido.

### `rodar_corte_conceitual.py`

Realiza a consolidação hierárquica dos tópicos gerados pelo BERTopic. No artigo, o corte conceitual adotado foi `dist-cut = 0.90`, que consolidou 44 tópicos válidos em 21 macro-tópicos.

## Principais resultados

O melhor equilíbrio entre coerência semântica e aproveitamento do corpus foi obtido com o modelo `ptbr-e5-small`.

Com esse modelo, o BERTopic gerou:

* **44 tópicos válidos**;
* **2.066 outliers**, correspondentes a **41,02%** das estrofes modeladas;
* **21 macro-tópicos** após consolidação hierárquica com corte conceitual 0,90.

Os dez macro-tópicos mais frequentes reuniram **2.511 das 2.971 estrofes agrupadas**, correspondendo a **84,5%** das estrofes não classificadas como outliers. Os macro-tópicos diretamente associados a relações afetivas, desejo, disputa, ex-relacionamento, perda e mudança de vida somaram **1.878 estrofes**, equivalentes a **63,2%** das estrofes agrupadas.

## Validação humana

Para reduzir a subjetividade da nomeação dos macro-tópicos, os rótulos dos dez macro-tópicos mais frequentes foram avaliados por quatro anotadores humanos.

Cada anotador atribuiu uma nota de 1 a 5 para a adequação entre o rótulo proposto e os elementos interpretativos do macro-tópico. Um rótulo foi considerado validado quando apresentou:

* média igual ou superior a 4,0;
* pelo menos três das quatro notas iguais ou superiores a 4.

Todos os dez rótulos avaliados foram validados, com média geral de **4,54**.

## Como executar

Recomenda-se utilizar Python 3.10 ou superior.

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instale as principais dependências:

```bash
pip install pandas numpy scikit-learn sentence-transformers bertopic umap-learn hdbscan gensim matplotlib seaborn openpyxl
```

Depois, execute os scripts na seguinte ordem:

```bash
python "pln-artigo/Códigos/letras_sertanejo_web_scrap.py"
python "pln-artigo/Códigos/rodar_bertopic_5_modelos_unificado.py"
python "pln-artigo/Códigos/avaliar_metricas_bertopic_multimodelo.py"
python "pln-artigo/Códigos/rodar_corte_conceitual.py"
```

Observação: dependendo do ambiente local, pode ser necessário ajustar caminhos de entrada e saída nos scripts.

## Reprodutibilidade

Os arquivos de saída armazenados nas pastas `Outputs Bertopic/` e `Outputs Cortes Conceituais/` permitem verificar os resultados utilizados no artigo, incluindo:

* documentos com tópicos atribuídos;
* métricas dos modelos;
* tópicos hierárquicos;
* macro-tópicos consolidados;
* outliers;
* cortes conceituais testados.

## Observação sobre direitos autorais

Este repositório tem finalidade acadêmica e de reprodutibilidade científica. As letras musicais são textos protegidos por direitos autorais, e seu uso deve respeitar as condições legais aplicáveis. Recomenda-se que redistribuições públicas priorizem metadados, saídas derivadas e scripts de processamento, evitando a reprodução integral de letras quando não houver autorização específica.

## Autoria

**Amanda Schmieleski Cossa Manfredini**
Instituto de Informática — Universidade Federal do Rio Grande do Sul (UFRGS)

**Karin Becker**
Instituto de Informática — Universidade Federal do Rio Grande do Sul (UFRGS)

## Declaração de uso de IA generativa

Foi utilizado o ChatGPT, versão GPT-5.5 Thinking, como apoio à geração e depuração de código e à revisão textual do artigo. Todos os códigos, parâmetros, resultados, rótulos e interpretações finais foram revisados, validados e assumidos integralmente como responsabilidade das autoras.

## Como citar

Caso utilize este material, cite o artigo associado ao repositório:

```bibtex
@inproceedings{manfredini2025sertanejo,
  title     = {Semantic Macro-Themes in Sertanejo Universit{\'a}rio Lyrics: Topic Modeling with BERTopic},
  author    = {Manfredini, Amanda Schmieleski Cossa and Becker, Karin},
  booktitle = {Symposium on Knowledge Discovery, Mining and Learning (KDMiLe)},
  year      = {2026}
}
```
