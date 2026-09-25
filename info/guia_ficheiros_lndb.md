##### **-------------------- FICHEIROS .CSV -------------------------**

**allNods.csv** -> lista completa de todos os nódulos no dataset; tanto nódulos anotados manualmente pelos radiologistas, como nódulos mencionados em relatórios médicos textuais.

Inclui...

* noduleID
* scanID
* coordenadas (se disponíveis)
* links de descrição textual



\-------------------------------------------------------------------------------------------

**chars\_trainNodules.csv** -> características dos nódulos anotados pelos radiologistas usados para treino:

* tamanho
* tipo (sólido, parte-sólido, ground-glass)
* localização
* morfologia
* features de consenso de radiologistas



\-------------------------------------------------------------------------------------------

**report.csv** -> extração tabular estruturada dos relatórios médicos com cada CT scan:

* metadados do paciente (género e ano de nascimento)
* descrição textual do radiologista
* findings
* recomendações



\-------------------------------------------------------------------------------------------

**rad2Fleischner.csv** -> mapeamento de nódulos anotados em imagem para categorias da Fleischner Society:

* categoria Fleischner (low risk, high risk, follow-up recommendation)
* tamanho/tipo do nódulo
* ID de anotação do radiologista



\-------------------------------------------------------------------------------------------

**text2Fleischner.csv** -> mapeamento de nódulos descritos textualmente (de relatórios) para categorias Fleischner:

* descrição do nódulo extraído 
* categoria Fleischner
* link para entrada no relatório 



\-------------------------------------------------------------------------------------------

**testNodules.csv** -> lista de nódulos para o test set do LNDb challenge:

* Nodule ID
* Scan ID
* Características básicas
* Ground truth labels



\-------------------------------------------------------------------------------------------

**testCTs.csv** -> list of CT scans belonging to the test set:

* scan identifiers
* acquisition metadata reference



\-------------------------------------------------------------------------------------------

**LNDbAcqParams.csv** -> acquisition parameters for each CT scan:

* slice thickness
* reconstruction kernel
* scanner model
* radiation dose



##### **-------------------- CT SCAN IMAGES -------------------------**

**data0.rar até data5.rar** -> os CT scans de treino. Cada arquivo contém um subset que, no seu conjunto, forma o dataset inteiro. 



\-------------------------------------------------------------------------------------------

**masks.rar** -> máscaras de segmentação para nódulos anotados:

* máscaras binárias
* uma máscara para cada nódulo
* alinhado com os volumes do CT



\-------------------------------------------------------------------------------------------

**testdata0.rar e testdata1.rar** -> os CT Scans de treino usados no LNDb challenge.



##### **------------------ AJUDA \& DOCUMENTAÇÃO ---------------------**

**README.pdf** -> a documentação principal do dataset:

* estrutura
* protocolo de anotação
* como os nódulos foram identificados
* como usar o dataset
* regras do LNDb challenge



\-------------------------------------------------------------------------------------------

**help.txt** -> explica todas as variáveis presentes nos ficheiros CSV.

* definição das colunas
* significado de cada feature
* como anotações textuais e de imagem foram emparelhadas
* 

##### **------------------ SCRIPTS \& TEMPLATES ----------------------**

**scripts.zip** -> scripts utility usados na LNDb challenge:

* pré-processamento
* métricas de avaliação
* exemplos de pipelines de modelo
* file loaders



\-------------------------------------------------------------------------------------------

**submission.zip** -> template para submeter predições para a LNDb challenge:

* formatos de ficheiros
* exemplo de submissão
* instruções para o packaging de resultados



\-------------------------------------------------------------------------------------------

**trainset\_csv.zip** -> ficheiros .csv comprimidos para o set de treino:

* provavelmente inclui definições de split
* metadados de treino
* listas de nódulos para treino.





