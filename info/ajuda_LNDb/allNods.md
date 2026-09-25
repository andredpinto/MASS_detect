A negrito, as colunas relevantes para usar no nosso modelo. 

**LNDbID** - corresponde ao ID do CT Scan onde se observa o nódulo; mapeia diretamente para o ficheiro .mhd correspondente (LNDbID 1 = LNDb-0001.mhd)
**RadID** - ID do radiologista que anotou o nódulo (pode ter vários radiologistas)
**RadFinding** - Nº de ordem do nódulo por radiologista e por CT
**FindingID** - ID do nódulo relativo ao CT scan
**Nodule** - 1 se for nódulo, 0 se for não-nódulo, para cada radiologista que anotou
**x** - coordenada do centro do nódulo no eixo x, em mm, no sistema de coordenadas do mundo real
**y** - coordenada do centro do nódulo no eixo y, em mm, no sistema de coordenadas do mundo real
**z** - coordenada do centro do nódulo no eixo z, em mm, no sistema de coordenadas do mundo real
**DiamEq_Rad** - Diâmetro equivalente à segmentação
**Características: Texture, Calcification, InternalStructure, Lobulation, Malignancy, Sphericity, Spiculation, Subtlety** - média das anotações feitas pelos radiologistas
**Lobe** - 1 'Left Upper Lobe', 2 'Left Lower Lobe', 3 'Right Upper Lobe', 4 'Middle Lobe', 5 'Right Lower Lobe'
TextInstanceID - ID da instância onde aparece no relatório médico
TextQuestion -  Incerteza
**Pos_Text** - mais detalhes sobre a localização do nódulo
Diam_Text - diâmetro escrito no relatório médico
**NodType** - tipo de nódulo
Charact_Text - características relacionadas com o nódulo no relatório médico
**Where** - se o nódulo está no relatório médico, na anotação do CT Scan ou em ambos. Não considerar nódulos que estejam apenas no relatório médico.