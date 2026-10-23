# Informação local e canais de comunicação em agentes Sheaf-ADMM

Materiais selecionados dos experimentos de transmissão local (P5), separação dos canais (P7) e informação inicial (P8), associados ao artigo de trabalho **Informação local e canais de comunicação na coordenação de agentes Sheaf-ADMM**.

Autores do estudo: Gustavo Daiske Mashiba Doki e Gustavo Valentim Loch (orientação). O manuscrito está em preparação para o SMNE; este depósito não indica publicação ou aceite.

## Escopo científico

O estudo investiga intervenções durante a inferência em 12 checkpoints experimentais (seis seeds × projeções interna/terminal), treinados com 4.096 Sudokus por cinco épocas. Cada ensaio principal usa 64 famílias de soluções-base; o complemento funcional de decimação reutiliza a coorte P5. Novas coortes de teste não são novos treinamentos independentes.

Os resultados caracterizam esses modelos e condições. Não demonstram superioridade no benchmark completo, aprendizagem exclusiva de coordenação, vantagem de treinamento com comunicação ou eficiência física de rede. O estudo E1 e outras extensões da dissertação não integram este depósito.

## Conteúdo

- `src/` e `configs/`: snapshot da implementação e configurações; parte do código deriva do projeto original.
- `scripts/`: executores, análises, auditorias e suas dependências locais necessárias ao recorte.
- `experiments/sheaf_admm_reasoning/`: protocolos e relatórios originais. Caminhos históricos documentam a execução; nem todos os arquivos citados estão neste depósito.
- `outputs/`: análises, tabelas familiares selecionadas, configurações e relatórios de auditoria preservados.
- `datasets/`: famílias sintéticas, arrays de avaliação e manifestos selecionados; não inclui o corpus original de treinamento nem o inventário histórico completo usado para excluir soluções anteriores.
- `experiments/sheaf_admm_reasoning/smne_2026/figures/`: figuras editoriais em PNG e PDF.
- `PUBLICATION_MANIFEST.json`: inventário e checksums dos arquivos distribuídos.

## Reprodução e limites do depósito

Em Python 3.10 ou superior, instale `python -m pip install -e .`. O arquivo `upstream_uv.lock` registra o lock do projeto de origem; não é um lock atualizado deste pacote com dependências adicionais de análise.

Para reproduzir a análise estatística de P7 e P8 a partir das tabelas familiares distribuídas:

```sh
python scripts/analyze_coordination_channel_ablation.py outputs/coordination_channel_ablation/p7 --bootstrap-draws 20000 --seed 20261008
python scripts/analyze_coordination_channel_ablation.py outputs/coordination_channel_ablation/p8 --bootstrap-draws 20000 --seed 20261008
```

Execute essas análises em uma cópia se quiser preservar os agregados depositados. Confira a interface de cada executor com `--help` e os parâmetros nos protocolos. Resultados exploratórios estão identificados nos relatórios e não são novas confirmações.

**Este depósito não permite reproduzir toda a inferência de ponta a ponta:** checkpoints treinados, logits brutos, chunks de todas as trajetórias, corpus original de treinamento e parte dos inventários históricos não estão incluídos. Os executores estão disponíveis, mas dependem desses artefatos nos caminhos registrados. As auditorias históricas distribuídas foram executadas no acervo original completo; sua presença aqui não significa que todos os seus arquivos de entrada estejam públicos. Os dados brutos e pesos poderão ser disponibilizados em uma versão posterior. Não anunciamos reprodução integral neste estágio.

Para verificar a integridade dos arquivos distribuídos: `python verify_manifest.py`. Isso verifica checksums, não a validade científica dos experimentos.

## Origem e licença

A arquitetura Sheaf-ADMM e a implementação-base são de Jeffrey Seely, Bartłomiej Cupiał e Llion Jones: [SakanaAI/sheaf-admm](https://github.com/SakanaAI/sheaf-admm). Este é um snapshot derivado para o estudo experimental, não o repositório oficial. Os metadados do pacote preservam a autoria do software original. Alterações experimentais estão identificadas em `NOTICE` e documentadas nos protocolos.

O código é distribuído sob Apache-2.0, com a licença original preservada em `LICENSE`. Referência da arquitetura: Seely, Cupiał e Jones (2026), *Learning Multi-Agent Coordination via Sheaf-ADMM*. Os autores deste estudo não reivindicam autoria de ADMM, Sinkhorn ou da arquitetura original.

## Metadados portáveis

Caminhos absolutos da estação foram convertidos em caminhos relativos em alguns JSONs. O manifesto lista os arquivos normalizados e o hash original. Os hashes de artefatos presentes em auditorias históricas referem-se ao acervo original; o manifesto público verifica as cópias distribuídas. Nenhuma estimativa foi modificada.
