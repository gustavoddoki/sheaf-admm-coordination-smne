# Coordenação de agentes Sheaf-ADMM: materiais de reprodução

Materiais selecionados de dois artigos de trabalho sobre coordenação de agentes Sheaf-ADMM:

1. **Informação local e canais de comunicação na coordenação de agentes Sheaf-ADMM**: experimentos de transmissão local (P5), separação dos canais (P7) e informação inicial (P8). Materiais na raiz deste repositório, descritos abaixo.
2. **Memória dual ativa e reservada em Sheaf-ADMM sob comunicação intermitente**: protocolos das duas etapas (48 famílias de desenvolvimento e 96 inéditas), métricas por família, dados sintéticos de avaliação e script de reanálise. Materiais na pasta [`memoria_reservada/`](memoria_reservada/), com README e manifesto próprios.

Autores dos estudos: Gustavo Daiske Mashiba Doki e Gustavo Valentim Loch (orientação). Os manuscritos estão em preparação para o SMNE; este depósito não indica publicação ou aceite.

## Informação local e canais de comunicação (P5, P7, P8)

### Escopo científico

O estudo investiga intervenções durante a inferência em 12 checkpoints experimentais (seis seeds × projeções interna/terminal), treinados com 4.096 Sudokus por cinco épocas. Cada ensaio principal usa 64 famílias de soluções-base; o complemento funcional de decimação reutiliza a coorte P5. Novas coortes de teste não são novos treinamentos independentes.

Os resultados caracterizam esses modelos e condições. Não demonstram superioridade no benchmark completo, aprendizagem exclusiva de coordenação, vantagem de treinamento com comunicação ou eficiência física de rede. O estudo E1 e outras extensões da dissertação não integram este depósito.

### Conteúdo

- `src/` e `configs/`: snapshot da implementação e configurações; parte do código deriva do projeto original.
- `scripts/`: executores, análises, auditorias e suas dependências locais necessárias ao recorte.
- `experiments/sheaf_admm_reasoning/`: protocolos e relatórios originais. Caminhos históricos documentam a execução; nem todos os arquivos citados estão neste depósito.
- `outputs/`: análises, tabelas familiares selecionadas, configurações e relatórios de auditoria preservados.
- `datasets/`: famílias sintéticas, arrays de avaliação e manifestos selecionados; não inclui o corpus original de treinamento nem o inventário histórico completo usado para excluir soluções anteriores.
- `experiments/sheaf_admm_reasoning/smne_2026/figures/`: figuras editoriais em PNG e PDF.

### Reprodução e limites

Em Python 3.10 ou superior, instale `python -m pip install -e .`. O arquivo `upstream_uv.lock` registra o lock do projeto de origem; não é um lock atualizado deste pacote com dependências adicionais de análise.

Para reproduzir a análise estatística de P7 e P8 a partir das tabelas familiares distribuídas:

```sh
python scripts/analyze_coordination_channel_ablation.py outputs/coordination_channel_ablation/p7 --bootstrap-draws 20000 --seed 20261008
python scripts/analyze_coordination_channel_ablation.py outputs/coordination_channel_ablation/p8 --bootstrap-draws 20000 --seed 20261008
```

Execute essas análises em uma cópia se quiser preservar os agregados depositados. Confira a interface de cada executor com `--help` e os parâmetros nos protocolos. Resultados exploratórios estão identificados nos relatórios e não são novas confirmações.

**Este depósito não permite reproduzir toda a inferência de ponta a ponta:** checkpoints treinados, logits brutos, chunks de todas as trajetórias, corpus original de treinamento e parte dos inventários históricos não estão incluídos. Os executores estão disponíveis, mas dependem desses artefatos nos caminhos registrados. As auditorias históricas distribuídas foram executadas no acervo original completo; sua presença aqui não significa que todos os seus arquivos de entrada estejam públicos. Os dados brutos e pesos poderão ser disponibilizados em uma versão posterior. Não anunciamos reprodução integral neste estágio.

## Memória dual ativa e reservada

A pasta [`memoria_reservada/`](memoria_reservada/) é autocontida. Com Python 3 e NumPy, `python reproduzir_metricas.py` (executado nessa pasta) confere os checksums do manifesto da pasta e reconstrói as médias por política nas duas etapas. Conteúdo, reprodução e limites estão no [README da pasta](memoria_reservada/README.md).

## Integridade dos arquivos

`PUBLICATION_MANIFEST.json` lista os arquivos distribuídos com seus checksums. Para verificá-los: `python verify_manifest.py`. Isso verifica checksums, não a validade científica dos experimentos.

Caminhos absolutos da estação foram convertidos em caminhos relativos em alguns JSONs. O manifesto lista os arquivos normalizados e o hash original. Os hashes de artefatos presentes em auditorias históricas referem-se ao acervo original; o manifesto público verifica as cópias distribuídas. Nenhuma estimativa foi modificada.

## Origem e licença

A arquitetura Sheaf-ADMM e a implementação-base são de Jeffrey Seely, Bartłomiej Cupiał e Llion Jones: [SakanaAI/sheaf-admm](https://github.com/SakanaAI/sheaf-admm). Este é um snapshot derivado para os estudos experimentais, não o repositório oficial. Os metadados do pacote preservam a autoria do software original. Alterações experimentais estão identificadas em `NOTICE` e documentadas nos protocolos.

O código é distribuído sob Apache-2.0, com a licença original preservada em `LICENSE`. Referência da arquitetura: Seely, Cupiał e Jones (2026), *Learning Multi-Agent Coordination via Sheaf-ADMM*. Os autores destes estudos não reivindicam autoria de ADMM, Sinkhorn ou da arquitetura original.
