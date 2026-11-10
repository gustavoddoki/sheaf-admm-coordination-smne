# Memória dual ativa e reservada sob comunicação intermitente — materiais

Materiais de reprodução parcial do artigo de trabalho **Memória dual ativa e
reservada em Sheaf-ADMM sob comunicação intermitente** (Gustavo Daiske Mashiba
Doki; Gustavo Valentim Loch, orientação). O manuscrito está em preparação para
o SMNE; este depósito não indica publicação ou aceite. Ele é independente do
estudo P5/P7/P8 descrito no README da raiz deste repositório.

## Conteúdo

- `protocolos/`: protocolo da primeira etapa (`gate_original.json`, 48 famílias
  históricas) e da segunda etapa (`gate_m5.json`, 96 famílias inéditas), além da
  auditoria das famílias inéditas (`auditoria_familias_m5.json`).
- `metricas/` e `metricas_m5/`: métricas por família, rodada, perfil e condição
  para as três inicializações (789, 1011, 1213) em cada etapa.
- `dados/`: exemplos avaliados com seus rótulos (NPZ) e manifestos das famílias.
- `analises/`: resumos das análises pré-registradas (`summary_original.json`,
  `summary_m5.json`), análises secundárias, curvas por rodada e verificação de
  tabelas.
- `reproduzir_metricas.py`: confere os checksums de `MANIFEST_SHA256.json` e
  reconstrói as médias por política nas duas etapas.

## Reprodução

Com Python 3 e NumPy, nesta pasta:

```sh
python reproduzir_metricas.py
```

O script reconstrói, entre outros valores, a concordância correta final sob
falhas de 75,14% (retenção) e 65,15% (padrão) na primeira etapa e de 75,91% e
65,60% na segunda.

## Limites

O protocolo da segunda etapa foi congelado localmente, com hashes SHA-256, antes
da avaliação; este depósito é posterior à avaliação e não substitui um registro
externo prévio. Os protocolos contêm caminhos do repositório de origem e o
comando de execução, preservados para documentar a proveniência; os arquivos
citados nem sempre estão aqui. Nomes históricos como fresh/confirmation no
protocolo da primeira etapa não significam dados inéditos: essas 48 famílias já
haviam sido usadas em estudos-piloto do projeto.

O depósito permite reanalisar as métricas salvas. Não inclui checkpoints
treinados, trajetórias completas, dados de treinamento nem o texto do
manuscrito, e não permite repetir o treinamento ou a inferência de ponta a ponta.
