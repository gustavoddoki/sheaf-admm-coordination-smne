# P5 funcional — resultados de comunicação × decimação

Data: 07/10/2026. Experimento funcional secundário pré-especificado no protocolo P5. A pergunta mecanística primária de P5 — transmissão de informação local por ADMM — está documentada em [P5_LOCAL_TRANSMISSION_RESULTS_2026-10-07.md](P5_LOCAL_TRANSMISSION_RESULTS_2026-10-07.md).

## Execução e auditoria

O holdout contém 64 famílias de soluções-base inéditas, avaliadas em seis seeds de checkpoint (123, 456, 789, 1011, 1213 e 20260927), posições interna/terminal, comunicação completa, máscaras aleatórias de 75% e 50% (três máscaras por fração) e controle sem mensagens ADMM com CG local. Cada bloco comparou inferência direta e decimação, até oito passadas e K=20 por passada.

A execução CPU terminou com 96/96 blocos e 6.144 linhas de orçamento pareado. A auditoria independente passou: 1.536/1.536 arquivos de passada e sidecars SHA-256 válidos, hashes dos 12 checkpoints válidos e 192 passadas de controle sem mensagens ADMM verificadas. O auditor registrou 98.304 linhas por passada e 5.376 interações. Nenhuma computação paga foi usada.

Configuração: SHA-256 `ac4fee68c5a42d3dcb18affc5cde80844d9625aed6b77f8bd1a76f377e83a8a1`. O holdout, o protocolo e os checkpoints estão identificados em [run_config.json](../../outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68/run_config.json), [audit.json](../../outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68/audit.json) e [partial_audit.json](../../outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68/partial_audit.json).

## Estimando e análise

A análise agrega primeiro as três máscaras dentro de checkpoint, depois os seis checkpoints e, por fim, as posições interna e terminal com peso igual dentro da família. A unidade independente é a família de solução-base. Os intervalos percentis de 95% usam bootstrap de 20.000 reamostragens por família, seed 20261007. Os valores abaixo são percentuais; entre colchetes está o intervalo bootstrap de 95%.

| Posição | Comunicação | Direta exata, orçamento pareado | Decimação exata | Ganho exato da decimação |
|---|---|---:|---:|---:|
| Interna | Completa | 54,7 [43,5; 65,6] | 98,4 [95,3; 100,0] | 43,8 [32,8; 54,9] |
| Interna | Aleatória 75% | 9,4 [4,9; 14,6] | 95,1 [90,6; 98,5] | 85,7 [79,5; 91,1] |
| Interna | Aleatória 50% | 0,1 [0,0; 0,3] | 87,4 [81,9; 92,4] | 87,3 [81,8; 92,4] |
| Interna | Sem mensagens ADMM | 0,0 [0,0; 0,0] | 22,7 [15,1; 30,5] | 22,7 [15,1; 30,7] |
| Terminal | Completa | 36,2 [27,6; 45,1] | 95,6 [92,2; 98,4] | 59,4 [51,0; 67,4] |
| Terminal | Aleatória 75% | 19,3 [13,4; 25,8] | 95,9 [92,6; 98,6] | 76,6 [70,2; 82,6] |
| Terminal | Aleatória 50% | 2,6 [1,1; 4,4] | 91,1 [85,5; 95,7] | 88,5 [82,8; 93,1] |
| Terminal | Sem mensagens ADMM | 0,0 [0,0; 0,0] | 10,7 [7,0; 14,6] | 10,7 [7,0; 14,6] |

**Correção de consistência em 08/10/2026.** A conferência da tabela com `summary_family_bootstrap.csv` e `analysis.json` encontrou um erro de transcrição na estimativa pontual da inferência direta terminal com máscara aleatória de 75%: o valor correto é 19,27% (19,3% arredondado a uma casa), não 19,5%; o IC95% continua [13,4; 25,8]. Recalculadas com peso igual entre posições a partir das médias por família, as estimativas agregadas de inferência direta são 45,4%, 14,3%, 1,3% e 0,0% para comunicação completa, 75%, 50% e controle sem mensagens ADMM. Nenhuma execução experimental foi refeita; os valores corrigidos mantêm o estimando congelado no protocolo.

O contraste funcional pré-especificado foi:

`(ganho da decimação com comunicação completa) − (ganho da decimação com máscara aleatória de 50%)`.

Com peso igual entre as duas posições, a estimativa foi **−36,33 pontos percentuais**, IC95% por família **[−47,14; −25,39]**. Os efeitos por posição foram −43,58 pp [−57,73; −29,17] na posição interna e −29,08 pp [−38,54; −19,62] na terminal.

O sinal é contrário à hipótese direcional de que o ganho incremental da decimação seria maior com comunicação completa. A decimação mantém alta taxa exata com mensagens reduzidas, enquanto o controle direto pareado cai para quase zero nas famílias internas e 2,6% nas terminais sob 50%. Assim, a interação negativa decorre principalmente do colapso do baseline direto: o ganho da decimação sobre esse baseline torna-se maior quando a comunicação é reduzida. Isso indica tolerância/compensação funcional da decimação, não que reduzir mensagens melhore a solução final. A taxa exata média da decimação cai de 97,0% com comunicação completa para 89,2% a 50%.

O controle sem mensagens ADMM teve 16,7% de exatidão média sob decimação (22,7% interna; 10,7% terminal), apesar de ainda compartilhar compromissos pelo tabuleiro. Comparado aos 97,0% com comunicação completa, esse resultado é compatível com contribuição das mensagens ADMM para a inferência final. A comparação não representa isolamento de toda comunicação, pois a propagação discreta dos commits permanece ativa.

## Compromissos e canais de comunicação

A média de commits corretos/incorretos por puzzle, calculada por família com medidas repetidas agregadas, foi:

| Posição | Completa | Aleatória 75% | Aleatória 50% | Sem mensagens ADMM |
|---|---:|---:|---:|---:|
| Interna | 6,83 / 0,05 | 17,90 / 0,22 | 23,78 / 0,48 | 28,83 / 0,13 |
| Terminal | 12,21 / 0,23 | 17,79 / 0,14 | 23,41 / 0,09 | 19,93 / 0,00 |

Cada commit atualiza três vistas incidentes. No braço de 50%, isso corresponde a cerca de 72,8 atualizações de vista por puzzle na posição interna e 70,5 na terminal. Portanto, o canal discreto continua transmitindo informação mesmo quando as mensagens ADMM são bloqueadas.

O custo abaixo é payload **modelado**, não bytes de rede medidos. São MiB por passada de inferência do lote de 64 famílias, incluindo o bootstrap inicial:

| Condição | Vetores ADMM | Parciais do CG | Escalares do CG | Bootstrap inicial | Total por passada |
|---|---:|---:|---:|---:|---:|
| Completa | 37,97 | 0,92 | 0,03 | 1,90 | 40,82 |
| Aleatória 75% | 28,48 | 0,92 | 0,03 | 1,90 | 31,33 |
| Aleatória 50% | 18,98 | 0,92 | 0,03 | 1,90 | 21,84 |
| Sem mensagens ADMM | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |

O controle sem mensagens zera vetores e reduções do CG, mas não os commits do tabuleiro compartilhado. Os tempos registrados incluem compilação e não são usados para alegar aceleração.

## Conclusão e próximo passo

Com P4/P5 mecanísticos, há evidência de que uma pista local altera propostas remotas após a troca ADMM; o contraste funcional, por sua vez, mostra que a decimação compensa parcialmente a perda de mensagens, mas não elimina sua importância. A conclusão deve separar esses dois achados: **transmissão local confirmada; maior ganho incremental da decimação sob 50% de mensagens, com o baseline direto muito mais degradado**.

A principal ambiguidade restante é a contribuição relativa das mensagens contínuas e dos commits broadcast. O próximo experimento recomendado é o fatorial em famílias novas, cruzando ADMM ligado/desligado (CG local no controle) com visibilidade global/local dos commits. A análise de quantidade e distribuição de pistas fica depois dessa separação causal.

Os intervalos dos desfechos secundários são estimativas bootstrap sem ajuste simultâneo. Não foram calculados p-valores nem feitas alegações de significância para a família de métricas secundárias; a conclusão confirmatória aqui se limita ao contraste funcional pré-especificado e ao protocolo mecanístico reportado separadamente.

## Arquivos de reprodução

- [Análise JSON](../../outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68/analysis.json)
- [Contraste funcional primário](../../outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68/primary_functional_interaction.csv)
- [Resumo bootstrap por família](../../outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68/summary_family_bootstrap.csv)
- [Interações por família](../../outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68/interaction_family_bootstrap.csv)
- [Tráfego e commits](../../outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68/communication_traffic_summary.csv)
- [Figura PNG](../../outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68/P5_FUNCTIONAL_COMMUNICATION.png) e [PDF](../../outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68/P5_FUNCTIONAL_COMMUNICATION.pdf)

Com o ambiente Python do repositório ativo, a auditoria e a análise podem ser repetidas da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe scripts/audit_coordination_decimation_confirmation.py outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68
.\.venv\Scripts\python.exe scripts/analyze_coordination_communication_decimation.py outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68 --bootstrap-draws 20000 --seed 20261007
.\.venv\Scripts\python.exe scripts/plot_coordination_communication_decimation.py outputs/p5_decim/n64_s6_a2_p8m3_ac4fee68
```
