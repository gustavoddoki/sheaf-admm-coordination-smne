# Resultados confirmatórios P7–P8

Data: 08/10/2026. Este relatório consolida as análises confirmatórias pré-registradas sobre coordenação entre agentes. Sudoku é o ambiente controlado; o desfecho primário mede qualidade da inferência distribuída antes dos compromissos discretos, não velocidade nem taxa de resolução como objetivo principal.

## Resultado central

Os experimentos separam duas perguntas complementares. P7 mostra que, mantendo fixa a política de commits, ligar as mensagens contínuas de Sheaf-ADMM melhora a log-probabilidade do valor correto antes do commit. P8 mostra que a importância marginal do orçamento de mensagens depende da informação inicial: a redução de comunicação prejudica mais a acurácia quando há menos pistas, e essa diferença diminui ao aumentar as pistas. Em conjunto, os resultados apoiam a interpretação de que os agentes usam as mensagens para coordenar inferências locais, e que informação inicial mais abundante pode compensar parte da comunicação reduzida.

## Desenho e integridade

O protocolo foi congelado antes de abrir os holdouts em [`P7_P8_PROTOCOLO_CONGELADO_2026-10-07.md`](P7_P8_PROTOCOLO_CONGELADO_2026-10-07.md). O conjunto final contém 136 famílias novas: oito de desenvolvimento, 64 reservadas a P7 (IDs 8–71) e 64 a P8 (IDs 72–135). A auditoria dos dados passou em 816 variantes; o SHA-256 do manifesto é `4993c25229a42e0f46a8463e57d8f32049a54fa34a3ad8afd1fc5cfb38bd101c`.

P7 executou seis seeds, duas posições do agente e seis condições por combinação seed × posição, totalizando 72 trajetórias de oito passadas. P8 executou seis seeds e duas posições; cruzou 32/40/48 pistas, distribuição equilibrada/concentrada, comunicação ADMM completa/50%, máscaras previstas e inferência direta/decimada, totalizando 576 trajetórias. Em P8, os compromissos discretos foram globais em todas as condições; portanto, a manipulação de comunicação é especificamente do canal ADMM.

| Auditoria | P7 | P8 |
|---|---:|---:|
| Famílias independentes | 64 | 64 |
| Trajetórias completas | 72 | 576 |
| Passadas verificadas com sidecar SHA-256 | 576/576 | 4.608/4.608 |
| Linhas pareadas por família | 4.608 | 36.864 |
| Atualizações de vista roteadas | 119.242 | 1.851.987 |
| Veredito | PASS | PASS |

As duas auditorias confirmaram que uma alteração em uma única vista não muda os logits remotos sem ADMM (delta máximo 0), que commits locais são enviados somente ao agente proponente e que commits globais chegam às três vistas incidentes. A reconstrução das vistas coincide com o tabuleiro canônico. P8 teve uma interrupção durante a execução; a retomada continuou a trajetória que já tinha passadas salvas, sem refazer as 385 trajetórias completas. A auditoria final verificou todos os arquivos e sidecars.

## P7 — efeito das mensagens ADMM e da visibilidade dos commits

O desfecho primário é a média familiar da log-probabilidade do valor correto, com peso igual por vista de agente e congelada antes de cada commit. A unidade independente é a família de solução; seeds, posição e máscaras são medidas repetidas. Os contrastes pareados comparam ADMM ligado versus desligado mantendo constante a visibilidade global ou local dos commits. ICs são bootstrap por família com 20.000 reamostragens; os dois testes primários têm ajuste de Holm.

| Contraste | Efeito em log-probabilidade | IC95% bootstrap por família | p ajustado de Holm |
|---|---:|---:|---:|
| ADMM ligado − desligado, commits globais | +1,4986 | [1,4603; 1,5326] | 4,15 × 10⁻⁶⁵ |
| ADMM ligado − desligado, commits locais | +1,2149 | [1,1788; 1,2464] | 3,94 × 10⁻⁶¹ |

O contraste secundário entre os efeitos global e local foi +0,2838. Com ADMM ligado, os commits médios por família foram 40,51 na rota global e 32,37 na local; em ambas, cerca de 0,26 commits incorretos em média. A rota global gerou 121,52 atualizações de vista por família, contra 32,37 na local. Esses resultados distinguem o ganho das mensagens contínuas do alcance do canal discreto. Não demonstram que a comunicação acelera a resolução, pois tempo e velocidade não são o estimando primário.

## P8 — quantidade de pistas como moderador da comunicação

O desfecho primário é acurácia pré-commit nas vistas dos agentes sob decimação e commits globais. A interação pré-especificada é `(acurácia em 48 pistas com 100% − 50% de comunicação) − (acurácia em 32 pistas com 100% − 50%)`, estimada separadamente por distribuição. Os dois testes são ajustados por Holm; ICs são bootstrap por família.

| Distribuição | Interação 48−32 × comunicação | IC95% | p ajustado de Holm |
|---|---:|---:|---:|
| Equilibrada | −15,68 pontos percentuais | [−17,31; −14,03] pp | 6,47 × 10⁻²⁷ |
| Concentrada | −13,92 pontos percentuais | [−15,55; −12,19] pp | 7,50 × 10⁻²⁴ |

O sinal negativo indica que a perda de acurácia ao reduzir a comunicação é menor com 48 pistas do que com 32. As médias de acurácia decimada ilustram a magnitude:

| Distribuição | Pistas | Comunicação 100% | Comunicação 50% | Perda ao reduzir para 50% |
|---|---:|---:|---:|---:|
| Equilibrada | 32 | 94,20% [92,22; 95,97] | 77,89% [75,84; 79,93] | 16,31 pp |
| Equilibrada | 40 | 99,98% [99,96; 100,00] | 97,30% [96,64; 97,86] | 2,68 pp |
| Equilibrada | 48 | 100,00% [99,99; 100,00] | 99,37% [99,19; 99,52] | 0,63 pp |
| Concentrada | 32 | 94,20% [92,20; 96,00] | 77,89% [75,81; 79,92] | 16,31 pp |
| Concentrada | 40 | 99,11% [98,41; 99,59] | 92,27% [90,86; 93,58] | 6,84 pp |
| Concentrada | 48 | 99,79% [99,64; 99,92] | 97,41% [96,83; 97,93] | 2,39 pp |

Assim, o efeito não se limita à distribuição equilibrada: sob 50% de comunicação, a condição concentrada perde mais acurácia que a equilibrada em 40 e 48 pistas, enquanto com 32 as médias coincidem no desenho usado. Isso sustenta moderação pela quantidade e também sugere que a distribuição importa nos níveis intermediários/altos. As curvas completas com intervalos estão em [`P8_CLUES_COMMUNICATION.png`](../../outputs/coordination_channel_ablation/p8/P8_CLUES_COMMUNICATION.png) e [PDF](../../outputs/coordination_channel_ablation/p8/P8_CLUES_COMMUNICATION.pdf).

### Sensibilidade exploratória na log-probabilidade pré-commit

Como a acurácia se aproxima de 100% em condições mais informativas, reanalisamos a mesma interação 48−32 pistas × comunicação sobre a log-probabilidade correta pré-commit, métrica secundária registrada no protocolo. Para cada família e célula experimental, as medidas de seeds, posições e máscaras foram primeiro agregadas; em seguida, a interação foi reamostrada por família (20.000 amostras percentis). A estimativa foi **−0,44497** (IC95% **[−0,59876; −0,27289]**) para pistas equilibradas e **−0,31994** (**[−0,48086; −0,13685]**) para pistas concentradas. A reconstrução do contraste de acurácia reproduziu exatamente os dois estimandos pontuais confirmatórios em `analysis.json`.

Esta análise é uma sensibilidade exploratória: o protocolo não definiu a interação de log-probabilidade como contraste confirmatório, portanto não reportamos p-valores nem fazemos uma alegação confirmatória adicional. A concordância de sinal mostra que o padrão não aparece apenas porque a acurácia satura, mas não separa a quantidade de informação da maior dedutibilidade dos puzzles com mais pistas.

## Relação com P5 e contribuição para o artigo

O fechamento global anterior está registrado como concluído em `outputs/article_followup/EXECUTION_STATE.json`: P0/P1 fechados, 12 checkpoints X prontos em P2, P3/P4 holdout concluídos e auditados, P5 baselines/latência concluídos, e P6 de mecanismo concluído em desenvolvimento. A análise confirmatória P3/P4 está em [`ARTICLE_CONFIRMATORY_ANALYSIS.md`](../../outputs/article_followup/ARTICLE_CONFIRMATORY_ANALYSIS.md); o sumário consolidado anterior permanece em [`EXPERIMENTOS_PRIORITARIOS_RESULTADOS_CONSOLIDADOS_2026-10-02.md`](EXPERIMENTOS_PRIORITARIOS_RESULTADOS_CONSOLIDADOS_2026-10-02.md). Esses estudos dão contexto sobre o sistema, mas não são contados como novas réplicas dos holdouts P7/P8.

P5 já havia mostrado transmissão causal local: uma pista inserida nas três vistas incidentes alterou a inferência de agentes a uma aresta quando ADMM estava ligado, com efeito primário +0,01598 em log-probabilidade (IC95% [0,01526; 0,01667]); o controle com CG local e mensagens bloqueadas permaneceu invariável. A análise completa está em [`P5_LOCAL_TRANSMISSION_RESULTS_2026-10-07.md`](P5_LOCAL_TRANSMISSION_RESULTS_2026-10-07.md). O P5 funcional mostrou que a decimação mantém alta exatidão sob mensagens reduzidas, embora a taxa exata média caia de 97,0% com comunicação completa para 89,2% a 50%; sua interpretação é compensação parcial, não benefício de reduzir mensagens ([resultados P5](P5_COMMUNICATION_DECIMATION_RESULTS_2026-10-07.md)).

A contribuição empírica conjunta pode ser apresentada como uma análise dos canais de coordenação do Sheaf-ADMM em três escalas: (1) a mensagem contínua transmite informação para agentes vizinhos (P5); (2) ela melhora a inferência mesmo quando o canal de commits é controlado (P7); e (3) a dependência dessas mensagens diminui quando há mais informação inicial disponível, tanto em pistas equilibradas quanto concentradas (P8). O ambiente Sudoku serve como sistema de teste das dinâmicas de proposta, consenso e commit; não é a alegação de contribuição em si.

## Limites

As conclusões são condicionais aos checkpoints treinados, às 64 famílias holdout de cada fase e ao desenho de Sudoku utilizado. P8 mede a moderação sob decimação com commits globais; não estima a interação entre pistas e visibilidade local/global de commits, que foi tratada em P7. Acurácia pré-commit não equivale a taxa de solução exata nem a tempo de execução. Não se deve afirmar que o sistema resolve mais rápido, que melhora em todos os problemas multiagente ou que a distribuição concentrada é sempre inferior. Nenhuma alegação de velocidade é sustentada por estes experimentos.

O pico RSS medido integralmente para P7 foi 548.179.968 bytes. A sessão de amostragem de recursos P8 não produziu um arquivo final verificável; por isso não reportamos pico de memória para P8. As duas fases foram executadas em CPU local, sem uso de GPU ou recurso pago.

## Artefatos e reprodução

- Registro operacional e retomadas: [`P7_P8_EXECUTION_LOG_2026-10-08.md`](P7_P8_EXECUTION_LOG_2026-10-08.md)
- Auditoria e análise P7: [`audit.json`](../../outputs/coordination_channel_ablation/p7/audit.json), [`analysis.json`](../../outputs/coordination_channel_ablation/p7/analysis.json), [`analysis.md`](../../outputs/coordination_channel_ablation/p7/analysis.md), [figura PNG](../../outputs/coordination_channel_ablation/p7/P7_ADMM_CHANNEL_EFFECT.png) e [PDF](../../outputs/coordination_channel_ablation/p7/P7_ADMM_CHANNEL_EFFECT.pdf)
- Auditoria e análise P8: [`audit.json`](../../outputs/coordination_channel_ablation/p8/audit.json), [`analysis.json`](../../outputs/coordination_channel_ablation/p8/analysis.json), [`analysis.md`](../../outputs/coordination_channel_ablation/p8/analysis.md), [figura PNG](../../outputs/coordination_channel_ablation/p8/P8_CLUES_COMMUNICATION.png) e [PDF](../../outputs/coordination_channel_ablation/p8/P8_CLUES_COMMUNICATION.pdf)
- Sensibilidade exploratória P8 na log-probabilidade: `scripts/analyze_p8_logprob_sensitivity.py` e [`logprob_interaction_sensitivity.json`](../../outputs/coordination_channel_ablation/p8/logprob_interaction_sensitivity.json); não integra os contrastes confirmatórios.
- Manifesto de dados: `datasets/sudoku_coordination_final_20261008/manifest.json`

Os hashes de configuração são `bfbe6591c3912fe05ba4e1ee3ef205a4565f562b199e87fa6c0827018bd1ba91` (P7) e `0a24e3673f0b78d6681e1b637ddc23587abcd9a8fafd804dc71fb21c0ffd63e0` (P8). A auditoria e a análise podem ser repetidas com:

```powershell
.\.venv\Scripts\python.exe scripts/audit_coordination_channel_ablation.py outputs/coordination_channel_ablation/p7
.\.venv\Scripts\python.exe scripts/analyze_coordination_channel_ablation.py outputs/coordination_channel_ablation/p7 --bootstrap-draws 20000 --seed 20261008
.\.venv\Scripts\python.exe scripts/audit_coordination_channel_ablation.py outputs/coordination_channel_ablation/p8
.\.venv\Scripts\python.exe scripts/analyze_coordination_channel_ablation.py outputs/coordination_channel_ablation/p8 --bootstrap-draws 20000 --seed 20261008
```
