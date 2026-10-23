# P5 — confirmação de transmissão local de informação

Data: 07/10/2026. Execução confirmatória em CPU local; nenhuma GPU ou recurso pago foi usado.

## Resultado

Uma pista correta, inserida somente nas três vistas incidentes, alterou de forma consistente as propostas de agentes a uma aresta no grafo, depois da primeira atualização ADMM. O controle com CG local e mensagens ADMM bloqueadas permaneceu exatamente invariável nas propostas remotas. Isso confirma transmissão causal de informação ao longo das mensagens nos checkpoints avaliados, sem depender de atualização global do tabuleiro.

O desfecho primário pré-especificado foi o aumento da log-probabilidade do valor correto nas posições ocultas dos agentes remotos a uma aresta, com peso igual por agente, descontado o mesmo contraste no controle sem mensagens:

`[(pista − sem pista) com ADMM] − [(pista − sem pista) com CG local]`.

Em 64 soluções-base independentes, quatro células oráculo por solução, seis seeds de treino e as duas posições da regra, o efeito estimado foi **+0,01598 em log-probabilidade**, IC95% bootstrap por solução **[+0,01526; +0,01667]**. Todos os 12 estratos seed × posição da regra tiveram estimativa positiva. As estimativas foram aproximadamente 0,0077–0,0085 para interno e 0,0179–0,0279 para terminal; a comparação entre posições é descritiva e não isola causalmente a colocação de AllDifferent.

Resultados secundários, também agregados com peso igual por agente e bootstrap por solução:

| Distância do agente | Variação de acurácia oculta | Fração top-1 alterada | Divergência JS média |
|---|---:|---:|---:|
| 1 aresta | +1,39 pp [1,27; 1,51] | 5,82% [5,58; 6,07] | 0,001364 [0,001289; 0,001443] |
| 2 arestas | +0,03 pp [−0,06; 0,11] | 0,92% [0,82; 1,01] | 0,000015 [0,000014; 0,000016] |

A auditoria confirmou invariância remota em 36/36 artefatos sem mensagens ADMM e em 24/24 artefatos de K=1, antes que a primeira mensagem pudesse influenciar os agentes remotos. Foram verificados 72 hashes de saída, todos os checkpoints, o manifesto e os quatro oráculos por família. O valor completo da solução só foi usado para selecionar o dígito-oráculo e pontuar; não foi passado ao modelo.

## Ressalva de análise

O protocolo congelado especificava primeiro calcular a média das posições vazias por agente e depois dar peso igual aos agentes. O resumo automático inicial agregou todas as posições remotas de uma vez e, assim, deu maior peso aos agentes com mais células vazias. Os logits brutos estavam preservados; a análise confirmatória acima foi recalculada com a ponderação pré-especificada e auditada diretamente dos NPZs. O estimador ponderado por célula, 0,01747 [0,01671; 0,01823], permanece no arquivo de auditoria somente como diagnóstico. A discrepância mudou a magnitude, não o sinal ou a conclusão.

## Limites e interpretação

Este teste identifica que as mensagens transmitem informação útil no grafo para os checkpoints treinados avaliados. Ele não mostra, por si só, que mensagens são necessárias para toda cascata de decimação nem que um modelo treinado sem comunicação teria desempenho inferior. O teste usa uma pista-oráculo isolada, quatro posições por solução e um único regime informativo: Sudoku com 40 pistas concentradas. Os intervalos agrupam famílias e são condicionais aos seis checkpoints existentes.

O resultado está alinhado à oportunidade destacada no artigo-base: o Sheaf-ADMM torna as dinâmicas de proposta, consenso e dual acessíveis a análise e intervenção direta ([PMLR, ICML 2026](https://proceedings.mlr.press/v306/seely26a.html)). A próxima etapa funcional, comunicação × decimação, está rodando separadamente no mesmo holdout para testar se essa transmissão altera a cascata de compromissos.

## Artefatos

- Protocolo congelado: `P5_PROTOCOLO_CONFIRMATORIO_2026-10-07.md`
- Holdout e auditoria de dados: `outputs/coordination_pilot_20261007/p5_confirmation/holdout_audit.json`
- Execução: `outputs/coordination_pilot_20261007/p5_confirmation/confirm_n64_s123-456-789-1011-1213-20260927_4cells_37d61e44/`
- Integridade e reanálise do desfecho primário: `audit.json`
- Métricas secundárias equal-agent: `analysis.json`, `secondary_family_bootstrap.csv`
- Predições por agente: `agent_metrics.csv`; logits brutos por seed/posição/condição/K em NPZs com sidecars SHA-256
- Scripts: `run_coordination_information_confirmation.py`, `audit_coordination_information_confirmation.py`, `analyze_coordination_information_confirmation.py`
- Figura da propagação por distância (PNG/PDF): `outputs/coordination_pilot_20261007/p5_confirmation/confirm_n64_s123-456-789-1011-1213-20260927_4cells_37d61e44/P5_LOCAL_PROPAGATION.png` e `.pdf`; gerada de `analysis.json` por `scripts/plot_coordination_local_propagation.py`. O ponto equiponderado em distância 1 é o efeito primário; os demais estratos e distância 2 são secundários.
