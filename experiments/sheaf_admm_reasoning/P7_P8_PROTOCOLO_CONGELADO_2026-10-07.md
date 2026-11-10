# Protocolo congelado P7–P8

Data do congelamento: 07/10/2026. Escopo: confirmar os canais de coordenação e a moderação pela informação inicial, sem escolher condições pelo desempenho no holdout.

## Populações, partição e checkpoints

Gerar 136 famílias novas e únicas, excluindo soluções exatas já presentes em `datasets/`. Reservar famílias 0–7 para desenvolvimento técnico, 8–71 para P7 e 72–135 para P8. Cada família contém máscaras aninhadas de 32, 40 e 48 pistas em distribuição equilibrada e concentrada e 20 células ocultas comuns. A solução deve ser única em cada máscara. As partições P7 e P8 não se sobrepõem.

Usar checkpoints já auditados dos seis seeds `123, 456, 789, 1011, 1213, 20260927`, nos dois tipos de modelo `internal` e `terminal`. Treino, arquitetura, `K=20`, regras do Sudoku e solver ficam fixos. Os labels entram apenas na calibração/análise, nunca na inferência.

Calibrar limiar de compromisso por checkpoint e tipo de modelo na tarefa de proprietário de linha, usando somente as 32 famílias de desenvolvimento do P5, 40 pistas concentradas. Aplicar o menor limiar cuja precisão observada das propostas de dígitos corretos seja pelo menos 0,995; se nenhum limiar atingir o alvo, não aceitar propostas naquele checkpoint e reportar cobertura zero. O conjunto é usado apenas para calibração; nenhuma conclusão confirmatória será calculada nele.

## Protocolo comum de inferência e compromissos

Os 27 agentes são as nove vistas de linha, nove de coluna e nove de caixa. Cada célula tem três vistas incidentes. Para evitar um agregador de previsões como canal de coordenação oculto, a vista de linha `r` é a única autorizada a propor dígitos nas células da linha `r`. Cada linha seleciona propostas independentemente: classe top-1 entre dígitos 1–9, confiança acima do limiar calibrado, legal na linha considerando as pistas e compromissos que aquela vista recebeu, ordem decrescente de confiança e desempate por coluna; até 25% das células vazias daquela linha por passada. Não há filtro central por conflitos em colunas ou caixas. Colisões são registradas como resultado, não usadas para escolher propostas.

Todos os agentes inferem simultaneamente antes de aplicar qualquer proposta da passada. Commit local altera somente a vista de linha proponente. Commit global envia o dígito às três vistas incidentes. As propostas são mensagens discretas e têm destinatário registrado. O modelo é novamente chamado a cada passada, como em P5; seu estado ADMM interno não é carregado entre passadas. Terminar após oito passadas, sem parar por resultado. Inferência direta não cria compromissos.

O controle sem ADMM usa a política `none` e CG local, sem redução de estado entre agentes. Uma auditoria técnica perturba uma única vista e exige logits invariantes nas demais vistas com ADMM desligado. A troca `local/global` deve alterar exatamente os destinatários de cada commit. A inferência é medida antes da aplicação dos commits de cada passada.

## P7 — efeito dos canais

Executar nas 64 famílias P7, com 40 pistas concentradas e as 20 células comuns. Condições confirmatórias:

1. Decimação: ADMM ligado × commits globais.
2. Decimação: ADMM desligado × commits globais.
3. Decimação: ADMM ligado × commits locais.
4. Decimação: ADMM desligado × commits locais.
5. Referência direta: ADMM ligado, sem commit.
6. Referência direta: ADMM desligado, sem commit.

Desfecho primário: log-probabilidade média do dígito correto nas 20 células comuns e nas três vistas incidentes. Para cada célula/vista, usar o último score antes de ela receber um commit; se nunca receber, usar o score da oitava passada. Assim, compromissos não viram probabilidades artificiais iguais a um. Primeiro tirar a média igualmente entre células/vistas, posições e checkpoints dentro da família; família é a unidade independente.

Contrastes confirmatórios primários: ligado versus desligado com commits globais; ligado versus desligado com commits locais. Estimar diferença pareada por família, IC bootstrap por família de 95% com 20.000 reamostragens e controlar a família desses dois testes por Holm a 5%. Reportar estimativas/IC ainda que os testes não sejam conclusivos. Broadcast, interação, acurácia, resolução, commits corretos/errados, colisões e custo por canal são secundários.

## Tamanho fixado e justificativa de precisão

O holdout P7 fica fixado em 64 famílias antes de sua abertura. Como referência de escala, os 64 efeitos por família do contraste de log-probabilidade local de P5 têm desvio-padrão entre famílias de 0,00148; uma aproximação normal daria meia-largura de 95% de cerca de 0,00036 com 64 famílias. O estimando P7 inclui mensagens e compromissos e pode ter variância diferente; este cálculo é apenas uma checagem de compatibilidade, não uma alegação de poder formal. O piloto técnico não poderá alterar N com base no efeito observado. P8 também fica fixado em 64 famílias novas; sua variável primária é acurácia, portanto a precisão será apresentada com seus intervalos por família, sem extrapolar a escala de P5.

## P8 — moderação pela quantidade e distribuição das pistas

Executar nas 64 famílias P8, em desenho pareado de 32/40/48 pistas × equilibrada/concentrada × comunicação 100%/50% × inferência direta/decimação. Para 50%, usar três máscaras fixas (seeds 391, 773, 1237), compartilhadas entre métodos e níveis pareados; médias de máscaras ficam dentro da família. Usar commits globais e o limiar congelado em P7. As mesmas 20 células de avaliação permanecem ocultas em todos os níveis.

Desfecho primário: acurácia pré-commit do dígito correto nas três vistas de cada célula comum, conforme a regra de congelamento definida acima. Estimando: interação entre nível de pistas (48 menos 32) e comunicação (100% menos 50%) sob decimação, estimada separadamente para as distribuições equilibrada e concentrada. Corrigir os dois testes por Holm a 5%. Reportar inferência direta, nível 40, log-probabilidade, exatidão, conflitos e caracterização de singles/candidatos como secundários.

## Integridade, parada e análise

Congelar antes do holdout: código, configuração, hashes de dados/checkpoints, calibração, lista de condições, estimandos, exclusões, seeds e script de análise. Cada passada grava NPZ e sidecar de hash atomicamente, incluindo as 27 vistas locais, estado de compromissos, scores pré-commit e RNG. Retomada valida hash/configuração e carrega apenas passadas consecutivas válidas. Ctrl+C ou um arquivo `PAUSE` interrompe no limite da passada; não marca uma condição como completa até todas as suas passadas estarem auditadas.

Antes do holdout, rodar piloto técnico nas oito famílias de desenvolvimento, testar vazamento e fazer execução curta com pausa/retomada contra uma execução sem interrupção. Dimensionar recursos e registrar duração por passada, memória e armazenamento. Não usar GPU paga sem conferir crédito/preço e registrar um teto específico. O holdout só é analisado após auditoria completa de pareamento, completude, hashes, destinatários, mensagens, retenção do score pré-commit e inexistência de entrada dos labels.

Nenhuma parada por significância, aumento de amostra após olhar o holdout ou seleção de checkpoint/posição pelo resultado. Erros no protocolo são documentados; famílias afetadas não são substituídas depois de ver seus labels/resultados. Payload estimado não será chamado de tráfego medido, e tempo com compilação não sustenta alegação de velocidade.
