# P5 — protocolo confirmatório congelado

Congelado em 07/10/2026 antes de gerar o holdout. A pergunta principal é mecanística: **uma pista vista apenas por três agentes é transmitida pelas mensagens ADMM a agentes de uma aresta de distância?** O estimando é condicional aos checkpoints existentes; não testa se treinar sem comunicação produz uma arquitetura melhor.

## Holdout e modelos

- Gerar 64 novas soluções-base, sem sobreposição exata com soluções em datasets, desenvolvimento ou pilotos anteriores.
- Construir variantes aninhadas em 32, 40, 48 e 56 pistas, com distribuições equilibradas e concentradas. A confirmação primária usa somente 40 pistas concentradas.
- Preservar 20 células ocultas comuns por família. Antes de executar qualquer modelo, selecionar quatro dessas células por família com gerador determinístico e guardar índices no manifesto de avaliação.
- Usar os checkpoints interno e terminal de seis seeds: 123, 456, 789, 1011, 1213 e 20260927. Pesos EMA quando disponíveis, conforme protocolo de carregamento P4.
- A solução completa não é fornecida aos modelos. O valor da célula-oráculo é inserido só nas vistas de seus agentes de linha, coluna e caixa. Cada uma das quatro pistas é uma intervenção isolada, sem atualização de tabuleiro compartilhado.

## Experimento primário de transmissão

Cruzar, para cada célula-oráculo, as condições sem pista e com pista local, sob mensagens ADMM ativas e sob mensagens ADMM bloqueadas com CG local. Observar K=1, 2 e 4. K=2 é o único horizonte primário; K=1 verifica ausência de efeito antes da primeira troca e K=4 mede persistência.

O desfecho por intervenção é a média do log-probabilidade do dígito correto nas posições vazias das vistas de agentes não incidentes a distância exatamente 1. Agentes incidentes são excluídos. O estimando primário é:

`[(com pista − sem pista) com ADMM ativo] − [(com pista − sem pista) com CG local]`, em K=2.

Primeiro, agregar slots locais por agente e agentes por intervenção; depois, agregar igualmente as quatro células, os checkpoints e as duas posições da regra dentro da família. A unidade independente de reamostragem é a solução-base. Mostrar também estimativas por seed e por posição da regra. Usar bootstrap por família com 20.000 reamostragens e intervalo de 95% percentil. Não tratar slots, pistas ou checkpoints como soluções independentes.

Desfechos secundários pré-especificados: acurácia top-1 nas mesmas posições, fração de classes top-1 alteradas, divergência Jensen–Shannon, magnitude de mudança de logit e efeitos nas distâncias 0 e 2. Holm será aplicado aos testes secundários inferenciais. K=1 e K=4 são diagnósticos temporais secundários.

Auditoria necessária: em CG local, logits e previsões remotas devem ser idênticos entre a condição com pista e sem pista; sob ADMM ativo em K=1, os agentes remotos também devem permanecer idênticos. Falha nessa auditoria interrompe a análise confirmatória até correção e novo congelamento.

## Experimento funcional secundário

Nas mesmas famílias, comparar inferência direta e decimação sob comunicação completa, máscaras aleatórias de 75% e 50%, e sem mensagens ADMM com CG local. Usar três máscaras aleatórias determinísticas e independentes por fração, pareadas entre direto e decimação. Fixar K=20, até oito passadas, fração máxima de compromisso de 25% por passada e limiar calibrado somente nas famílias de desenvolvimento. Máscaras periódicas ficam fora deste protocolo.

Relatar como secundários: taxa exata e acurácia nas células ocultas após decimação; ganho sobre inferência direta com orçamento de passadas pareado; diferença desse ganho entre comunicação completa e 50%; commits certos e errados; atualizações de vistas por commit; payload separado de projeções vetoriais e reduções globais do CG. O controle sem mensagens bloqueia mensagens ADMM e CG global, mas ainda transmite os commits pelo tabuleiro compartilhado.

## Geração, retomada e análise

- Dataset novo em `datasets/sudoku_coordination_confirmation_20261007/`; não sobrescrever dados P1–P4. Seed de geração: 20261008. Recontar unicidade da base e de cada variante, auditar as quatro células ocultas e salvar SHA-256 de cada checkpoint e artefato.
- Congelar hash do código do runner e hash do manifesto antes da primeira inferência confirmatória. Alteração após a primeira saída válida exige nova versão de protocolo e novo diretório de execução.
- Salvar atomicamente por família × célula × checkpoint × condição × horizonte. Cada chave inclui hashes do código, configuração, dataset e checkpoint. Retomar apenas artefatos com proveniência e checksum idênticos.
- Executar primeiro um smoke test em dados de desenvolvimento, verificar equivalência entre lote contínuo e retomado, auditorias de isolamento e tempo/memória. Nenhum resultado do desenvolvimento altera o estimando, métricas ou exclusões.
- Rodar o holdout em blocos de até oito famílias. Interrupção ocorre ao terminar a tarefa corrente; o estado salvo permite retomar sem duplicar inferências. A análise confirmatória só começa com todos os blocos completos.
- Publicar todas as estimativas e intervalos, inclusive resultados nulos ou contrários à hipótese. Resultados de pista × número/distribuição de pistas e ramificação permanecem estudos posteriores, sem entrar no teste primário.
