# Auditoria de Vulnerabilidades e Débito Técnico - Neural Browser

Esta auditoria foca em falhas críticas de implementação, bugs de concorrência e vulnerabilidades de segurança encontradas no código-fonte.

## 1. Deadlock em "Triângulo" de Processadores
- **Local**: `src/main.rs`
- **Falha**: O uso de canais limitados (`bounded`) sem uma estratégia de cancelamento ou timeout pode causar um deadlock. Se a NPU travar processando uma imagem (que é síncrona), o canal `cpu_to_npu` enche, fazendo a thread de CPU bloquear. Se a GPU tentar navegar ou enviar mensagens para a CPU enquanto esta está bloqueada, o sistema inteiro para de responder.
- **Risco**: Crítico (System Hang).

## 2. Bypass de Segurança em Redirecionamentos
- **Local**: `src/cpu/network.rs` (Função `fetch`)
- **Falha**: A validação de esquema (apenas `http/https`) ocorre apenas na URL inicial. O redirecionador automático do `ureq` pode seguir um redirecionamento 3xx para esquemas perigosos como `file://`, `gopher://` ou `javascript:`, permitindo acesso a arquivos locais ou ataques de SSRF.
- **Risco**: Alto (Vulnerabilidade de Segurança).

## 3. Stack Overflow Recursivo em Manipulação de DOM
- **Local**: `src/cpu/dom.rs` (Funções `update_depth` e `detach_subtree`)
- **Falha**: Embora o parser tenha um `MAX_NESTING_DEPTH`, funções como `set_inner_html` podem ser chamadas repetidamente para criar árvores mais profundas do que o limite do parser. Como `update_depth` e `detach_subtree` são recursivas e não possuem guardas, isso resultará em estouro de pilha (Stack Overflow).
- **Risco**: Alto (Crash do Processo).

## 4. Exaustão de Memória em `set_inner_html`
- **Local**: `src/cpu/dom.rs`
- **Falha**: Ao contrário do `parse_html` global, a função `set_inner_html` não verifica o limite `MAX_DOM_NODES`. Um script malicioso pode inserir milhões de nós via `innerHTML` até esgotar a RAM do sistema.
- **Risco**: Médio (DDoS local).

## 5. Vazamento de Memória / Performance em Imagens
- **Local**: `src/gpu/layout.rs` e `src/npu/mod.rs`
- **Falha**: O campo `image_data` (vetor raw de pixels RGBA) é clonado integralmente para cada `LayoutBox` durante cada recomputação de layout (ex: ao redimensionar a janela ou dar zoom). Isso causa picos massivos de memória e pressão desnecessária no GC/Heap.
- **Risco**: Alto (Gargalo de Performance/RAM).

## 6. Vulnerabilidade de XSS na Reconstrução de HTML
- **Local**: `src/cpu/dom.rs` (Função `reconstruct_node`)
- **Falha**: Ao reconstruir o HTML para a NPU após a execução de JS, o motor escapa atributos mas **NÃO escapa o conteúdo de texto**. Se um nó de texto contiver tags injetadas (ex: `</div><script>...`), elas serão reinseridas como HTML bruto, enganando a NPU ou futuras execuções.
- **Risco**: Alto (Injeção de Conteúdo/XSS).

## 7. Colisão de IDs na Memória Semântica
- **Local**: `src/memory/mod.rs` (Cálculo de `id`)
- **Falha**: A lógica `(epoch_secs << 16) | (seq & 0xFFFF)` causa overflow em `u32` (já que o epoch atual ~1.7B já usa mais de 16 bits). Isso resultará em IDs colidindo a cada ~18 horas de execução, corrompendo o histórico no NietzscheDB.
- **Risco**: Médio (Integridade de Dados).

## 8. Parser JSON Frágil e Inseguro
- **Local**: `src/memory/mod.rs`
- **Falha**: O parser manual de JSON não lida corretamente com sequências de escape complexas (como `\uXXXX`) e falha ao processar objetos aninhados (metadata que contém outros objetos `{}`). Isso pode levar ao truncamento de dados ou falhas de busca silenciosas.
- **Risco**: Baixo/Médio (Fragilidade).

## 9. Detecção de Charset Incompleta
- **Local**: `src/cpu/network.rs` (Função `detect_meta_charset`)
- **Falha**: A detecção baseia-se em uma conversão lossy para UTF-8 antes de buscar o charset. Se o site usar UTF-16 ou codificações legadas incompatíveis, a busca falhará ou retornará lixo, impedindo a renderização correta de sites internacionais.
- **Risco**: Médio (Compatibilidade).

## 10. Limitação de Renderização de Listas Aninhadas
- **Local**: `src/gpu/layout.rs`
- **Falha**: O motor de layout suporta apenas **um nível** de aninhamento de listas. Listas Triplas ou quádruplas (comuns em documentação técnica) serão renderizadas de forma plana ou incorreta, prejudicando a usabilidade.
- **Risco**: Baixo (Usabilidade).

---
**Conclusão**: O Neural Browser possui uma arquitetura inovadora, mas sofre de problemas graves de robustez em casos de borda e manipulação de memória. Recomenda-se a migração para parsers JSON/HTML padrão e o uso de `Arc` para dados pesados como imagens.
